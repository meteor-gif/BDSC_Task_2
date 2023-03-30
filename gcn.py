import numpy as np
import random
import logging

from pip import main 
from utils import EarlyStopMonitor
import torch
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import argparse
import sys 
# import oss2
import time
import math
from dataset import MyOwnDataset
import faiss
from tqdm import tqdm
import json

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

seed = 3407
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser('Interface for SHARE experiments on link predictions')
parser.add_argument('--batch_size', type=int, default=4096, help='batch_size')
parser.add_argument('--num_epoches', type=int, default=100, help='number of epochs')
parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--users_dim', type=int, default=64, help='Dimensions of the users embedding')
parser.add_argument('--items_dim', type=int, default=64, help='Dimensions of the items embedding')
parser.add_argument('--time_dim', type=int, default=64, help='Dimensions of the time embedding')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.batch_size
NUM_EPOCHES = args.num_epoches
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NUM_LAYERS = args.num_layers
LEARNING_RATE = args.lr
USERS_DIM = args.users_dim
ITEMS_DIM = args.items_dim
TIME_DIM = args.time_dim

get_emb_path = lambda epoch: f'./saved_embs/emb_{epoch}.npy'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(int(time.time()))))))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


train_data, validation_data = MyOwnDataset('./')

device = torch.device('cuda')

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # chaining two convolutions with a standard relu activation
        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim = -1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple = False).t()

gcn = Net(USERS_DIM, USERS_DIM, USERS_DIM)
optimizer = torch.optim.Adam(gcn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()
gcn = gcn.to(device)
# out = gcn.encode(train_data.x, train_data.edge_index)


def train():
    gcn.train()
    optimizer.zero_grad()
    emb = gcn.encode(train_data.x, train_data.edge_index)

    edge_label_index = torch.cat(
        [train_data.edge_label_index, train_data.negative_edge_label_index],
        dim=-1,
    )
    edge_label = torch.cat([
        torch.ones(train_data.edge_label_index.size(1), dtype=torch.float, device=device),
        torch.zeros(train_data.negative_edge_label_index.size(1), dtype=torch.float, device=device)
    ], dim=0)

    out = gcn.decode(emb, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss, emb

def validation(emb, data, dimension, topK):
    with open('./processed/usernum2id.json', 'r') as load_f:
        usernum2id = json.load(load_f)
    sequence_len = int(data.edge_index.shape[1] / 2)
    inviters_index = data.edge_index[0, :sequence_len].data.cpu().numpy()
    voters_index = data.edge_index[0, sequence_len:].data.cpu().numpy()
    inviters_emb = emb[inviters_index]
    indexL2 = faiss.IndexFlatL2(dimension)
    indexL2.add(emb)
    distance, topKindex = indexL2.search(inviters_emb, topK)
    rank_list = []
    for i in tqdm(range(len(voters_index))):
        voter = voters_index[i]
        topKlist = list(topKindex[i, :])
        if voter in topKlist:
            rank = topKlist.index(voter)
            rank_list.append(1.0 / (rank + 1))
        else:
            rank_list.append(0)

    mrr = np.mean(rank_list)
    return mrr

def generate_submission(emb, dimension, topK, inviters_index):
    with open('./processed/usernum2id.json', 'r') as load_f:
        usernum2id = json.load(load_f)
    inviters_emb = emb[inviters_index]
    indexL2 = faiss.IndexFlatL2(dimension)
    indexL2.add(emb)
    distance, topKindex = indexL2.search(inviters_emb, topK)
    rank_list = []
    submission_A = []
    for i in tqdm(range(len(inviters_index))):
        topKlist = list(topKindex[i, :])
        candidate_voter_list = [usernum2id[str(top_voter_index)] for top_voter_index in topKlist]
        submission_A.append({'triple_id': str('%06d' % i), 'candidate_voter_list': candidate_voter_list})

    with open('submission_A.json', 'w') as f:
        json.dump(submission_A, f)


with open('./raw/A_for_users.json', 'r') as f:
    competition_A = json.load(f)

with open('./processed/userid2num.json', 'r') as load_f:
        userid2num = json.load(load_f)

A_inviters_index = []
for line in competition_A:
    A_inviters_index.append(int(userid2num[line['inviter_id']]))
early_stopper = EarlyStopMonitor()
topK = 5

for epoch in range(0, NUM_EPOCHES):
    loss, emb = train()
    emb = emb.data.cpu().numpy()
    val_mrr = validation(emb, validation_data, USERS_DIM, topK)
    if early_stopper.early_stop_check(val_mrr):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best embedding at epoch {early_stopper.best_epoch}')
        best_emb_path = get_emb_path(early_stopper.best_epoch)
        emb = np.load(best_emb_path)
        logger.info(f'Loaded the best embedding at epoch {early_stopper.best_epoch} for inference')
        break
    else:
        # print(emb)
        np.save(get_emb_path(epoch), emb)
        logger.info('epoch {}, loss:{},'.format(epoch, loss))
        logger.info('val_mrr: {}'.format(val_mrr))
        
generate_submission(emb, USERS_DIM, topK, np.array(A_inviters_index))

        




