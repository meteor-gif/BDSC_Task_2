import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import json
import tqdm

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['item_share_train_info.json', 'user_info.json', 'item_info.json']

    @property
    def processed_file_names(self):
        return ['data.pt', 'userid2num.json', 'usernum2id.json', 'itemid2num.json', 'itemnum2id.json']


    def process(self):
        inviter_sequence = []
        item_sequence = []
        voter_sequence = []
        timestamp_sequence = []
        userid2num = {}
        usernum2id = {}
        itemid2num = {}
        itemnum2id = {}

        with open(self.raw_paths[0], 'r') as f:
            item_share = json.load(f)

        with open(self.raw_paths[1], 'r') as f:
            user_info = json.load(f)

        with open(self.raw_paths[2], 'r') as f:
            item_info = json.load(f)
        
        for user in user_info:
            user_id = user['user_id']
            if user_id not in userid2num:
                userid2num[user_id] = len(userid2num)
                usernum2id[str(len(usernum2id))] = user_id

        for item in item_info:
            item_id = item['item_id']
            if item_id not in itemid2num:
                itemid2num[item_id] = len(itemid2num)
                itemnum2id[str(len(itemnum2id))] = item_id

        print(len(item_share))
        for share in item_share:
            inviter_sequence.append(int(userid2num[share['inviter_id']]))
            voter_sequence.append(int(userid2num[share['voter_id']]))
            item_sequence.append(int(itemid2num[share['item_id']]))
            timestamp_sequence.append(share['vote_time'])

        inviter_array = np.array(inviter_sequence) 
        voter_array = np.array(voter_sequence) 
        item_array = np.array(item_sequence)

        with open(self.processed_paths[1], 'w') as f:
            json.dump(userid2num, f)

        with open(self.processed_paths[2], 'w') as f:
            json.dump(usernum2id, f)

        with open(self.processed_paths[3], 'w') as f:
            json.dump(itemid2num, f)

        with open(self.processed_paths[4], 'w') as f:
            json.dump(itemnum2id, f)
        
        num_users = len(userid2num)
        users_dim = 64
        x = torch.from_numpy(np.random.rand(num_users, users_dim).astype(np.float32))
        
        train_index = int(len(item_share) * 0.8)
        
        train_inviters = inviter_array[:train_index]
        train_voters = voter_array[:train_index]

        validation_inviters = inviter_array[train_index:]
        validation_voters = voter_array[train_index:]
        
        train_edge_index = torch.tensor([list(train_inviters) + list(train_voters), list(train_voters) + list(train_inviters)], dtype=torch.long)
        train_edge_label_index = torch.tensor([list(train_inviters), list(train_voters)], dtype=torch.long)
        train_negative_index = np.random.randint(0, len(np.unique(train_voters)), train_edge_label_index.size(1))
        train_negative_edge_label_index = torch.tensor([list(train_inviters), list(train_voters[train_negative_index])], dtype=torch.long)
        train_data = Data(x=x, edge_index=train_edge_index, 
                          edge_label_index=train_edge_label_index, negative_edge_label_index=train_negative_edge_label_index)


        validation_edge_index = torch.tensor([list(validation_inviters) + list(validation_voters), list(validation_voters) + list(validation_inviters)], dtype=torch.long)
        validation_edge_label_index = torch.tensor([list(validation_inviters), list(validation_voters)], dtype=torch.long)
        validation_negative_index = np.random.randint(0, len(np.unique(voter_array)), validation_edge_label_index.size(1))
        validation_negative_edge_label_index = torch.tensor([list(validation_inviters), list(voter_array[validation_negative_index])], dtype=torch.long)
        validation_data = Data(x=x, edge_index=validation_edge_index,
                               edge_label_index=validation_edge_label_index, negative_edge_label_index=validation_negative_edge_label_index)
        
        data_list = [train_data, validation_data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
