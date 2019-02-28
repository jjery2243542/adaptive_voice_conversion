import torch
from torch.utils.data import Dataset
import os 
import pickle 
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

class CollateFn(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def make_frames(self, tensor):
        out = tensor.view(tensor.size(0), tensor.size(1) // self.frame_size, self.frame_size * tensor.size(2))
        out = out.transpose(1, 2)
        return out 

    def __call__(self, l):
        data_tensor = torch.from_numpy(np.array(l)).transpose(0, 1)
        segment, segment_pos, segment_neg = [self.make_frames(element) for element in data_tensor]
        return segment, segment_pos, segment_neg

def get_data_loader(dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False):
    _collate_fn = CollateFn(frame_size=frame_size) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

class PickleDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t1, t2, neg_utt_id, t_neg = self.indexes[ind]
        segment = self.data[utt_id][t1:t1 + self.segment_size]
        segment_pos = self.data[utt_id][t2:t2 + self.segment_size]
        segment_neg = self.data[neg_utt_id][t_neg:t_neg + self.segment_size]
        return segment, segment_pos, segment_neg

    def __len__(self):
        return len(self.indexes)

