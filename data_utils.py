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
        segment = self.make_frames(torch.from_numpy(np.array([batch[0] for batch in l])))
        segment_pos = self.make_frames(torch.from_numpy(np.array([batch[1] for batch in l])))
        segment_neg = self.make_frames(torch.from_numpy(np.array([batch[2] for batch in l])))
        return segment, segment_pos, segment_neg

def get_data_loader(dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False):
    _collate_fn = CollateFn(frame_size=frame_size) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.utt_ids = list(self.data.keys())

    def __getitem__(self, ind):
        utt_id = self.utt_ids[ind]
        ret = self.data[utt_id].transpose()
        return ret

    def __len__(self):
        return len(self.utt_ids)

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
        # negtive segment twice long
        segment_neg = self.data[neg_utt_id][t_neg:t_neg + self.segment_size * 2]
        return segment, segment_pos, segment_neg

    def __len__(self):
        return len(self.indexes)

