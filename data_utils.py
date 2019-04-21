import torch
from torch.utils.data import Dataset
import os 
import pickle 
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
'''
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

def get_speaker_ids(train_index_path, test_index_path, output_dir):
    indexes = json.load(open(train_index_path))
    utt2sid = {}
    speaker2sid = {}
    for index in indexes:
        utt_id = index[0]
        speaker_id = utt_id[:4]
        if speaker_id not in speaker2sid:
            speaker2sid[speaker_id] = len(speaker2sid)
        utt2sid[utt_id] = speaker2sid[speaker_id]
    indexes = json.load(open(test_index_path))
    for index in indexes:
        utt_id = index[0]
        speaker_id = utt_id[:4]
        utt2sid[utt_id] = speaker2sid[speaker_id]
    with open(os.path.join(output_dir, 'utt2sid.json'), 'w') as f:
        json.dump(utt2sid, f)
    with open(os.path.join(output_dir, 'speaker2sid.json'), 'w') as f:
        json.dump(speaker2sid, f)
    
class ClaDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size, utt2sid_path, speaker2sid_path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        with open(utt2sid_path, 'r') as f:
            self.utt2sid = json.load(f)
        with open(speaker2sid_path, 'r') as f:
            self.speaker2sid = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t, _, _ = self.indexes[ind]
        segment = self.data[utt_id][t:t + self.segment_size]
        speaker_id = self.utt2sid[utt_id]
        return segment, speaker_id 

    def __len__(self):
        return len(self.indexes)

def get_cla_data_loader(dataset, batch_size, shuffle=True):
    def _collate_fn(l):
        data = torch.from_numpy(np.array([a for a, _ in l])).transpose(1, 2)
        target = torch.from_numpy(np.array([b for _, b in l]))
        return data, target
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

def get_cla_data_loader(dataset, batch_size, shuffle=True):
    def _collate_fn(l):
        data = torch.from_numpy(np.array([a for a, _ in l])).transpose(1, 2)
        target = torch.from_numpy(np.array([b for _, b in l]))
        return data, target
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

class ClaDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size, utt2sid_path, speaker2sid_path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        with open(utt2sid_path, 'r') as f:
            self.utt2sid = json.load(f)
        with open(speaker2sid_path, 'r') as f:
            self.speaker2sid = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t, _, _ = self.indexes[ind]
        segment = self.data[utt_id][t:t + self.segment_size]
        speaker_id = self.utt2sid[utt_id]
        return segment, speaker_id 

    def __len__(self):
        return len(self.indexes)
'''
class CollateFn(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def make_frames(self, tensor):
        out = tensor.view(tensor.size(0), tensor.size(1) // self.frame_size, self.frame_size * tensor.size(2))
        out = out.transpose(1, 2)
        return out 

    def __call__(self, l):
        data_tensor = torch.from_numpy(np.array(l)).transpose(0, 1)
        segment, segment_cond = [self.make_frames(element) for element in data_tensor]
        return segment, segment_cond

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
        utt_id, t1, t2, _, _ = self.indexes[ind]
        segment = self.data[utt_id][t1:t1 + self.segment_size]
        segment_cond = self.data[utt_id][t2:t2 + self.segment_size]
        return segment, segment_cond

    def __len__(self):
        return len(self.indexes)

