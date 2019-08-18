import pickle 
import librosa 
import sys
import glob 
import random
import os
from collections import defaultdict
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms

def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = line.strip().split()[0]
            speaker_ids.append(speaker_id)
    return speaker_ids

def read_filenames(root_dir):
    speaker2filenames = defaultdict(lambda : [])
    for path in sorted(glob.glob(os.path.join(root_dir, '*/*'))):
        filename = path.strip().split('/')[-1]
        speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
        speaker2filenames[speaker_id].append(path)
    return speaker2filenames

def wave_feature_extraction(wav_file, sr):
    y, sr = librosa.load(wav_file, sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def spec_feature_extraction(wav_file):
    mel, mag = get_spectrograms(wav_file)
    return mel, mag

if __name__ == '__main__':
    data_dir = sys.argv[1]
    speaker_info_path = sys.argv[2]
    output_dir = sys.argv[3]
    test_speakers = int(sys.argv[4])
    test_proportion = float(sys.argv[5])
    sample_rate = int(sys.argv[6])
    n_utts_attr = int(sys.argv[7])

    speaker_ids = read_speaker_info(speaker_info_path)
    random.shuffle(speaker_ids)

    train_speaker_ids = speaker_ids[:-test_speakers]
    test_speaker_ids = speaker_ids[-test_speakers:]

    speaker2filenames = read_filenames(data_dir)

    train_path_list, in_test_path_list, out_test_path_list = [], [], []

    for speaker in train_speaker_ids:
        path_list = speaker2filenames[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion)
        train_path_list += path_list[:-test_data_size]
        in_test_path_list += path_list[-test_data_size:]

    with open(os.path.join(output_dir, 'in_test_files.txt'), 'w') as f:
        for path in in_test_path_list:
            f.write(f'{path}\n')

    for speaker in test_speaker_ids:
        path_list = speaker2filenames[speaker]
        out_test_path_list += path_list

    with open(os.path.join(output_dir, 'out_test_files.txt'), 'w') as f:
        for path in out_test_path_list:
            f.write(f'{path}\n')

    for dset, path_list in zip(['train', 'in_test', 'out_test'], \
            [train_path_list, in_test_path_list, out_test_path_list]):
        print(f'processing {dset} set, {len(path_list)} files')
        data = {}
        output_path = os.path.join(output_dir, f'{dset}.pkl')
        all_train_data = []
        for i, path in enumerate(sorted(path_list)):
            if i % 500 == 0 or i == len(path_list) - 1:
                print(f'processing {i} files')
            filename = path.strip().split('/')[-1]
            mel, mag = spec_feature_extraction(path)
            data[filename] = mel
            if dset == 'train' and i < n_utts_attr:
                all_train_data.append(mel)
        if dset == 'train':
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {'mean': mean, 'std': std}
            with open(os.path.join(output_dir, 'attr.pkl'), 'wb') as f:
                pickle.dump(attr, f)
        for key, val in data.items():
            val = (val - mean) / std
            data[key] = val
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

