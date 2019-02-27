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
    _, mag = get_spectrograms(wav_file)
    return mag

def my_std(data, mean):
    square_sum = 0.
    for val in data:
        square_sum += (val - mean) ** 2
    std = np.sqrt(square_sum / len(data))
    return std 

if __name__ == '__main__':
    data_dir = sys.argv[1]
    speaker_info_path = sys.argv[2]
    output_dir = sys.argv[3]
    test_speakers = int(sys.argv[4])
    test_proportion = float(sys.argv[5])
    sample_rate = int(sys.argv[6])
    feature_type = sys.argv[7]

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

    with open(os.path.join(output_dir, 'in_test_files.txt')) as f:
        for path in in_test_path_list:
            f.write(f'{path}\n')

    for speaker in test_speaker_ids:
        path_list = speaker2filenames[speaker]
        out_test_path_list += path_list

    with open(os.path.join(output_dir, 'out_test_files.txt')) as f:
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
            wav_data = feature_extraction(path, sr=sample_rate)
            data[filename] = wav_data
        #    if dset == 'train':
        #        all_train_data.append(wav_data)
        #if dset == 'train':
        #    all_train_data = np.concatenate(all_train_data)
        #    mean = all_train_data.mean()
        #    std = all_train_data.std()
        #    print(f'mean={mean:.3f}, std={std:.3f}')
        #    attr = {'mean': float(mean), 'std': float(std)}
        #    with open(os.path.join(output_dir, 'mean_std.json'), 'w') as f:
        #        json.dump(attr, f)
        #for key, value in data.items():
        #    value = (value - mean) / std
        #    data[key] = value
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

