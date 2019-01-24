import json 
import pickle 
import sys
import os
import random


if __name__ == '__main__':
    pickle_path = sys.argv[1]
    sample_path = sys.argv[2]
    n_samples = int(sys.argv[3])
    segment_size = int(sys.argv[4])

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # (utt_id, timestep_1, timestep_2, neg_utt_id, neg_timestep)
    samples = []

    # filter length > 2 * segment_size
    utt_list = [key for key in data]
    utt_list = sorted(list(filter(lambda u : len(data[u]) > 2 * segment_size, utt_list)))
    print(f'{len(utt_list)} utterances')
    sample_utt_index_list = random.choices(range(len(utt_list)), k=n_samples)

    for i, utt_ind in enumerate(sample_utt_index_list):
        if i % 500 == 0:
            print(f'sample {i} samples')
        pos_utt_id = utt_list[utt_ind]
        neg_utt_id = random.choice(utt_list[:utt_ind] + utt_list[utt_ind + 1:])
        t1 = random.randint(0, len(data[pos_utt_id]) - 2 * segment_size)
        t2 = random.randint(t1 + segment_size, len(data[pos_utt_id]) - segment_size)
        # random swap t1, t2
        t1, t2 = random.sample([t1, t2], k=2)
        t_neg = random.randint(0, len(data[neg_utt_id]) - segment_size)
        samples.append((pos_utt_id, t1, t2, neg_utt_id, t_neg))

    with open(sample_path, 'w') as f:
        json.dump(samples, f)

