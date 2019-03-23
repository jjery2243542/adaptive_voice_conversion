stage=1
#segment_size=16384
segment_size=64
data_dir=/groups/jjery2243542/data/vctk/trimmed_vctk_spectrograms/sr_24000_mel/
raw_data_dir=/groups/jjery2243542/data/raw/VCTK-Corpus
n_out_speakers=20
test_prop=0.1
sample_rate=24000
training_samples=10000000
testing_samples=10000

if [ $stage -le 0 ]; then
    python3 make_datasets.py $raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $sample_rate
fi

if [ $stage -le 1 ]; then
    # sample training samples
    python3 sample_single_segments.py $data_dir/train.pkl $data_dir/train_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 2 ]; then
    # sample testing samples
    python3 sample_single_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python3 sample_single_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
