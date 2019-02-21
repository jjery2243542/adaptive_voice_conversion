stage=1
segment_size=16384
data_dir=/groups/jjery2243542/data/vctk/trimmed_vctk_waveform/librosa/split_10_0.1/sr_22050
raw_data_dir=/groups/jjery2243542/data/raw/VCTK-Corpus
n_out_speakers=10
test_prop=0.1
sample_rate=22050
training_samples=10000000
testing_samples=10000

if [ $stage -le 0 ]; then
    python3 make_datasets.py $raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $sample_rate
fi

if [ $stage -le 1 ]; then
    # sample training samples
    python3 sample_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 2 ]; then
    # sample testing samples
    python3 sample_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python3 sample_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
