stage=0
segment_size=64
data_dir=/groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/
raw_data_dir=/groups/jjery2243542/data/raw/LibriTTS/
test_prop=0.05
n_samples=5000
training_samples=10000000
testing_samples=10000

twice_segment_size=$(( $segment_size * 2 ))

if [ $stage -le 0 ]; then
    python3 make_datasets_libri.py $raw_data_dir/ $data_dir $test_prop $n_samples
fi

if [ $stage -le 1 ]; then
    python3 reduce_dataset.py $data_dir/train.pkl $data_dir/train_$twice_segment_size.pkl $twice_segment_size
fi

if [ $stage -le 2 ]; then
    # sample training samples
    python3 sample_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python3 sample_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python3 sample_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
