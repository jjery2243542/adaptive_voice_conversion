. libri.config

if [ $stage -le 0 ]; then
    python3 make_datasets_libri.py $raw_data_dir/ $data_dir $test_prop $n_utts_attr $train_set $test_set
fi

if [ $stage -le 1 ]; then
    python3 reduce_dataset.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl 
fi

if [ $stage -le 2 ]; then
    # sample training samples
    python3 sample_single_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python3 sample_single_segments.py $data_dir/dev.pkl $data_dir/dev_samples_$segment_size.json $testing_samples $segment_size
    python3 sample_single_segments.py $data_dir/test.pkl $data_dir/test_samples_$segment_size.json $testing_samples $segment_size
fi
