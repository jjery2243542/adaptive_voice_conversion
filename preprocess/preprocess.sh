stage=2
#output_dir=/groups/jjery2243542/data/vctk/trimmed_vctk_wavform
output_dir=/storage/feature/voice_conversion/trimmed_vctk_waveform/librosa/split_10_0.1
if [ $stage -le 0 ]; then
    python3 make_datasets.py /groups/jjery2243542/data/raw/VCTK-Corpus/wav48 /groups/jjery2243542/data/raw/VCTK-Corpus/speaker-info.txt /groups/jjery2243542/data/vctk/trimmed_vctk_wavform 10 0.1 22050
fi

if [ $stage -le 1 ]; then
    # sample training samples
    python3 sample_segments.py /groups/jjery2243542/data/vctk/trimmed_vctk_wavform/train.pkl /groups/jjery2243542/data/vctk/trimmed_vctk_wavform/train_samples_4000.json 10000000 4000
fi
if [ $stage -le 2 ]; then
    # sample testing samples
    python3 sample_segments.py $output_dir/in_test.pkl $output_dir/in_test_samples_4000.json 10000 4000
    python3 sample_segments.py $output_dir/out_test.pkl $output_dir/out_test_samples_4000.json 10000 4000
fi
