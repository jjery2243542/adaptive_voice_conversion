stage=1
if [ $stage -le 0 ]; then
    python3 make_datasets.py /storage/datasets/VCTK/VCTK-Corpus/wav48 /storage/datasets/VCTK/VCTK-Corpus/speaker-info.txt /storage/feature/voice_conversion/vctk_wavform 10 0.1 22050
fi

if [ $stage -le 1 ]; then
    # sample training samples
    python3 sample_segments.py /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/train.pkl /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/train_samples.json 1000000 8000
    # sample testing samples
    python3 sample_segments.py /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/in_test.pkl /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/in_test_samples.json 1000 8000
    python3 sample_segments.py /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/out_test.pkl /storage/feature/voice_conversion/vctk_waveform/librosa/waveform/out_test_samples.json 1000 8000
fi
