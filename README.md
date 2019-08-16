# One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization
This is the official repository for the paper [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742).
By separately learning speaker and content representations, we can achieve one-shot VC by only one utterance from source speaker and one utterace from target speaker. 
<img src="https://github.com/jjery2243542/adaptive_voice_conversion/blob/public/model.png" width="400" img align="right">
# Dependencies
- python 3.6+
- pytorch 1.0.1
- numpy 1.16.0
- librosa 0.6.3
- SoundFile 0.10.2 <br>
We also use some preprocess script from [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron) and [magenta/magenta/models/gansynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/gansynth).

# Differences from the paper
The implementations are a little different from the paper, which I found them useful to stablize training process or improve audio quality. However, the experiments requires human evaluation, we only update the code but not update the paper. The differences are listed below: 
- Not to apply dropout to the speaker encoder and content encoder.
- Normalization placed at pre-activation position.
- Use the original KL-divergence loss for VAE rather than unit variance version.
- Use KL annealing, and the weight will increase to 1. 

# Preprocess
We provide the preprocess script for two datasets. The links are below.
- [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
- [LibriTTS Corpus](http://www.openslr.org/60/)

The preprocess code is at *preprocess/*



# Cite us
Please cite our paper if you find this repository useful.
```
@article{chou2019one,
  title={One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization},
  author={Chou, Ju-chieh and Yeh, Cheng-chieh and Lee, Hung-yi},
  journal={arXiv preprint arXiv:1904.05742},
  year={2019}
}
```

