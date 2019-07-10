## Introduction


**<span style="color:red">New:<span>** Interactive demo using Google Colaboratory can be found [here](https://colab.research.google.com/drive/1cws1INmucRJ702eV4sKNJHzMDvrkg_lh)

TTS-Cube is an end-2-end speech synthesis system that provides a full processing pipeline to train and deploy TTS models.
   
It is entirely based on neural networks, requires no pre-aligned data and can be trained to produce audio just by using character or phoneme sequences.

Markdown does not allow embedding of audio files. For a better experience [check-out the project's website](https://tiberiu44.github.io/TTS-Cube/).

For installation please follow [these instructions](GETTING_STARTED.md). 
Training and usage examples can be found [here](TRAINING.md). 
A notebook demo can be found [here](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/tts-colab-demo.ipynb). 

## Output examples

**Encoder outputs:**

*"Arată că interesul utilizatorilor de internet față de acțiuni ecologiste de genul Earth Hour este unul extrem de ridicat."* 
![encoder_output_1](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/encoder/anca_dcnews_0023.png "Encoder output example 1")

*"Pentru a contracara proiectul, Rusia a demarat un proiect concurent, South Stream, în care a încercat să atragă inclusiv o parte dintre partenerii Nabucco."*
![encoder_output_2](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/encoder/anca_dcnews_0439.png "Encoder output example 2")


**Vocoder output (conditioned on gold-standard data)**

**Note**: The mel-spectrum is computed with a frame-shift of 12.5ms. This means that Griffin-Lim reconstruction produces sloppy results at most (regardless on the number of iterations)

[original](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0127.orig.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0127.mp3)

[original](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0439.orig.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0439.mp3)

[original](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0925.orig.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/vocoder/anca_dcnews_0925.mp3)

## End to end decoding

The encoder model is still converging, so right now the examples are still of low quality. We will update the files as soon as we have a stable Encoder model. 

[synthesized](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0023.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [original(unseen)](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0023.orig.mp3)

[synthesized](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0810.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [original(unseen)](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0810.orig.mp3)

[synthesized](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0852.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [original(unseen)](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0852.orig.mp3)

[synthesized](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0001.mp3) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [original(unseen)](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0001.orig.mp3)

## Technical details
 
TTS-Cube is based on concepts described in Tacotron (1 and 2), Char2Wav and WaveRNN, but it's architecture does not stick to the exact recipes:

- It has a dual-architecture, composed of (a) a module (Encoder) that converts sequences of characters or phonemes into mel-log spectrogram and (b) a RNN-based Vocoder that is conditioned on the spectrogram to produce audio
- The Encoder is similar to those proposed in Tacotron [(Wang et al., 2017)](http://bengio.abracadoudou.com/cv/publications/pdf/wang_2017_arxiv.pdf) and Char2Wav [(Sotelo et al., 2017)](https://openreview.net/pdf?id=B1VWyySKx), but 
    - has a lightweight architecture with just a two-layer BDLSTM encoder and a two-layer LSTM decoder
    - uses the guided attention trick [(Tachibana et al., 2017)](https://arxiv.org/pdf/1710.08969), which provides incredibly fast convergence of the attention module (in our experiments we were unable to reach an acceptable model without this trick)
    - does not employ any CNN/pre-net or post-net
    - uses a simple highway connection from the attention to the output of the decoder (which we observed that forces the encoder to actually learn how to produce the mean-values of the mel-log spectrum for particular phones/characters)
- The initail vocoder was similar to WaveRNN[(Kalchbrenner et al., 2018)](https://arxiv.org/pdf/1802.08435), but instead of modifying the RNN cells (as proposed in their paper), we used two coupled neural networks
- We are now using [Clarinet (Ping et al., 2018)](https://arxiv.org/abs/1807.07281)

    
## References

The ParallelWavenet/ClariNet code is adapted from this [ClariNet repo](https://github.com/ksw0306/ClariNet).    

