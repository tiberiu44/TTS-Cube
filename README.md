## Introduction

TTS-Cube is an end-2-end speech synthesis system that provides a full processing pipeline to train and deploy TTS models.
   
It is entirely based on neural networks, requires no pre-aligned data and can be trained to produce audio just by using character or phoneme sequences.

## Output examples

**Encoder outputs:**

*"Arată că interesul utilizatorilor de internet față de acțiuni ecologiste de genul Earth Hour este unul extrem de ridicat."* 
![encoder_output_1](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/encoder/anca_dcnews_0023.png "Encoder output example 1")

*"Pentru a contracara proiectul, Rusia a demarat un proiect concurent, South Stream, în care a încercat să atragă inclusiv o parte dintre partenerii Nabucco."*
![encoder_output_2](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/encoder/anca_dcnews_0439.png "Encoder output example 2")


**Vocoder output (conditioned on gold-standard data)**

**Note**: The mel-spectrum is computed with a frame-shift of 12.5ms. This means that Griffin-Lim reconstruction produces sloppy results at most (regardless on the number of iterations)

[original](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0127.orig.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0127.mp4)

[original](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0439.orig.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0439.mp3)

[original](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0925.orig.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[vocoder](https://github.com/tiberiu44/TTS-Cube/raw/master/examples/vocoder/anca_dcnews_0925.mp3)

## End to end decoding

[example 1](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0023.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[example 2](https://raw.githubusercontent.com/tiberiu44/TTS-Cube/master/examples/e2e/anca_dcnews_0810.wav)

## Technical details
 
TTS-Cube is based on concepts described in Tacotron (1 and 2), Char2Wav and WaveRNN, but it's architecture does not stick to the exact recipes:

- It has a dual-architecture, composed of (a) a module (Encoder) that converts sequences of characters or phonemes into mel-log spectrogram and (b) a RNN-based Vocoder that is conditioned on the spectrogram to produce audio
- The Encoder is similar to those proposed in Tacotron and Char2Wav, but 
    - has a lightweight architecture with just a two-layer BDLSTM encoder and a two-layer LSTM decoder
    - uses the guided attention trick (Tachibana et al., 2017), which provides incredibly fast convergence of the attention module (in our experiments we were unable to reach an acceptable model without this trick)
    - does not employ any CNN/pre-net or post-net
    - uses a simple highway connection from the attention to the output of the decoder (which we observed that forces the encoder to actually learn how to produce the mean-values of the mel-log spectrum for particular phones/characters)
- The Vocoder is similar to WaveRNN, but instead of modifying the RNN cells (as proposed in their paper), we used two coupled neural networks
    
    

