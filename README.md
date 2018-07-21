# Introduction

TTS-Cube is an end-2-end speech synthesis system that provides a full processing pipeline to train and deploy TTS models.
   
It is entirely based on neural networks, requires no pre-aligned data and can be trained to produce audio just by using character or phoneme sequences.

# Output examples

Encoder outputs:
 
![alt text](https://github.com/tiberiu44/TTS-Cube/examples/ "Logo Title Text 1")


#Technical details

TTS-Cube is based on concepts described in Tacotron (1 and 2), Char2Wav and WaveRNN, but it's architecture does not stick to the exact recipes:

- It has a dual-architecture, composed of (a) a module (Encoder) that converts sequences of characters or phonemes into mel-log spectrogram and (b) a RNN-based Vocoder that is conditioned on the spectrogram to produce audio
- The Encoder is similar to those proposed in Tacotron and Char2Wav, but 
    - has a lightweight architecture with just a two-layer BDLSTM encoder and a two-layer LSTM decoder
    - uses the guided attention trick (Tachibana et al., 2017), which provides incredibly fast convergence of the attention module (in our experiments we were unable to reach an acceptable model without this trick)
    - does not employ any CNN/pre-net or post-net
    - uses a simple highway connection from the attention to the output of the decoder (which we observed that forces the encoder to actually learn how to produce the mean-values of the mel-log spectrum for particular phones/characters)
- The Vocoder is similar to WaveRNN, but instead of modifying the RNN cells (as proposed in their paper), we used two coupled neural networks
    
    

