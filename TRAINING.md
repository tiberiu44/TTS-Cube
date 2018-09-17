# Training a new TTS model

This is a short tutorial on how to build your TTS model using your own data.

For some languages you could train models to use only character-level features. However, Grapheme-to-Phoneme (G2P) conversion before training helps build robust models, since G2P lexicons contain far more training examples than the speech corpus itself. As such, we recommend training a G2P lexicon if you want to English TTS or use our pre-trained G2P model on CMUDICT.  

In order to train a new TTS model you need to follow these steps:

## Step 0 (optional) - Train a G2P model

In order to train a phonetic transcription model you need a lexicon split into a training and a development set. Optionally, you can also use a separate test set and evaluate the model after the training process is done. However, unless you are writing a report/paper on this there is no reason to keep training data aside for this particular reason.

The lexicon should be in a format similar to [CMUDICT](https://github.com/cmusphinx/cmudict). Each line represents a single entry. Tokenization is done by splitting over `SPACE` or `TAB`. The first token is the word and the next tokens represent the symbols used for phonetic transcription:

```text
test T EH1 S T
tested T EH1 S T IH0 D
tester T EH1 S T ER0
```  

The order of the words is irrelevant. Also, the numbers used to represent lexical accent (0, 1, ...) are stripped away by the model during training. This means that **you should not use numbers as symbols for phonetic transcription**.

Given that you have named your train and development sets `en-train` and respectively `en-dev`, you can use the following command to build your model (*in the TTS-Cube folder*):

```bash
python3 cube/g2p.py --train --train-file=en-train --dev-file=en-dev --batch-size=100 --store=data/models/en-g2p --patience=20 
```

Your output should look something like this:

```text
[dynet] random seed: 9
[dynet] allocating memory: 2048MB
[dynet] memory allocation done.
Loading datasets...
Trainset has 121581 entries
Devset has 13510 entries
Found 32 characters
Found 45 phonemes
Starting epoch 1
    training... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 avg loss=20.143797003812338 execution time 885.804271697998
    evaluating... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 word accuracy=0.00037009622501849027 and phone edit distance=0.8222559140264432
    Storing data/models/g2p-en-bestAcc.network
    Storing data/models/g2p-en-last.network
Starting epoch 2
    training... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 avg loss=15.406379161236098 execution time 906.7657270431519
    evaluating... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 word accuracy=0.005847520355292346 and phone edit distance=0.616158040076471
    Storing data/models/g2p-en-bestAcc.network
    Storing data/models/g2p-en-last.network
Starting epoch 3
    training... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 avg loss=10.685976136422088 execution time 936.7018821239471
    evaluating... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 word accuracy=0.1061435973353072 and phone edit distance=0.34612406435515786
    Storing data/models/g2p-en-bestAcc.network
    Storing data/models/g2p-en-last.network
Starting epoch 4
    training... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 avg loss=7.397071910354918 execution time 837.4504680633545
    evaluating... 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 word accuracy=0.27179866765358995 and phone edit distance=0.22312772557144
    Storing data/models/g2p-en-bestAcc.network
    Storing data/models/g2p-en-last.network
```

## Step 1 - Import the corpus

