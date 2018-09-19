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

Before training the models you need to import your training data. For this, you need to split your data in two folders: one for training and one for development data.

For multi-speaker training there is a naming convention: each file must start with the name (or identifier) of the speaker followed by the '_' character and any character sequence.

Otherwise, feel free to name your files however you want, but do note that **if you use '_' in the names, TTS-Cube will consider that it is a delimiter for the speaker's name**.

Each sentence must be stored in a separate `WAVE` file. Also, it must be accompanied by a `TXT` file with the same name. If you plan on using your own custom text-features, you should add a `LAB` file with the same base name as the `WAVE` and `TXT`.

The lab file has a specific TAB-separated format:

- the first column should contain each letter/phoneme inside the utterance
- the second column should contain the speaker identity as `SPEAKER:<speaker id>`. It is mandatory that you add this column. Just use `none` as a speaker identifier if you are planning on single-speaker training
- you can have any additional number of columns with any features that you extract. Just make sure they are uniquely identifiable and they do not create confusion.

**Example:** Presume you have a `WAVE` with the sentence *"This is a test."*
An example txt file is:
```text
This is a test.
```
An example lab file is:
```text
t   SPEAKER:none    WORD:this   TYPE:character
h   SPEAKER:none    WORD:this   TYPE:character
i   SPEAKER:none    WORD:this   TYPE:character
s   SPEAKER:none    WORD:this   TYPE:character
    SPEAKER:none    TYPE:symbol
i   SPEAKER:none    WORD:is   TYPE:character
s   SPEAKER:none    WORD:is   TYPE:character
    SPEAKER:none    TYPE:symbol
a   SPEAKER:none    WORD:a   TYPE:character
    SPEAKER:none    TYPE:symbol
t   SPEAKER:none    WORD:test   TYPE:character
e   SPEAKER:none    WORD:test   TYPE:character
s   SPEAKER:none    WORD:test   TYPE:character
t   SPEAKER:none    WORD:test   TYPE:character
.   SPEAKER:none    TYPE:symbol
```

**Note:** this are fictive features. We just wanted to show that you don't need to use the same number of features (columns) for every character.

If you don't want to use any custom-defined features, just don't add any '.lab' files in your training set.

**Suggestion:** Don't use more than 5 files in the development set. It takes long enough to synthesize them and they slow down the training process. Instead, just choose the "right" ones. The results are fairly stable once the model converges. Also, we found that the loss and accuracy don't correlate with the actual quality of the produced speech. This is way we just store a snapshot of the model periodically. You should check the quality of the samples and stop training once you are satisfied.

Naming example:
```text
anca_dcnews_0001.txt
anca_dcnews_0001.wav
anca_dcnews_0023.txt
anca_dcnews_0023.wav
tss_0887.txt
tss_0887.wav
tss_0888.txt
tss_0888.wav
``` 
In the above example, we have a corpus composed of two speakers: `anca` and `tss`, with no external text features (lab files).

The `TXT` files just contain the sentence or sentences that are uttered in the `WAVE`. The whole text should appear in a single line. Punctuation and letter-casing should be untouched, but we do recommend that you normalize the training data and replace any numbers or abbreviation with their spoken form. For example, replace "1000" with "one thousand" and "dr." with "doctor".

**Note:** If you are restarting the import process for another corpus, make sure you cleanup your environment before proceeding. In the `TTS-Cube` folder type:

```bash
rm data/models/*
rm data/processed/train/*
rm data/processed/dev/*
rm data/processed/output/*
``` 

Once you have prepared the corpus, just run the following command (in the `TTS-Cube` folder):

```bash
python3 cube/trainer.py --phase=1 --train-folder=<path to your training folder> --dev-folder=<path to your development folder>
```

## Step 2 - Train the Vocoder

The following step is much easier than the corpus preparation and import process. Once you have finished with 'step 1', just type:

```bash
python3 cube/trainer.py --phase=2 --use-gpu --set-mem 8192 --autobatch --batch-size 8000
```

```text
Found 19909 training files and 374 development files
	Rendering devset
		1/374 processing file data/processed/dev/bas_rnd1_491 
(314, 60)
		2/374 processing file data/processed/dev/htm_rnd1_490 
(153, 60)
		3/374 processing file data/processed/dev/eme_rnd1_495 
(642, 60)
		4/374 processing file data/processed/dev/pcs_rnd2_495 
(244, 60)
		5/374 processing file data/processed/dev/rms_rnd2_490 
(423, 60)

Starting epoch 1
Shuffling training data
	1/19909 processing file data/processed/train/pcs_rnd2_437 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 avg loss=[2.62019864] execution time=20.65229702
	2/19909 processing file data/processed/train/tim_rnd1_466 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 avg loss=[2.80310284] execution time=20.7937879562
	3/19909 processing file data/processed/train/sgs_rnd1_259 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 avg loss=[2.07256974] execution time=20.8006060123
	4/19909 processing file data/processed/train/sgs_rnd1_060 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 avg loss=[2.56461518] execution time=35.2882008553
	5/19909 processing file data/processed/train/tss_rnd1_427 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 avg loss=[2.87694773] execution time=23.7653918266
	6/19909 processing file data/processed/train/eme_rnd3_373 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 avg loss=[2.42770364] execution time=26.72453022
	7/19909 processing file data/processed/train/dcs_rnd2_197 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 avg loss=[2.56457675] execution time=26.6598260403
...
    49/19909 processing file data/processed/train/sgs_rnd1_149 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 avg loss=[2.50035518] execution time=14.3753509521
Synthesizing devset
		1/5 processing file data/processed/dev/bas_rnd1_491/work/adriana-work/TTS-Cube_latest/TTS-Cube/cube/models/vocoder.py:132: RuntimeWarning: divide by zero encountered in log
  scaled_prediction = np.log(probs) / temperature
 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 execution time=148.720160007
		2/5 processing file data/processed/dev/htm_rnd1_490 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 execution time=63.9193780422
		3/5 processing file data/processed/dev/eme_rnd1_495 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 execution time=340.819341898
		4/5 processing file data/processed/dev/pcs_rnd2_495 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 execution time=127.656816959
		5/5 processing file data/processed/dev/rms_rnd2_490 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 execution time=212.470052004
...
```

After every 50 files in the training set, we synthesize the development set and store a model checkpoint.

Just listen to the samples produced and stop the process when you are satisfied. The model is stored in `data/models/rnn_vocoder.network`

## Step 3 - Train the Encoder

Once you have a viable Vocoder (step 2) you need to train the text-encoder:

```bash
python3 cube/trainer.py --phase=3 --use-gpu --set-mem 8192 --autobatch
```

**Note 1:** Both commands (step 2 and step 3) can resume the training process. Just add `--resume` as a commandline parameter and you will get a message saying `Resuming from previous checkpoint`.

**Note 2:** Modify `--set-mem` parameter to fit in the actual memory of you Video Card. For training the encoder you should have at least 8GB. For lower video card memory, you will need to decrease the `--batch-size` parameter for the vocoder and remove longer sentences from the Encoder training. Right now the Encoder trains of full utterances only, while the Vocoder segments them into slices. It is probable that you will obtain worse results if you have to decrease the maximum length of the utterances and the batch-size.  

## Step 4 - Ready to go

To synthesize text just type:
```bash
python3 cube/synthesis.py --input-file=<your input file> --output-file=<output wave file> --speaker=<speaker id>
```





