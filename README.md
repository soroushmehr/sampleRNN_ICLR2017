# SampleRNN
Code accompanying the paper [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://openreview.net/forum?id=SkxKPDv5xl). Samples are available [here](https://soundcloud.com/samplernn/sets).

## Dependencies
- cuDNN 5105
- Python 2.7.12
- Numpy 1.11.1
- Theano 0.8.2 (0.9 for WaveNet re-implementation)
- Lasagne 0.2.dev1

## Datasets
Music dataset was created from all 32 Beethoven’s piano sonatas available publicly on [archive.org](https://archive.org/). `datasets/music` contains scripts to preprocess and build this dataset. It is also available [here](https://drive.google.com/drive/folders/0B7riq_C8aslvbWJuMGhJRFBmSHM?usp=sharing) for download. Extract the tar file and put all the numpy files in `datasets/music` directory.

## Training
To train a model on an existing dataset with accelerated GPU processing, you need to run following lines from the root of `sampleRNN_ICLR2017` folder which corresponds to the best found set of hyper-paramters.

Mission control center:
```
$ pwd
/u/mehris/sampleRNN_ICLR2017
```
### SampleRNN (2-tier)
```
$ python models/two_tier/two_tier.py -h
usage: two_tier.py [-h] [--exp EXP] --n_frames N_FRAMES --frame_size
                   FRAME_SIZE --weight_norm WEIGHT_NORM --emb_size EMB_SIZE
                   --skip_conn SKIP_CONN --dim DIM --n_rnn {1,2,3,4,5}
                   --rnn_type {LSTM,GRU} --learn_h0 LEARN_H0 --q_levels
                   Q_LEVELS --q_type {linear,a-law,mu-law} --which_set
                   {ONOM,BLIZZ,MUSIC} --batch_size {64,128,256} [--debug]
                   [--resume]

two_tier.py No default value! Indicate every argument.

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             Experiment name
  --n_frames N_FRAMES   How many "frames" to include in each Truncated BPTT
                        pass
  --frame_size FRAME_SIZE
                        How many samples per frame
  --weight_norm WEIGHT_NORM
                        Adding learnable weight normalization to all the
                        linear layers (except for the embedding layer)
  --emb_size EMB_SIZE   Size of embedding layer (0 to disable)
  --skip_conn SKIP_CONN
                        Add skip connections to RNN
  --dim DIM             Dimension of RNN and MLPs
  --n_rnn {1,2,3,4,5}   Number of layers in the stacked RNN
  --rnn_type {LSTM,GRU}
                        GRU or LSTM
  --learn_h0 LEARN_H0   Whether to learn the initial state of RNN
  --q_levels Q_LEVELS   Number of bins for quantization of audio samples.
                        Should be 256 for mu-law.
  --q_type {linear,a-law,mu-law}
                        Quantization in linear-scale, a-law-companding, or mu-
                        law compandig. With mu-/a-law quantization level shoud
                        be set as 256
  --which_set {ONOM,BLIZZ,MUSIC}
                        ONOM, BLIZZ, or MUSIC
  --batch_size {64,128,256}
                        size of mini-batch
  --debug               Debug mode
  --resume              Resume the same model from the last checkpoint. Order
                        of params are important. [for now]
```
To run:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier.py --exp BEST_2TIER --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 3 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 128 --weight_norm True --learn_h0 True --which_set MUSIC
```
### SampleRNN (3-tier)
```
$ python models/three_tier/three_tier.py -h
usage: three_tier.py [-h] [--exp EXP] --seq_len SEQ_LEN --big_frame_size
                     BIG_FRAME_SIZE --frame_size FRAME_SIZE --weight_norm
                     WEIGHT_NORM --emb_size EMB_SIZE --skip_conn SKIP_CONN
                     --dim DIM --n_rnn {1,2,3,4,5} --rnn_type {LSTM,GRU}
                     --learn_h0 LEARN_H0 --q_levels Q_LEVELS --q_type
                     {linear,a-law,mu-law} --which_set {ONOM,BLIZZ,MUSIC}
                     --batch_size {64,128,256} [--debug] [--resume]

three_tier.py No default value! Indicate every argument.

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             Experiment name
  --seq_len SEQ_LEN     How many samples to include in each Truncated BPTT
                        pass
  --big_frame_size BIG_FRAME_SIZE
                        How many samples per big frame in tier 3
  --frame_size FRAME_SIZE
                        How many samples per frame in tier 2
  --weight_norm WEIGHT_NORM
                        Adding learnable weight normalization to all the
                        linear layers (except for the embedding layer)
  --emb_size EMB_SIZE   Size of embedding layer (> 0)
  --skip_conn SKIP_CONN
                        Add skip connections to RNN
  --dim DIM             Dimension of RNN and MLPs
  --n_rnn {1,2,3,4,5}   Number of layers in the stacked RNN
  --rnn_type {LSTM,GRU}
                        GRU or LSTM
  --learn_h0 LEARN_H0   Whether to learn the initial state of RNN
  --q_levels Q_LEVELS   Number of bins for quantization of audio samples.
                        Should be 256 for mu-law.
  --q_type {linear,a-law,mu-law}
                        Quantization in linear-scale, a-law-companding, or mu-
                        law compandig. With mu-/a-law quantization level shoud
                        be set as 256
  --which_set {ONOM,BLIZZ,MUSIC}
                        ONOM, BLIZZ, or MUSIC
  --batch_size {64,128,256}
                        size of mini-batch
  --debug               Debug mode
  --resume              Resume the same model from the last checkpoint. Order
                        of params are important. [for now]
```
To run:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/three_tier/three_tier.py --exp BEST_3TIER --seq_len 512 --big_frame_size 8 --frame_size 2 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 128 --weight_norm True --learn_h0 True --which_set MUSIC
```

## Reference
If you are using this code, please cite the paper.

SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, Yoshua Bengio, 5th International Conference on Learning Representations (ICLR 2017), submitted and under review.

[Bib to be added soon...]

If needed, please don't hesitate to contact us.
