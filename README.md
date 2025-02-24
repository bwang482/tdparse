# TDParse - Multi-target-specific sentiment recognition on Twitter
From the EACL 2017 paper, TDParse utilises the syntactic information from parse-tree in conjunction with the left-right context of the target and achieves the state-of-the-art performance on both the benchmarking single-target corpus and new multi-target election data.

Our approximated version of the LSTM models proposed in [COLING 2016](https://arxiv.org/abs/1512.01100), can be found at [here](https://github.com/bluemonk482/tdlstm).

## Dependencies
- Python 2.7
- sklearn >= 0.18.1
- gensim == 0.13.4
- networkx == 1.11
- [ftfy](https://github.com/LuminosoInsight/python-ftfy) >= 4.1.1
- [TweeboParser](https://github.com/ikekonglp/TweeboParser) >= April 1, 2016

## Data
You can find our election corpus at [here](https://dx.doi.org/10.6084/m9.figshare.4479563.v1).

## Usage

Run **TDParse**
```bash
## e.g. using LibLinear with parameter tuning:
./run.sh lidong tdparse liblinear scale,tune,pred ../data/lidong/parses/lidong.train.conll ../data/lidong/parses/lidong.test.conll
## or without parameter tuning; adding your choice of C-parameter in the end:
./run.sh lidong tdparse liblinear scale,pred ../data/lidong/parses/lidong.train.conll ../data/lidong/parses/lidong.test.conll 0.01
```
Run **Naive-seg**
```bash
## e.g. using scikit-learn implementation of Linear SVM
./run.sh election naiveseg sklearnSVM
```

## Reference
"TDParse - Multi-target-specific sentiment recognition on Twitter" - Bo Wang, Maria Liakata, Arkaitz Zubiaga, Rob Procter, to be published in EACL 2017

## Acknowledgement
Thanks to Duy-Tin Vo and Yue Zhang, the authors of *"Target-dependent Twitter Sentiment Classification with Rich Automatic Features"*, for sharing their code which I have built my implementation upon.
