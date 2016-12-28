#!/bin/bash

data="$1"
trainparsefile="$2" #../data/lidong/parses/lidong.train.conll
testparsefile="$3" # ../data/lidong/parses/lidong.test.conll

# python tdparse.py --data "$data" --trainparse "$trainparsefile" --testparse "$testparsefile"
# python classification.py --data lidong --steps scale,tune,pred


python tdparse.py --data "$data" --trainparse "$trainparsefile" --testparse "$testparsefile"
python classification.py --data "$data" --steps scale
python runscikit.py --data "$data"
# python classification.py --data lidong --steps scale,tune,pred