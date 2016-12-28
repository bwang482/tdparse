#!/bin/bash

data=$1
model=$2
classifier=$3
if [ "$classifier" == "liblinear" ]; then
	steps=$4 #scale,tune,pred
	trainparsefile="$5" #../data/lidong/parses/lidong.train.conll
	testparsefile="$6" #../data/lidong/parses/lidong.test.conll
elif [ "$classifier" == "sklearnSVM" ]; then
	trainparsefile="$4" #../data/lidong/parses/lidong.train.conll
	testparsefile="$5" #../data/lidong/parses/lidong.test.conll
else
	 echo "Please make sure to use appropriate model and classifier."
 fi


if [ "$model" == "naiveseg" ] && [ "$classifier" == "liblinear" ]; then
	python naive-seg.py --data "$data"
	python liblinear.py --data "$data" --steps "$steps"
elif [ "$model" == "tdparse" ] && [ "$classifier" == "liblinear" ]; then
	python tdparse.py --data "$data" --trainparse "$trainparsefile" --testparse "$testparsefile"
	python liblinear.py --data "$data" --steps "$steps"

elif [ "$model" == "naiveseg" ] && [ "$classifier" == "sklearnSVM" ]; then
	python naive-seg.py --data "$data"
	python liblinear.py --data "$data" --steps scale
	python sklearnSVM.py --data "$data"
elif [ "$model" == "tdparse" ] && [ "$classifier" == "sklearnSVM" ]; then
	python tdparse.py --data "$data" --trainparse "$trainparsefile" --testparse "$testparsefile"
	python liblinear.py --data "$data" --steps scale
	python sklearnSVM.py --data "$data"

else
	 echo "Please make sure to use appropriate model and classifier."
 fi