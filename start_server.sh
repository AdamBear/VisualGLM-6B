#!/usr/bin/env bash

if [ ! $1 ];then
  echo "need provider GPU number!"
  exit 1
fi


export CUDA_VISIBLE_DEVICES=$1
nohup python api_hf.py --gpu $1 >> log_gpu$1.log 2>&1 & echo $! > pid_gpu$1.txt