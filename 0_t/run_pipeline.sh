#!/bin/bash
app=$1
model=$2
gpu_option=$3

python src/train_val_stu.py $app $model $gpu_option