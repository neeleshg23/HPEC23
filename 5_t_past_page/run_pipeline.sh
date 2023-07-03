#!/bin/bash
app=$1
model=$2
gpu_option=$3

python src/train_tch.py $app $model $gpu_option
python src/validate_tch.py $app $model $gpu_option
python src/train_stu.py $app $model $gpu_option