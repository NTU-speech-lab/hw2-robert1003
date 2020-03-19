#!/bin/bash

if [ $# -ne 6 ]; then
  echo -e "usage:\t./hw2_generative.sh train.csv test_no_label.csv X_train Y_train X_test prediction.csv"
  exit
fi

python3 hw2_generative.py $1 $2 $3 $4 $5 $6
