#!/bin/bash
python 0_cut_filelist.py --number 15 &&

taskset -c 0 python 1_preprocess_train_f0.py -i part_0.txt &
taskset -c 1 python 1_preprocess_train_f0.py -i part_1.txt &
taskset -c 2 python 1_preprocess_train_f0.py -i part_2.txt &
taskset -c 3 python 1_preprocess_train_f0.py -i part_3.txt &
taskset -c 4 python 1_preprocess_train_f0.py -i part_4.txt &
taskset -c 5 python 1_preprocess_train_f0.py -i part_5.txt &
taskset -c 6 python 1_preprocess_train_f0.py -i part_6.txt &
taskset -c 7 python 1_preprocess_train_f0.py -i part_7.txt &
taskset -c 8 python 1_preprocess_train_f0.py -i part_8.txt &
taskset -c 9 python 1_preprocess_train_f0.py -i part_9.txt &
taskset -c 10 python 1_preprocess_train_f0.py -i part_10.txt &
taskset -c 11 python 1_preprocess_train_f0.py -i part_11.txt &
taskset -c 12 python 1_preprocess_train_f0.py -i part_12.txt &
taskset -c 13 python 1_preprocess_train_f0.py -i part_13.txt &
taskset -c 14 python 1_preprocess_train_f0.py -i part_14.txt &

