#!/bin/bash
python 0_cut_filelist.py --number 20
wait $!
taskset -c 0 python 3_preprocess_train_mel.py -i part_0.txt &
taskset -c 1 python 3_preprocess_train_mel.py -i part_1.txt &
taskset -c 2 python 3_preprocess_train_mel.py -i part_2.txt &
taskset -c 3 python 3_preprocess_train_mel.py -i part_3.txt &
taskset -c 4 python 3_preprocess_train_mel.py -i part_4.txt &
taskset -c 5 python 3_preprocess_train_mel.py -i part_5.txt &
taskset -c 6 python 3_preprocess_train_mel.py -i part_6.txt &
taskset -c 7 python 3_preprocess_train_mel.py -i part_7.txt &
taskset -c 8 python 3_preprocess_train_mel.py -i part_8.txt &
taskset -c 9 python 3_preprocess_train_mel.py -i part_9.txt &
taskset -c 10 python 3_preprocess_train_mel.py -i part_10.txt &
taskset -c 11 python 3_preprocess_train_mel.py -i part_11.txt &
taskset -c 12 python 3_preprocess_train_mel.py -i part_12.txt &
taskset -c 13 python 3_preprocess_train_mel.py -i part_13.txt &
taskset -c 14 python 3_preprocess_train_mel.py -i part_14.txt &
taskset -c 15 python 3_preprocess_train_mel.py -i part_15.txt &
taskset -c 16 python 3_preprocess_train_mel.py -i part_16.txt &
taskset -c 17 python 3_preprocess_train_mel.py -i part_17.txt &
taskset -c 18 python 3_preprocess_train_mel.py -i part_18.txt &
taskset -c 19 python 3_preprocess_train_mel.py -i part_19.txt &