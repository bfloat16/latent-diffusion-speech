#!/bin/bash
python 0_cut_filelist.py --number 6
wait
taskset -c 0,1 python 2_preprocess_train_unit.py -i part_0.txt &
taskset -c 2,3 python 2_preprocess_train_unit.py -i part_1.txt &
taskset -c 4,5 python 2_preprocess_train_unit.py -i part_2.txt &
taskset -c 6,7 python 2_preprocess_train_unit.py -i part_3.txt &
taskset -c 8,9 python 2_preprocess_train_unit.py -i part_4.txt &
taskset -c 10,11 python 2_preprocess_train_unit.py -i part_5.txt &
