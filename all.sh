#!/bin/bash
export PYTHONWARNINGS=ignore
echo "1===================================================================="
python 0_cut_filelist.py --number 20
wait
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
taskset -c 15 python 1_preprocess_train_f0.py -i part_15.txt &
taskset -c 16 python 1_preprocess_train_f0.py -i part_16.txt &
taskset -c 17 python 1_preprocess_train_f0.py -i part_17.txt &
taskset -c 18 python 1_preprocess_train_f0.py -i part_18.txt &
taskset -c 19 python 1_preprocess_train_f0.py -i part_19.txt &
wait
echo "2===================================================================="
python 0_cut_filelist.py --number 6
wait
taskset -c 0,1 python 2_preprocess_train_unit.py -i part_0.txt &
taskset -c 2,3 python 2_preprocess_train_unit.py -i part_1.txt &
taskset -c 4,5 python 2_preprocess_train_unit.py -i part_2.txt &
taskset -c 6,7 python 2_preprocess_train_unit.py -i part_3.txt &
taskset -c 8,9 python 2_preprocess_train_unit.py -i part_4.txt &
taskset -c 10,11 python 2_preprocess_train_unit.py -i part_5.txt &
wait
echo "3===================================================================="
python 0_cut_filelist.py --number 20
wait
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
wait
echo "4===================================================================="
python 4_preprocess_train_tts.py
wait
echo "5===================================================================="
python 0_cut_filelist.py --number 20
wait
taskset -c 0 python 5_preprocess_train_volume.py -i part_0.txt &
taskset -c 1 python 5_preprocess_train_volume.py -i part_1.txt &
taskset -c 2 python 5_preprocess_train_volume.py -i part_2.txt &
taskset -c 3 python 5_preprocess_train_volume.py -i part_3.txt &
taskset -c 4 python 5_preprocess_train_volume.py -i part_4.txt &
taskset -c 5 python 5_preprocess_train_volume.py -i part_5.txt &
taskset -c 6 python 5_preprocess_train_volume.py -i part_6.txt &
taskset -c 7 python 5_preprocess_train_volume.py -i part_7.txt &
taskset -c 8 python 5_preprocess_train_volume.py -i part_8.txt &
taskset -c 9 python 5_preprocess_train_volume.py -i part_9.txt &
taskset -c 10 python 5_preprocess_train_volume.py -i part_10.txt &
taskset -c 11 python 5_preprocess_train_volume.py -i part_11.txt &
taskset -c 12 python 5_preprocess_train_volume.py -i part_12.txt &
taskset -c 13 python 5_preprocess_train_volume.py -i part_13.txt &
taskset -c 14 python 5_preprocess_train_volume.py -i part_14.txt &
taskset -c 15 python 5_preprocess_train_volume.py -i part_15.txt &
taskset -c 16 python 5_preprocess_train_volume.py -i part_16.txt &
taskset -c 17 python 5_preprocess_train_volume.py -i part_17.txt &
taskset -c 18 python 5_preprocess_train_volume.py -i part_18.txt &
taskset -c 19 python 5_preprocess_train_volume.py -i part_19.txt &
wait
echo "6===================================================================="
python 6_preprocess_train_pitch.py
