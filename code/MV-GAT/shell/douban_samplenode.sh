#!/bin/bash
for cuda in 0
do
for dataset in cora
do
for K_sample_num in 8 2 5 10 1
do
for init_prob in 0.1 0.3 0.5 0.8 1.0
do
for N_average_num in 10 30 50
do
for FT_weight_share in 0
do
python execute_doubanmovie_sparse.py --cuda $cuda --dataset $dataset --K_sample_num $K_sample_num --init_prob $init_prob --N_average_num $N_average_num --FT_weight_share $FT_weight_share >>douban_DSGAT_0908.log
done
done
done
done
done
done