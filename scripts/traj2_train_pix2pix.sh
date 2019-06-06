#!/bin/bash
nohup python train.py --dataroot ./datasets/trajs2 --name trajs2_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --load_size 256 --crop_size 256 --pool_size 0 --preprocess none --gpu_ids=4 --epoch_count 0 --niter 50 --niter_decay 50 --is_16_bit > train2.log 2>&1 &
tail -f train2.log
