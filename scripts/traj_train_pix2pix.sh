#!/bin/bash
nohup python train.py --dataroot ./datasets/gen-5 --name trajs_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --load_size 256 --crop_size 256 --pool_size 0 --preprocess none --gpu_ids=5 --epoch_count 0 --niter 100 --niter_decay 100 --is_16_bit > train.log 2>&1 &
tail -f train.log
