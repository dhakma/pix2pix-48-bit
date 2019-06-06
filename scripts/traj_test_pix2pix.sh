#!/bin/bash
nohup python test.py --dataroot ./datasets/trajs --name trajs_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm batch --load_size 256 --crop_size 256 --preprocess none --gpu_ids=5 --is_16_bit --num_test 1500 > test.log 2>&1 &
#nohup python test.py --dataroot ./datasets/trajs --name trajs_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm batch --load_size 256 --crop_size 256 --preprocess none --gpu_ids=1 > test.log 2>&1 &
tail -f test.log

