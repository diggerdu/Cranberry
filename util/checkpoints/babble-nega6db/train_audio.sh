#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
expName=frames128-cirm-large
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/diggerdu/dataset/Large/clean" \
 --PathNoise "/home/diggerdu/dataset/Large/noise" \
 --snr -6 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --input_nc 1 --output_nc 1 \
 --nfft 1024 --hop 512 --nFrames 128 --batchSize  7\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 0.001 \
 --gpu_ids 0 \
 --continue_train
#  --serial_batches
