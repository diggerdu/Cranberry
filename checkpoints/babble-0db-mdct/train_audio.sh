#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
expName=babble-0db-mdct
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/diggerdu/Assd/dataset/men/clean" \
 --PathNoise "/home/diggerdu/Assd/dataset/men/noise" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --nfft 512 --hop 256 --nFrames 256 --batchSize  5\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 0.001 \
 --gpu_ids 0 \
#  --serial_batches
