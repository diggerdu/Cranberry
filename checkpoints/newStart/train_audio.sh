#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
expName=newStart
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/mnt/dataset/VCTK-Corpus/wav48" \
 --PathNoise "/home/diggerdu/Assd/dataset/Large/babble" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --nfft 1024 --hop 512 --nFrames 128 --batchSize 6\
 --split_hop 0 \
 --niter 100000000000000000000000000000000000 --niter_decay 30 \
 --lr 1e-2 \
 --gpu_ids 0,1,2 \
# --continue_train
#  --serial_batches
