export CUDA_VISIBLE_DEVICES=2
expName=babble-0db-mdct
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
cp $selfPath checkpoints/$expName/

python test.py \
 --PathClean "/home/diggerdu/Assd/dataset/men/clean" \
 --PathNoise "/home/diggerdu/Assd/dataset/men/noise" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --nThreads 1 \
 --nfft 512 --hop 256 --nFrames 128 \
 --gpu_ids 0 --batchSize 1  --how_many 6 \
 --split_hop 0 \

mv *.wav checkpoints/$expName/

