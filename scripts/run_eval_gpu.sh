
DEVICE_ID=$1
PATH_CHECKPOINT=$2
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
python ./eval.py --pretrain_ckpt $PATH_CHECKPOINT \
--device_target 'GPU' > eval.log 2>&1 &
