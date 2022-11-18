
DEVICE_ID=$1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./train.py --device_target 'GPU' > train.log 2>&1 &

