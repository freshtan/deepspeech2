
PATH_CHECKPOINT=$1
python ./eval.py --pretrain_ckpt $PATH_CHECKPOINT --device_target 'CPU' > eval.log 2>&1 &
