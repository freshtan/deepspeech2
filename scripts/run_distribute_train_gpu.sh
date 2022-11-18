
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
  python ./train.py --is_distributed --device_target 'GPU' > train_8p.log 2>&1 &
