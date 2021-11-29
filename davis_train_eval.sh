config="configs.resnet101_comatch"
gpu_num=1
start_gpu=0
batch_size=1
datasets="davis2017"

python   tools/eval_net.py --config ${config} --datasets ${datasets} --batch_size ${per_batch_size} --gpu_num ${gpu_num} --start_gpu ${start_gpu}
