# Launches single-gpu jobs for given agent on all gpus.
agent=$1
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
let "n = $num_gpus - 1"
for i in $(seq 0 $n)
do
    echo "CUDA_VISIBLE_DEVICES=$i wandb agent proteins/gremlin-contacts/$agent &> $i.out &"
    # sometimes jobs time out when we run too many, so we increase the timeout
    CUDA_VISIBLE_DEVICES=$i WANDB_HTTP_TIMEOUT=60 wandb agent proteins/gremlin-contacts/$agent &> $i.out &
done
