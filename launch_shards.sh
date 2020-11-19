n=$1
for i in $(seq 0 $n)
do
    echo "python weight_correlations.py --shard $i --total_shards $n &"
    python weight_correlations.py --shard $i --total_shards $n &
done
