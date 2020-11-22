n=$1
for i in $(seq 0 $n)
do
    echo "python get_gremlin_stratified_contacts.py --shard $i --total_shards $n &"
    python get_gremlin_stratified_contacts.py --shard $i --total_shards $n &
done
