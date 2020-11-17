import subprocess
from multiprocessing import Pool

def run(i):
    if i <= 9:
        str_i = f'0{i}'
    else:
        str_i = str(i)
    filename = f'fams_{str_i}.txt'
    shard = i
    subprocess.run(f'python weight_correlations.py --pdb_filename {filename} --shard {shard}', shell=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_shards", type=int)
    args = parser.parse_args()

    with Pool(args.num_workers) as p:
        p.map(run, list(range(args.num_shards)))
