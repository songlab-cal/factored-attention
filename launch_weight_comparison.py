import subprocess
from multiprocessing import Pool

def run(i):
    filename = f'fams_0{i}.txt'
    shard = i
    subprocess.run(f'python weight_correlations.py --pdb_filename {filename} --shard {shard}', shell=True)


if __name__ == '__main__':
    with Pool(5) as p:
        p.map(run, list(range(10)))
