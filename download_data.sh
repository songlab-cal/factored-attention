#!/bin/bash
mkdir data
mkdir gremlin-results
mkdir fatt-pl-results
mkdir bert-results

wget -P data/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.i90c75.a3m
wget -P data/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.pdb
wget -P data/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.cf

wget -P data/5cajA -q -nc https://files.ipd.uw.edu/krypton/5cajA.i90c75.a3m
wget -P data/5cajA -q -nc https://files.ipd.uw.edu/krypton/5cajA.pdb
wget -P data/5cajA -q -nc https://files.ipd.uw.edu/krypton/5cajA.cf

# aws s3 cp s3://songlabdata/proteindata/data_raw_pytorch/trrosetta.tar.gz ./data/
