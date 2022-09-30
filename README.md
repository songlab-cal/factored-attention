# factored-attention

This repository contains code for reproducing results in our paper [Interpreting Potts and Transformer Protein Models Through the Lens of Simplified Attention](https://psb.stanford.edu/psb-online/proceedings/psb22/bhattacharya.pdf) (we also host the [Appendix](appendix.pdf) in this repo). This code is built entirely on [Mogwai](https://github.com/nickbhat/mogwai), a small library for MRF models of protein families.  If you wish to use our Potts or attention implementations for your own exploration, it is easier to use [Mogwai](https://github.com/nickbhat/mogwai) directly. If you have questions, feel free to contact us or open an issue!

## Installing

After cloning, please install mogwai and necessary dependencies with
```bash
$ make build
```

## Updating Mogwai Submodule

Anytime you pull, please be sure to update the Mogwai submodule as well
```bash
$ git pull
$ make
```


## Running a training run

Once you have set up your environment, run:

```bash
python train.py --model=factored_attention --attention_head_size=32 --batch_size=128 --l2_coeff=0.001 --learning_rate=0.005 --max_steps=5000 --num_attention_heads=256 --optimizer=adam --pdb=3er7_1_A
```
