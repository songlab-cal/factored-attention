# iclr-2021-factored-attention

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

todo: out of the box apex?


## Running a training run

Once you have set up your environment, run:

```bash
python train.py --model=factored_attention --attention_head_size=32 --batch_size=128 --l2_coeff=0.001 --learning_rate=0.005 --max_steps=5000 --num_attention_heads=256 --optimizer=adam --pdb=3er7_1_A
```
