# ENeSy

This repository provides the source code & data of our paper: [Neural-Symbolic Entangled Framework for Complex Query Answering (NeurIPS 2022)](https://arxiv.org/pdf/2209.08779).

## Requirement

```
torch==1.10.0
tensorboardX
numpy
tqdm
```

## Train the model

The standard datasets used in our work can be found in the public repository of previous work BetaE[1] and can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).

### 1st step

Train the embedding of entities and relations with link prediction.

```shell
python main.py --cuda --do_train --do_valid --data_path=data/DATASET -lr=0.00001 --geo=ns --tasks="1p" -kge=RotatE -pre_1p
```

### 2nd step

Train the MLP function which is used to convert symbolic vector to embedding.

```shell
python main.py --cuda --do_train --do_valid --data_path=data/DATASET -lr=0.00001 --geo=ns --tasks="1p" -kge=RotatE -newloss --checkpoint_path=CHECKPOINTPATH --warm_up_steps=STEP
```

### 3rd step(Opt)

Fine-tune the model with complex query data.

```shell
python main.py --cuda --do_train --do_valid --data_path=data/DATASET -lr=0.00001 --geo=ns --tasks="1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up" -kge=RotatE --checkpoint_path=CHECKPOINTPATH --warm_up_steps=STEP
```

`--geo`: string, select the reasoning model, `vec` for GQE, `box` for Query2box, `beta` for BetaE, `ns` for neural-symbolic.

`--tasks`: string, tasks connected by dot.

`-kge`: string, select the neural reasoning way of projection.

## Test the model

```bash
python main.py --cuda --do_test --data_path=data/DATASET -lr=0.00001 --geo=ns --tasks="1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up" -kge=RotatE --checkpoint_path=CHECKPOINTPATH -lambdas="0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5"
```

`-lambdas`: string, lambda used for ensemble prediction for each task connected by semicolon.

## Citation

If you find this code useful, please cite the following paper.
 ```
@inproceedings{
    xu2022neuralsymbolic,
    title={Neural-Symbolic Entangled Framework for Complex Query Answering},
    author={Zezhong Xu and Wen Zhang and Peng Ye and Hui Chen and Huajun Chen},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    url={https://arxiv.org/pdf/2209.08779}
}
 ```

[1] Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs, Hongyu Ren and Jure Leskovec, NeurIPS 2020.
