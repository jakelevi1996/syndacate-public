# syndacate-public

Public code implementation for the paper "SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference", which is [available on Arxiv](https://arxiv.org/pdf/2506.17558).

Results highlights and commands can be found in our attached [Jupyter notebook](syndacate.ipynb). Included in this `README.md` are installation instructions, and commands to replicate the full sweeps shown in Figure 1 in the paper.

## Contents

- [syndacate-public](#syndacate-public)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Full classification data-efficiency sweeps](#full-classification-data-efficiency-sweeps)
  - [Full `PartsToChars` depth sweeps](#full-partstochars-depth-sweeps)
  - [Full `ImToParts` results averaged over 5 seeds](#full-imtoparts-results-averaged-over-5-seeds)

## Installation

```
python -m pip install -U pip
python -m pip install -U jutility==0.0.28
python -m pip install -U juml-toolkit==0.0.5
git clone https://github.com/jakelevi1996/syndacate-public.git
cd syndacate-public
python -m pip install -e .
```

Verify installation:

```sh
syndacate -h
syndacate plotsyndacate -h
syndacate plotsyndacate
syndacate train -h
syndacate train --dataset PartsToChars --model SetTransformer --trainer.BpSp.epochs 1
# ...
# Model name = `dPC_lA_mSd2eIh8m64pIx2.0_tBb100e1lCle1E-05oAol0.001_s0`
# Final metrics = 10.51837 (train), 10.54011 (test)
# Time taken for `train` = 21.1465 seconds

syndacate plotcharpredictions --model_name dPC_lA_mSd2eIh8m64pIx2.0_tBb100e1lCle1E-05oAol0.001_s0
```

## Full classification data-efficiency sweeps

```sh
syndacate sweep \
  --dataset ImToClass \
  --model RzCnn \
  --model.RzCnn.embedder CoordConv \
  --model.RzCnn.pooler Average2d \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset ImToClass \
  --model CapsNet \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PreTrainedPartsToClass \
  --model RzCnn \
  --model.RzCnn.embedder CoordConv \
  --model.RzCnn.pooler Average2d \
  --model.RzCnn.num_stages 1 \
  --model.RzCnn.blocks_per_stage 3 \
  --model.RzCnn.stride 1 \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PreTrainedPartsToClass \
  --model CapsNet \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToClass \
  --model SetTransformer \
  --model.SetTransformer.pooler SetAverage \
  --model.SetTransformer.depth 5 \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToClass \
  --model RzMlp \
  --model.RzMlp.embedder Flatten \
  --model.RzMlp.embedder.Flatten.n 2 \
  --model.RzMlp.depth 5 \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 5000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToClass \
  --model RzMlp \
  --model.RzMlp.embedder Flatten \
  --model.RzMlp.embedder.Flatten.n 2 \
  --model.RzMlp.depth 5 \
  --trainer BpSpDe \
  --trainer.BpSpDe.steps 50000 \
  --sweep.params '{"trainer.BpSpDe.n_train":[100, 300, 1000, 3000, 10000, 30000, 60000]}' \
  --sweep.log_x trainer.BpSpDe.n_train \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate comparesweeps \
  --comparesweeps.config '[
    {
      "series_name":"CNN",
      "sweep_name":"dICLlCmRZCb2eCk5m64n3pAs2x2.0tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"CapsNet",
      "sweep_name":"dICLlCmCAr3tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"CNN+PTPE",
      "sweep_name":"dPTPCLlCmRZCb3eCk5m64n1pAs1x2.0tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"CapsNet+PTPE",
      "sweep_name":"dPTPCLlCmCAr3tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"ST+GTP",
      "sweep_name":"dPCLlCmSd5eIh8m64pSEx2.0tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"MLP (flat)+GTP",
      "sweep_name":"dPCLlCmRZMd5eFen2m100pIx2.0tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s5000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    },
    {
      "series_name":"MLP (flat, 10xS)+GTP",
      "sweep_name":"dPCLlCmRZMd5eFen2m100pIx2.0tBSDb100lCle1E-05n1,3,600,0,00oAol0.001s50000s0,1,2,3,4",
      "param_name":"trainer.BpSpDe.n_train"
    }
  ]' \
  --comparesweeps.xlabel "Training set size" \
  --comparesweeps.clabel "Model type" \
  --comparesweeps.log_x \
  --comparesweeps.name classification_all \
  --comparesweeps.plot_type Show \
  --dataset PartsToClass
```

## Full `PartsToChars` depth sweeps

```sh
syndacate sweep \
  --dataset PartsToChars \
  --model SetTransformer \
  --trainer.BpSp.epochs 100 \
  --sweep.params '{"model.SetTransformer.depth":[0,1,2,3,4,5]}' \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToChars \
  --model SetTransformer \
  --model.SetTransformer.model_dim 128 \
  --trainer.BpSp.epochs 100 \
  --sweep.params '{"model.SetTransformer.depth":[0,1,2,3,4,5]}' \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToChars \
  --model DeepSetToSet \
  --trainer.BpSp.epochs 100 \
  --sweep.params '{"model.DeepSetToSet.depth":[0,1,2,3,4,5]}' \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToChars \
  --model RzMlp \
  --model.RzMlp.embedder Flatten \
  --model.RzMlp.embedder.Flatten.n 2 \
  --model.RzMlp.pooler Unflatten \
  --model.RzMlp.pooler.Unflatten.n 2 \
  --trainer.BpSp.epochs 100 \
  --sweep.params '{"model.RzMlp.depth":[0,1,2,3,4,5]}' \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate sweep \
  --dataset PartsToChars \
  --model RzMlp \
  --trainer.BpSp.epochs 100 \
  --sweep.params '{"model.RzMlp.depth":[0,1,2,3,4,5]}' \
  --sweep.no_cache \
  --sweep.devices '[[0]]'

syndacate comparesweeps \
  --comparesweeps.config '[
    {
      "series_name":"ST",
      "sweep_name":"dPClAmSh8m64n0,1,2,3,4,5x2.0tBb100e100lCle1E-05oAol0.001s0,1,2,3,4",
      "param_name":"model.SetTransformer.depth"
    },
    {
      "series_name":"ST (2xW)",
      "sweep_name":"dPClAmSd0,1,2,3,4,5eIh8m128pIx2.0tBb100e100lCle1E-05oAol0.001s0,1,2,3,4",
      "param_name":"model.SetTransformer.depth"
    },
    {
      "series_name":"DSTS",
      "sweep_name":"dPClAmDeIh100n0,1,2,3,4,5pItBb100e100lCle1E-05oAol0.001s0,1,2,3,4",
      "param_name":"model.DeepSetToSet.depth"
    },
    {
      "series_name":"MLP (flat)",
      "sweep_name":"dPClAmRZMeFen2m100n0,1,2,3,4,5pUpn2x2.0tBb100e100lCle1E-05oAol0.001s0,1,2,3,4",
      "param_name":"model.RzMlp.depth"
    },
    {
      "series_name":"MLP (EW)",
      "sweep_name":"dPClAmRZMeIm100n0,1,2,3,4,5pIx2.0tBb100e100lCle1E-05oAol0.001s0,1,2,3,4",
      "param_name":"model.RzMlp.depth"
    }
  ]' \
  --comparesweeps.xlabel Depth \
  --comparesweeps.clabel "Model type" \
  --comparesweeps.name pc_depth \
  --comparesweeps.plot_type Show \
  --dataset PartsToChars
```

## Full `ImToParts` results averaged over 5 seeds

```
syndacate sweep \
  --dataset ImToParts \
  --model RzCnn \
  --model.RzCnn.embedder CoordConv \
  --model.RzCnn.pooler LinearSet2d \
  --trainer.BpSp.epochs 100 \
  --sweep.no_cache \
  --sweep.devices '[[0]]'
```
