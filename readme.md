# CLaRE: A Cubic Lattice Network for Joint Entity and Relation Extraction

## Code Structure

```sh
./dataset: the directory to store the datasets
./train_llms: the directory to train llms
./train_bert_like_models: the directory to train bert-like models
```

## Dataset

These are the datasets used in our paper:ACE05,ADE,CMeIE.

The detail of ach dataset can be found in our paper.

## Environment Configuration

| Package            | Version   |
|--------------------|-----------|
| `torch`            | 2.1.2+cu121 |
| `transformers`     | 4.38.1    |
| `datasets`         | 2.17.1    |
| `peft`             | 0.8.2     |
| `deepspeed`        | 0.12.6    |
| `accelerate`       | 0.27.2    |
| `numpy`            | 1.23.4    |
| `scikit-learn`     | 1.2.1     |
| `tqdm`             | 4.64.1    |
| `sentencepiece`    | 0.1.99    |
| `evaluate`         | 0.4.1     |

## Code for Bert Like Models Training

```sh
cd ./train_bert_like_models/run_script
bash run_ace05.sh
```

## Code for LLM Training

```sh
cd ./train_llms/LLaMA-Factory/CLARE_scirpts
bash train.sh
```