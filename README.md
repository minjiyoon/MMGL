# Multimodal Graph Learning

Most multimodal learning algorithms focus on modeling simple one-to-one pairs of data from two modalities, such as image-caption pairs, or audiotext pairs. However, in most real-world settings, entities of different modalities
interact with each other in more complex and multifaceted ways, going beyond one-to-one mappings.

We propose Multimodal Graph Learning (MMGL), a systematic framework for capturing information from multiple multimodal
neighbors with relational structures among them.
In particular, we focus on MMGL for generative tasks, building upon pretrained Language Models (LMs), aiming to
augment their text generation with multimodal neighbor contexts.

The original paper can be found at [MMGL](https://arxiv.org/pdf/2310.07478.pdf)

## Setup

Create a new conda environment, install [PyTorch](https://pytorch.org) and the remaining requirements:
```
conda create python==3.7 -n mmgl
conda activate mmgl
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

The code supports the [WikiWeb2M] datasets (https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md):

## Data preprocessing

First, make a folder to download the WikiWeb2M dataset: `mkdir wikiweb2m/raw`.
Then download all Train/Validation/Test files from the WikiWeb2M into `wikiweb2m/raw'.

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

## Training

#### Logistic Regression

To encode a sparse feature matrix with specified features:
- Item Response Theory (IRT): `-i` 
- PFA: `-s -sc -w -a` 
- DAS3H: `-i -s -sc -w -a -tw`
- Best logistic regression features (Best-LR): `-i -s -ic -sc -tc -w -a`

```
python encode.py --dataset <dataset codename> <feature flags>
```

To train a logistic regression model with a sparse feature matrix encoded through encode.py:

```
python train_lr.py --X_file data/<dataset codename>/X-<feature suffix>.npz --dataset <dataset codename>
```

#### Deep Knowledge Tracing

To train a DKT model:

```
python train_dkt2.py --dataset <dataset codename> 
```

#### Self-Attentive Knowledge Tracing

To train a SAKT model:

```
python train_sakt.py --dataset <dataset codename>
```

## Results (AUC)

| Algorithm      | assist09      | assist12 | assist15      | assist17 | bridge06 | algebra05 | spanish  | statics  |
| -------------- | ------------- | -------- | ------------- | -------- | -------- | --------- | -------- | -------- |
| IRT            | 0.69          | 0.71     | 0.64          | 0.68     | 0.75     | 0.77      | 0.68     | 0.79     |       
| PFA            | 0.72          | 0.67     | 0.69          | 0.62     | 0.77     | 0.76      | 0.85     | 0.69     |
| DAS3H          | -             | 0.74     | -             | 0.69     | 0.79     | **0.83**  | -        | -        |
| Best-LR        | **0.77**      | 0.75     | 0.70          | 0.71     | **0.80** | **0.83**  | **0.86** | 0.82     |
| DKT            | 0.75          | **0.77** | **0.73**      | **0.77** | 0.79     | 0.82      | 0.83     | **0.83** |
| SAKT           | 0.75          | 0.73     | **0.73**      | 0.72     | 0.78     | 0.80      | 0.83     | 0.81     |
