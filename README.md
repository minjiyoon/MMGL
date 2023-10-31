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
The code is implemented on PyTorch DistributedDataParallel.
The code supports the [WikiWeb2M](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md) dataset.

## Data preprocessing

First, make a folder to download the WikiWeb2M dataset: `mkdir wikiweb2m/raw`.
Then download all Train/Validation/Test files from the WikiWeb2M into `wikiweb2m/raw`.
Next, make a folder to download images: `mkdir wikiweb2m/raw/images`.
Finally, run `preprocess_data.py` to convert the WikiWeb2M dataset into pytorch format.

```
python preprocess_data.py
```

The output training/validation/test set sizes for section summarization is as follows:

| Number of | Train | Validation | Test |
| ---- | ---- | ---- | ---- |
| Sections | 680K | 170K | 170K |

## Training

#### Script

In `script/train_generation.sh`, you can specify the base model (`MODEL_NAME`), the task (`TASK`; currently we support only section summarization 'section'), the neighbor context (`CONTEXT`).
For `CONTEXT`, there are four options as follows:

| CONTEXT | description |
| ---- | ---- |
| section_only | use only text in the target section |
| section_all | use text and images in the target section |
| text_only | use only text in the all page |
| all | use text and images in the all page |

You can set how to encode text neighbors using `NEIGHBOR_MODE`. There are two options as follows:

| NEIGHBOR_MODE | description |
| ---- | ---- |
| raw | concatenate text neighbors as raw text into the input text |
| embedding | embed text neighbors using `text_model` and concatenate embeddings into the input text |

You can set the parameter-efficient fine-tuning (PEFT) option in the script using `PEFT_TYPE`. There are four PEFT options.

| CONTEXT | description |
| ---- | ---- |
| none | full finetune |
| prompt | prompt tuning |
| prefix | prefix tuning |
| lora | LoRA |
| flamingo | fine-tune only newly added cross-attention; can be used on decode-only models with `neighbor_mode = embedding`|

In the script, you can change `max_input_length` and `max_output_length` in addition to other optimization hyperparameters (e.g., `epochs`, `learning_rate`, `per_device_train_batch_size`). 
You can set which models to encode text and image neighbors using `text_model` and `visual_model`.
All arguments you can set are defined under `Argument` class in `language_modelling/run_generation.py`.

#### File description

We provide brief descriptions for each file as follows:

| Directory/File | description |
| ---- | ---- |
| wikiweb2m/ | codes related to WikiWeb2M dataset |
| wikiweb2m/cider | compute CIDEr scores |
| wikiweb2m/data.py | prepare each training point based on `context` and `neighbor_mode` |
| wikiweb2m/preprocess_data.py | codes to preprocess WikiWeb2M dataset and download images |
| script/ | codes to run MMGL |
| script/train_generation.sh | set hyperparameters |
| language_modelling/ | main directory |
| language_modelling/run_generation.py | prepare models, read datasets, train/validation loops |
| language_modelling/utils.py | utility functions |
| model/ | language models |
| model/modelling_self_attention.py | LMs only with self-attention; including encoder-decoder and decoder-only models  |
| model/modelling_cross_attention.py | LMs with cross-attention to encode neighbor information; decoder-only models|

## Citation
If you find this work or our code useful, please consider citing:
```
@article{yoon2023multimodal,
  title={Multimodal Graph Learning for Generative Tasks},
  author={Yoon, Minji and Koh, Jing Yu and Hooi, Bryan and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2310.07478},
  year={2023}
}
```
