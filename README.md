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
