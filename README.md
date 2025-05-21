# LMCD: Language Models are Zeroshot Cognitive Diagnosis Learners

## Step 0. Environment version
Key Dependency Versions

| Dependency | Version |
|------------|---------|
| torch | 2.3.0 |
| transformers | 4.45.0 |
| deepspeed | 0.13.5 |

## Step 1. Download datasets and language models
### LLM Model
Download qwen2.5-1.5B-base from https://huggingface.co/Qwen/Qwen2.5-1.5B

Put the downloaded model into the "llm_model" folder



### Datasets
exercise_cold_start, cross-domain cold start data, and knowledge concept descriptions can be downloaded at the link:
https://huggingface.co/datasets/AuroraX/LMCD/tree/main

Put the downloaded data into the "data" folder,The file format is as follows:<br>

data/ <br>
├── cross_domain_cold_start/<br>
├── kcs_discription/<br>
├── question_cold_start/<br>
&nbsp;&nbsp;&nbsp;├─── NIPS34/<br>
&nbsp;&nbsp;&nbsp;└─── XES3G5M/<br>


## Step 2. Training configuration
* In config/ds_config.json, you can configure batch size, whether to use offload strategy, zero strategy, etc.

* In config/train_config.yaml, configure different task names, different datasets, epoch size, learning rate, etc.

* If you want to use wandb, please configure the corresponding key in wandb_config.json, and set use_wandb to True in train_config.yaml.


## Step 3.do train

bash run.sh

After each training epoch, it will infer on the test set once, and the result will be printed on the console.
