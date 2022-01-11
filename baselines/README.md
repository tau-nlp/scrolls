# SCROLLS - Baselines

Following are instructions to reproduce the experiments reported in the paper, on the SCROLLS benchmark.

***

## Quick Links
1. [Setup](#setup)
2. [Train](#train)
3. [Predict](#predict)
4. [Evaluate](#evaluate)

*** 
## Setup
### Requirements

Our experiments were conducted in a Python 3.8 environment.
To clone the repository and set up the environment, please run the following commands:
```
git clone https://github.com/tau-nlp/scrolls.git
cd scrolls/baselines
pip install torch==1.9.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

***
## Train
[scripts/commands/finetune.py](https://github.com/tau-nlp/scrolls/blob/main/baselines/scripts/commands/finetune.py) includes the construction all of the commands used for the experiments in the paper.

To run fine-tuning:
```
python scripts/execute.py scripts/commands/finetune.py {dataset}_{model}
```

The options for `dataset` are:
```
["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]
```

The options for `model` are:
```
["256-bart", "512-bart", "1024-bart", "led-1024", "led-4096", "led-16384"]
```

Notes:
- All of the experiments were run with 8 NVIDIA V100 GPUs. For other GPUs you will probably have to change the number of tokens in batch per GPU, `tokens_bsz` ([here](https://github.com/tau-nlp/scrolls/blob/main/baselines/scripts/commands/finetune.py#L14)), to prevent OOM error.
- Change `num_gpus` ([here](https://github.com/tau-nlp/scrolls/blob/main/baselines/scripts/commands/finetune.py#L15)) to control the number of GPUs used.
- It's possible to evaluate the dataset metrics on the validation set in each epoch by changing `generate_in_eval` ([here](https://github.com/tau-nlp/scrolls/blob/main/baselines/scripts/commands/finetune.py#L19)) to `True`.

***
## Predict
To generate predictions:
```
python scripts/execute.py scripts/commands/generate.py {dataset}_{model}_{split} --checkpoint_path path/to/model/folder
```

The options for `dataset` and `model` are the same as in [training](#train).
The options for `split` are: 
```
["validation", "test"]
```

***
## Evaluate
See [the evaluator](https://github.com/tau-nlp/scrolls/tree/main/evaluator).
