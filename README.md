# SCROLLS

This repository contains the official code of the paper: ["SCROLLS: Standardized CompaRison Over Long Language Sequences"](https://arxiv.org/abs/2201.03533).

Setup instructions are in the [baselines](https://github.com/tau-nlp/scrolls/tree/main/baselines)   and [evaluator](https://github.com/tau-nlp/scrolls/tree/main/evaluator)   folders. 

For the live leaderboard, checkout the [official website](https://scrolls-benchmark.com/). 

***
## Loading the SCROLLS Benchmark Datasets
- via [🤗 Datasets (huggingface/datasets)](https://github.com/huggingface/datasets) library (recommended):

    1. [Installation](https://github.com/huggingface/datasets#installation)
    2. Usage:

        ```python
        from datasets import load_dataset

        qasper_dataset = load_dataset("tau/scrolls", "qasper")
        """
        Options are: ["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]
        """
        ```
- via ZIP files, where each split is in a JSONL file:
  - [GovReport](https://huggingface.co/datasets/tau/scrolls/resolve/main/gov_report.zip)
  - [SummScreenFD](https://huggingface.co/datasets/tau/scrolls/resolve/main/summ_screen_fd.zip)
  - [QMSum](https://huggingface.co/datasets/tau/scrolls/resolve/main/qmsum.zip)
  - [NarrativeQA](https://huggingface.co/datasets/tau/scrolls/resolve/main/narrative_qa.zip)
  - [Qasper](https://huggingface.co/datasets/tau/scrolls/resolve/main/qasper.zip)
  - [QuALITY](https://huggingface.co/datasets/tau/scrolls/resolve/main/quality.zip)
  - [ContractNLI](https://huggingface.co/datasets/tau/scrolls/resolve/main/contract_nli.zip)


## Citation
```
@misc{shaham2022scrolls,
      title={SCROLLS: Standardized CompaRison Over Long Language Sequences}, 
      author={Uri Shaham and Elad Segal and Maor Ivgi and Avia Efrat and Ori Yoran and Adi Haviv and Ankit Gupta and Wenhan Xiong and Mor Geva and Jonathan Berant and Omer Levy},
      year={2022},
      eprint={2201.03533},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
When citing SCROLLS, please make sure to cite all the original dataset papers. [[bibtex]](https://github.com/tau-nlp/scrolls/tree/main/scrolls_datasets.bib)