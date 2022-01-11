# SCROLLS

This repository contains the official code of the paper: ["SCROLLS: Standardized CompaRison Over Long Language Sequences"](https://arxiv.org/abs/2201.03533).

### Links
- [Official Website](scrolls-benchmark.com)  
- [Baselines](https://github.com/eladsegal/longer/tree/clean/baselines)  
- [Evaluator](https://github.com/eladsegal/longer/tree/clean/evaluator)  

### Citation
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
  - [GovReport](https://scrolls-tau.s3.us-east-2.amazonaws.com/gov_report.zip)
  - [SummScreenFD](https://scrolls-tau.s3.us-east-2.amazonaws.com/summ_screen_fd.zip)
  - [QMSum](https://scrolls-tau.s3.us-east-2.amazonaws.com/qmsum.zip)
  - [NarrativeQA](https://scrolls-tau.s3.us-east-2.amazonaws.com/narrative_qa.zip)
  - [Qasper](https://scrolls-tau.s3.us-east-2.amazonaws.com/qasper.zip)
  - [QuALITY](https://scrolls-tau.s3.us-east-2.amazonaws.com/quality.zip)
  - [ContractNLI](https://scrolls-tau.s3.us-east-2.amazonaws.com/contract_nli.zip)
