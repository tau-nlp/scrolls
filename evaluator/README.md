# SCROLLS - Evaluator

Following are instructions to:
1. Evaluate predictions and compute metrics for each dataset in the SCROLLS benchmark, and for the entire benchmark (only for the validation splits).
2. Prepare a submission file for the [SCROLLS leaderboard](https://www.scrolls-benchmark.com/leaderboard).

***

## Quick Links
1. [Setup](#setup)
2. [Predictions Format](#predictions-format)
3. [Dataset-level Evaluation](#dataset-level-evaluation)
4. [Prepare Submission File](#prepare-submission-file)
5. [Benchmark-level Evaluation](#benchmark-level-evaluation)

*** 
## Setup
### Requirements

The evaluation is conducted in a Python 3.8 environment.
To clone the repository and set up the environment, please run the following commands:
```
git clone https://github.com/tau-nlp/scrolls.git
cd scrolls/evaluator
pip install -r requirements.txt
```

***
## Predictions Format
For each dataset, the predictions should be in a JSON file that is a mapping from an ID to a textual prediction:
```JSON
{
    "example_id1": "prediction1",
    "example_id2": "prediction2",
    ...
}
```

***
## Dataset-level Evaluation
```python
python dataset_evaluator.py --split validation --dataset_name DATASET_NAME --predictions PREDICTIONS_JSON  --metrics_output_dir METRICS_OUTPUT_DIR
```

The options for `DATASET_NAME` are:
```
["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]
```

***
## Prepare Submission File
A script for preparing a SCROLLS submission file.

The inputs are paths to 7 json files, one per dataset, in the format described in [Predictions Format](#predictions-format). The output file will be saved in the path specified in `OUTPUT_DIR`. 

```python
python prepare_submission.py \
--gov_report_file GOV_REPORT_PREDS_FILE \
--summ_screen_file SUMM_SCREEN_FD_PREDS_FILE \
--qmsum_file QMSUM_PREDS_FILE \
--narrative_qa_file NARRATIVE_QA_PREDS_FILE \
--qasper_file QASPER_PREDS_FILE \
--quality_file QUALITY_PREDS_FILE \
--contract_nli_file CONTRACT_NLI_PREDS_FILE \
--output_dir OUTPUT_DIR
```

Upload the output file of this script to the [SCROLLS website](https://www.scrolls-benchmark.com) (you may need to login first).


***
## Benchmark-level Evaluation
Set SUBMISSION_CSV to be the path to the submission file you got from [Prepare Submission File](#prepare-submission-file) and run:

```python
python benchmark_evaluator.py --split validation --all_predictions SUBMISSION_CSV --metrics_output_dir METRICS_OUTPUT_DIR
```
