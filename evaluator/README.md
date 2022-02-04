# SCROLLS - Evaluator

This folder contains instructions to

* [Evaluate predictions for a single dataset (validation only)](https://github.com/tau-nlp/scrolls/blob/main/evaluator/EVALUATE_DATASET.md) 
* [Evaluate predictions for the entire benchmark (validation only)](https://github.com/tau-nlp/scrolls/blob/main/evaluator/EVALUATE_BENCHMARK.md)
* [Prepare Submission File](https://github.com/tau-nlp/scrolls/blob/main/evaluator/PREPARE_SUBMISSION_FILE.md)
* [Verify Submission File](https://github.com/tau-nlp/scrolls/blob/main/evaluator/VERIFY_SUBMISSION_FILE.md)


*** 
## Setup

Prerequisites:
- python 3.8
- git version >= 2.17.1

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