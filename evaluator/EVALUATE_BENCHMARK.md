# SCROLLS - Evaluate Validation Sets Benchmark

Following are instructions to evaluate predictions and compute metrics for the entire benchmark (validation splits only).

*** 

### Requirements
* [Setup](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#setup) environment.
* Use [Prepare Submission File](https://github.com/tau-nlp/scrolls/blob/main/evaluator/PREPARE_SUBMISSION_FILE.md#) to create a submmision file expected by the script below.

***

Please set:
* `SUBMISSION_CSV` to be the path to your submission file.
  
* `METRICS_OUTPUT_DIR` to be the path you want the results and errors (if any) will be saved to.

Run:

```python
python benchmark_evaluator.py --split validation --all_predictions SUBMISSION_CSV --metrics_output_dir METRICS_OUTPUT_DIR
```
