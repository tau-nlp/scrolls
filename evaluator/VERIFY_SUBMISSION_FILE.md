# SCROLLS - Verify Submission File

To generate a submission file, we recommend using our script from  [Prepare Submission File](https://github.com/tau-nlp/scrolls/blob/main/evaluator/PREPARE_SUBMISSION_FILE.md).
***

### Requirements
* [Setup](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#setup) environment.


***
To verify that your submission file is valid, please set:
* `SUBMISSION_CSV`  to be the path to your submission file
* `METRICS_OUTPUT_DIR` to be the path you want the errors (if any) will be saved to.
  
Run:

```python
python benchmark_evaluator.py --verify_only --split test --all_predictions SUBMISSION_CSV --metrics_output_dir METRICS_OUTPUT_DIR
```

If the file is valid, you should expect to see:
```python
The verification was succesful.
```