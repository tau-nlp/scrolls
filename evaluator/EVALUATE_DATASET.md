# SCROLLS - Evaluate Single Dataset

Following are instructions to evaluate predictions and compute metrics for each dataset in the SCROLLS benchmark seperatly (only for the validation splits).


If you have predictions for all of the scrolls datasets, you can go to [Evaluate Benchmark](https://github.com/tau-nlp/scrolls/blob/main/evaluator/EVALUATE_BENCHMARK.md) to evaluate all of them at once.
***

### Requirements
* [Setup](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#setup) environment.
* [Prediction Format](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#prediction-format) expected by the script below.


***

Please set:

* `DATASET_NAME`  to be one of:
    ```
    ["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]
    ```

* `PREDICTIONS_JSON` to be the path to your prediction file (in the format specified in [Prediction Format](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#prediction-format)).
  
* `METRICS_OUTPUT_DIR` to be the path you want the results and errors (if any) will be saved to.
  
Run:

```python
python dataset_evaluator.py --split validation --dataset_name DATASET_NAME --predictions PREDICTIONS_JSON  --metrics_output_dir METRICS_OUTPUT_DIR
```