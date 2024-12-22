# onnxInsights: Insights of ONNX Models for AI Applications
Getting to understand ONNX models better for efficient inference, practical applications and pipelines

- onnxHelpers/onnxProfiler.py = script to statically profile memory and compute requirements of onnx models and modify operators in onnx models

### ONNX Model Static Memory and Compute Profiling:
  * #### Llama3 8B FP16 model:
    * Copy the onnx model to profile to [onnxInsights/scripts/onnxProfile/models](https://github.com/shamith2/onnxInsights/tree/main/scripts/onnxProfile) directory. For this example, the onnx model is downloaded from https://huggingface.co/aless2212/Meta-Llama-3-8B-Instruct-onnx-fp16
    * Use ONNXProfiler to profile the model (make sure the model's inputs and outputs are static). For this example, the script to invoke the profiler is located at [onnx_llama_profiling.py](https://github.com/shamith2/onnxInsights/blob/main/scripts/onnxProfile/onnx_llama_profiling.py)
    * The profiling logs will be saved in [onnxInsights/results/onnxProfile/logs](https://github.com/shamith2/onnxInsights/tree/main/results/onnxProfile/logs/llama3_8b_fp16)
    * Example log: Profiling Operator-wise Grouped Summary in Decode Phase: [profile-grouped-summary-csv](https://github.com/shamith2/onnxInsights/blob/main/results/onnxProfile/logs/llama3_8b_fp16/llama3_8b_fp16_decodenPhase_grouped_summary.csv)
