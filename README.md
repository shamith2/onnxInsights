# onnxInsights: Insights of ONNX Models for AI Applications
Getting to understand ONNX models better for efficient inference, practical applications and pipelines

- onnxHelpers/onnxProfiler.py = script to statically profile memory and compute requirements of onnx models and modify operators in onnx models

### ONNX Model Static Memory and Compute Profiling:
  * Copy the onnx model to profile to [onnxInsights/scripts/onnxProfile/models](https://github.com/shamith2/onnxInsights/tree/main/scripts/onnxProfile) directory
  * Use ONNXProfiler to profile the model (make sure the model's inputs and outputs are static). For this example, the script to invoke the profiler is located at [onnx_llm_profiling.py](https://github.com/shamith2/onnxInsights/blob/main/scripts/onnxProfile/onnx_llm_profiling.py)
  * The profiling logs will be saved in [onnxInsights/results/onnxProfile/logs](https://github.com/shamith2/onnxInsights/tree/main/results/onnxProfile/logs/llama3_8b_fp16)
