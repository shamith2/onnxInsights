# Script for profiling onnx model

import os
from pathlib import Path

import onnxruntime
from onnxInsights import ONNXProfiler, memoryView

root = Path(__file__).parents[2].resolve()
workspace = Path(__file__).parent.resolve()


def get_shapes(onnx_model_path, params: dict, verbose: bool = False):
    """
    Get input and output shapes for LLMs
    """
    dummy_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )

    if verbose:
        print("Inputs: ")
        for _input in dummy_session.get_inputs():
            print(_input.name, _input.shape, _input.type)
        
        print("Outputs: ")
        for output in dummy_session.get_outputs():
            print(output.name, output.shape, output.type)


    for i, _input in enumerate(dummy_session.get_inputs()):
        if 'past_key_values' in _input.name:
            params['NON_CACHE_INPUT_INDEX'] = i

            params['KV_CHANNELS'] = _input.shape[1]
            params['KV_EMBED_SIZE'] = _input.shape[-1]

            break

    params['NUM_LAYERS'] = (len(dummy_session.get_inputs()) - params['NON_CACHE_INPUT_INDEX']) // 2
    params['OUTPUT_EMBED_SIZE'] = dummy_session.get_outputs()[0].shape[-1]

    assert params['NUM_LAYERS'] * 2 == len(dummy_session.get_outputs()) - 1

    return params


def shape_infer(cls, params: dict):
    """
    Prepare input and output shapes and do shape infer for LLMs
    """
    params = get_shapes(params['PATH_TO_ONNX_MODEL'], params, verbose=False)

    assert params['NON_CACHE_INPUT_INDEX'] <= 3, "shape_infer expects model to have 2-3 inputs apart from KV Cache"

    params['SEQ_LEN'] = params['SEQ_LEN'] if params['PHASE'] == 'PREFILL' else 1

    if 'DECODE' in args['PHASE']:
        try:
            CACHE_LEN = int(args['PHASE'][-1])

        except Exception as _:
            CACHE_LEN = args['MAX_LEN'] - 1

    else:
        CACHE_LEN = 1

    # inputs_shape := [input_ids, attention_mask]
    input_shapes = [(params['BATCH_SIZE'], params['SEQ_LEN']), (params['BATCH_SIZE'], params['MAX_LEN'])]

    # cache_position or position_ids
    if params['NON_CACHE_INPUT_INDEX'] > 2:
        input_shapes.append((params['BATCH_SIZE'], params['SEQ_LEN']))

    for _ in range(params['NUM_LAYERS']):
        input_shapes.append((params['BATCH_SIZE'], params['KV_CHANNELS'], CACHE_LEN, params['KV_EMBED_SIZE'])) # for key
        input_shapes.append((params['BATCH_SIZE'], params['KV_CHANNELS'], CACHE_LEN, params['KV_EMBED_SIZE'])) # for value

    output_shapes = [(params['BATCH_SIZE'], params['SEQ_LEN'], params['OUTPUT_EMBED_SIZE'])]

    for _ in range(params['NUM_LAYERS']):
        output_shapes.append((params['BATCH_SIZE'], params['KV_CHANNELS'], CACHE_LEN + 1, params['KV_EMBED_SIZE'])) # for key
        output_shapes.append((params['BATCH_SIZE'], params['KV_CHANNELS'], CACHE_LEN + 1, params['KV_EMBED_SIZE'])) # for value

    inferred_onnx_model_path = cls.shapeInfer(
        params['PATH_TO_ONNX_MODEL'],
        None,
        input_shapes,
        output_shapes
    )

    return inferred_onnx_model_path


if __name__ == '__main__':
    args = {
        # user-defined params: for llms
        'PHASE': 'DECODEN',
        'BATCH_SIZE': 1,
        'SEQ_LEN': 1024,
        'MAX_LEN': 2048,

        # model specific params
        'MODEL_DIR': 'gemma-1.1-2b-it-fp16-onnx',
        'PATH_TO_ONNX_MODEL': os.path.join(workspace, 'models', 'gemma-1.1-2b-it-fp16-onnx', 'rank_0_gemma-1.1-2b-it_decoder_merged_model_fp16.onnx'),
        'GENERATE_MEMORY_VIEW': True
    }

    onnx_p = ONNXProfiler(
        model_name=f"{args['MODEL_DIR']}_{args['PHASE'].lower()}Phase",
        model_dir=args['MODEL_DIR']
    )

    # run profiler
    inferred_onnx_model_path = shape_infer(onnx_p, params=args)
    onnx_p.profileModel(inferred_onnx_model_path)


    if args['GENERATE_MEMORY_VIEW']:
        local_memory_view = memoryView(
            model_dir=args['MODEL_DIR'],
            model_profile=f"{args['MODEL_DIR']}_{args['PHASE'].lower()}Phase_summary.csv",
            outputs_profile=f"{args['MODEL_DIR']}_{args['PHASE'].lower()}Phase_track_output_summary.csv"
        )

        for local_memory_size in [1, 3, 9, 40, 80]: # range(1, 20 + 1, 1):
            score = local_memory_view.run_with_cache(
                local_memory_size=local_memory_size,
                cache_size=0,
                final_outputs=('logits'),
                plot_memory=True
            )

            print("Local Memory Size: {}, Score: {}\n".format(local_memory_size, score))


    # onnx_p.profileModelonCPU(inferred_onnx_model_path)
    # onnx_p.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)
