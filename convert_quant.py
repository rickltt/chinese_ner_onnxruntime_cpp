import onnx
import torch
import os
import onnxruntime
import numpy as np
from pprint import pprint
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from model import BertForNER
import onnx
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize(model_path):
    model_fp32 = os.path.join(model_path,"model.onnx")
    model_quant = os.path.join(model_path,"model_quant.onnx")
    onnx_model = onnx.load(model_fp32)
    nodes = [n.name for n in onnx_model.graph.node]
    nodes_to_exclude = [m for m in nodes if 'output' in m]

    quantized_model = quantize_dynamic(model_fp32, model_quant, op_types_to_quantize=['MatMul'], per_channel=True,
                    reduce_range=False, weight_type=QuantType.QUInt8, nodes_to_exclude=nodes_to_exclude)
    
def printinfo(onnx_session):
    print("----------------- 输入部分 -----------------")
    input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
    for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表

        input_info = {
            "name" : input_tensor.name,
            "type" : input_tensor.type,
            "shape": input_tensor.shape,
        }
        pprint(input_info)

    print("----------------- 输出部分 -----------------")
    output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
    for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表

        output_info = {
            "name" : output_tensor.name,
            "type" : output_tensor.type,
            "shape": output_tensor.shape,
        }
        pprint(output_info)

if __name__ == "__main__":
    model_name = "./output/best_checkpoint"
    task = "token-classification"
    onnx_dir = "./onnx_output"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForNER.from_pretrained(model_name)
    main_export(
        model_name,
        monolith=True,
        no_post_process=True,
        task=task,
        output=onnx_dir,
    )
    quantize(onnx_dir)
    ort_session = onnxruntime.InferenceSession(os.path.join(onnx_dir,"model_quant.onnx"))
    printinfo(ort_session)