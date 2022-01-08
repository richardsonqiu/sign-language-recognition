import json
import time
import sys
import os
from azureml.core.model import Model
from azureml.core import Workspace

import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

def init():
    global session
    # model = Model.get_model_path(model_name='glamorous-wave-9')
    # ws = Workspace(subscription_id="5aeeb535-21e1-4410-a3d0-539fa06445d7",
    #            resource_group="SignRecognition",
    #            workspace_name="SignRecognition")
    # model = Model(ws, name='glamorous-wave-9',version=1)
    # session = onnxruntime.InferenceSession(model)
    session = onnxruntime.InferenceSession(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "fresh-fire-9.onnx")
    )


def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    return np.array(json.loads(input_data_json)['data']).astype('float32')

def postprocess(result):
    return np.array(result).tolist()

def run(input_data_json):
    try:
        start = time.time()   # start timer
        input_data = preprocess(input_data_json)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: input_data})
        end = time.time()     # stop timer
        return {"result": postprocess(result),
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}