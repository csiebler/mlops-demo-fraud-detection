import json
import os
import numpy as np
import pandas as pd
import joblib
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from azureml.monitoring import ModelDataCollector

# Automatically generate the swagger interface by providing an data example
input_sample = [{
    "type": "PAYMENT",
    "amount": 9839.64,
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
  }]
output_sample = [[0.7, 0.3]]

def init():
    # Load model
    global model
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    model_path = os.path.join(model_dir, 'aml-demo-model.pkl')
    model = joblib.load(model_path)

    # Setup Data Collection
    global data_collector
    data_collector = ModelDataCollector("best_model", designation="data_collection")

@input_schema('data', StandardPythonParameterType(input_sample))
@output_schema(StandardPythonParameterType(output_sample))
def run(data):
    try:
        # Predict
        df = pd.DataFrame(data)
        proba = model.predict_proba(df)
        result = {"predict_proba": proba.tolist()}
        
        # Collect data
        df_pred = pd.DataFrame(data=proba, columns=["not_fraud", "fraud"])
        data_collector.collect(pd.concat([df, df_pred], axis=1))

        return result
    except Exception as e:
        error = str(e)
        return error
