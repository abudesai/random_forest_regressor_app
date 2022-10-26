

import os, warnings, sys
import io
import traceback
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
from flask import Flask, request, Response

sys.path.insert(0, './../../')


import app.src.preprocessing.pipelines as pipelines
import app.src.model.regressor as regressor
import app.src.utils as utils


test_data_path = "./../data/processed_data/testing/"
schema_path = "./../data/data_config/"
model_path = "./model/artifacts/"

# load preprocessors
inputs_pipeline, target_pipeline = pipelines.load_preprocessors(model_path)

# load model 
model = regressor.load_model(model_path)

# get data schema
data_schema = utils.get_data_schema(schema_path)

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. """
    status = 200
    response=f"Hello - I am {regressor.MODEL_NAME} model and I am at your service!"
    return Response(response=response, status=status, mimetype="application/json")




@app.route("/predict", methods=["POST"])
def predict():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if request.content_type == "text/csv":
        data = request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s)
        
    else:                
        return Response(
            response="This predictor only supports CSV data", 
            status=415, mimetype="text/plain"
        )

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try: 
        
        # preprocess test inputs
        processed_test_inputs = inputs_pipeline.transform(data)
        
        # make predictions
        predictions_arr = model.predict(processed_test_inputs)
        
        # inverse transform predictions
        rescaled_preds = pipelines.get_inverse_transform_on_preds(target_pipeline, predictions_arr)
        
        preds_df = data[[data_schema["inputDatasets"]["regressionBaseMainInput"]["idField2"]]]
        preds_df["predictions"] = np.round(rescaled_preds, 4) 
        
        # Convert from dataframe to CSV
        out = io.StringIO()
        preds_df.to_csv(out, index=False)
        result = out.getvalue()

        return Response(response=result, status=200, mimetype="text/csv")

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during inference: ' + str(err) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        
        return Response(
            response="Error generating predictions.", 
            status=400, mimetype="text/plain"
        )



if __name__ == "__main__":
    app.run()