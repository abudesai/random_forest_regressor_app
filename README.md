Random Forest Regressor inference service using Scikit-Learn+FastAPI on Diamond dataset.

- random forest
- bagging
- ensemble
- python
- feature engine
- FASTAPI
- uvicorn
- docker
- diamond

This is an inference service Random forest regressor using Scikit-Learn.

A Random Forest algorithm fits a number of decision trees on various samples of the dataset and uses mean of all outputs to improve the predictive accuracy and controls over-fitting.

The sample size is controlled by the max_samples parameter and bootstrapping is generally used as default, otherwise the entire dataset is used to build each tree.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - Standard scale data after yeo-johnson
  - Clip values to +/- 4.0 (to remove outliers)

- for target variable
  - No transformations are applied to the target

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine for preprocessing, FastAPI for web service. The web service listens on port 80 and offers two endpoints:

- `/ping`: This is a GET request. It should return a 200 status code when the service is healthy.
- `/predict`: This is a POST request. You should send a JSON formatted data with the feature names and values to be used as the sample for predict. Sample JSON data to send for inference looks as follows:

```
{
  "Id": "Fair",
  "Carat Weight": 2.0,
  "Cut": "Good",
  "Color": "D",
  "Clarity": "FL",
  "Polish": "G",
  "Symmetry": "ID",
  "Report": "GIA"
}
```

Note that the request data should include the sample id (key "Id" in the exhibit above). The service will return the prediction - in this case, the predicted diamond value. Response data will look as follows:

```
{
  "data": {
    "Id": "Fair",
    "Carat Weight": 2,
    "Cut": "Good",
    "Color": "D",
    "Clarity": "FL",
    "Polish": "G",
    "Symmetry": "ID",
    "Report": "GIA"
  },
  "prediction": 49835.4973
}
```

The FastAPI app also validates the data sent for prediction. The input fields must meet certain schema. Check the data_model.py file.
