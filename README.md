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

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine for preprocessing, FastAPI for web service. The web service provides two endpoints- /ping for health check and /predict for predictions in real time.
