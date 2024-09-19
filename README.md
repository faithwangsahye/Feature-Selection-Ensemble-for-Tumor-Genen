# Feature Selection Ensemble for Tumor Gene Subset Selection

## Overview

This project aims to apply various feature selection algorithms and machine learning models to RNA-seq data from different types of tumors to select the optimal gene subset for classification models. By integrating multiple feature selection methods, this package provides a comprehensive approach for identifying tumor-specific feature gene subsets, thus enhancing the accuracy and interpretability of tumor classification models.

## Features

- **Multiple Feature Selection Algorithms**: The package integrates 14 distinct feature selection algorithms, including:
  - LightGBM
  - XGBoost
  - CatBoost
  - RandomForest
  - AdaBoost
  - Decision Tree
  - ReliefF
  - Fisher Score
  - SPEC
  - Laplacian Score
  - MCFS (Multi-Cluster Feature Selection)
  - Leshy (from ARFS package)
  - f_score (Statistical feature selection)
  - Linear Regression
  
  These algorithms have been carefully selected and implemented to provide diverse methods for selecting key tumor-specific genes.

- **Ensemble Approach**: The package supports an ensemble of feature selection methods, allowing users to combine the results of multiple algorithms to create a robust, highly accurate feature ranking.

- **Optimization for Best Parameters**: A scoring and optimization mechanism is included to fine-tune the ranking and selection process, ensuring the best performance for downstream classification tasks. The optimal parameters are determined by evaluating accuracy through cross-validation.

- **Model Integration**: After feature selection, the package allows users to build machine learning models such as Support Vector Machines (SVM), XGBoost, LightGBM, and others, to validate the selected gene subsets.

- **Cross-Validation and Evaluation**: Cross-validation is used to assess the accuracy and performance of the selected gene subset, with detailed outputs on model performance.

## Installation

To install this package, clone the repository and use the following command:

```bash
pip install .
```

Make sure you have the required dependencies installed. The major dependencies include:

  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost
  - matplotlib
  - skfeature
  - arfs

You can install these dependencies via pip:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib skfeature arfs
```

## Usage

### 1. Importing Data

The data should be provided in a CSV format, where the rows correspond to different samples and the columns represent gene expression levels. A typical usage for importing data would look like this:

```python
from feature_selection_ensemble import data_import

x, y, data, final, x_data, y_data, x_final, y_final, gene_name = data_import('path_to_data.csv')
```

### 2. Applying Feature Selection Algorithms

Once the data is loaded, you can apply multiple feature selection algorithms and combine their outputs. For example:

```python
from feature_selection_ensemble import xgb_model, lgb_model, reliefF_model

importance_list = pd.DataFrame()
importance_list = xgb_model(x_data, y_data, importance_list, gene_name)
importance_list = lgb_model(x_data, y_data, importance_list, gene_name)
importance_list = reliefF_model(x_data, y_data, importance_list, gene_name)
```

3. Optimizing Feature Selection
You can fine-tune the ranking of selected features through an optimization process that maximizes classification accuracy:

```python
from feature_selection_ensemble import process_data, scale_order, Step_in_out_ACC

Scoring = process_data(importance_list)
order, StandardScore = scale_order(Scoring, new_min=0)
current_ACC, selected_features = Step_in_out_ACC(x_data, y_data, order, numb=200, cv_n=5)
```

4. Model Training and Evaluation
After feature selection, train and evaluate a model with the selected features:

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_model = SVC(probability=True)
svm_model.fit(x_data[selected_features], y_data)
predictions = svm_model.predict(x_final[selected_features])
accuracy = accuracy_score(y_final, predictions)
print(f"Accuracy: {accuracy}")
```

5. Ensemble Evaluation
The package also supports evaluating the performance of the ensemble of feature selection methods and providing a final ranking of features:

```python
from feature_selection_ensemble import Integrated_predictions

Integrated_predictions = svm_model.predict(x_final[selected_features])
Integrated_acc = accuracy_score(y_final, Integrated_predictions)
print(f"Integrated Accuracy: {Integrated_acc}")
```

## Future Development

We plan to further enhance the package by adding:

  - **Customizable Interfaces**: Users will be able to customize which feature selection algorithms and machine learning models to include or exclude.
  - **Expanded Dataset Support**: Ensure compatibility with various types of genomic and transcriptomic datasets.
  - **Enhanced Visualization**: Provide visual tools for feature selection and model performance assessment.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues if you encounter any problems or have suggestions for improvements.
