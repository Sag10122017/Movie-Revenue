# Movie-Evaluation

Our project has two main features: predicting movie revenue based on their data and classifying them into one of five revenue classes with an interval of 20 million dollars.

## UTILITIES

1. Clone our repository: `git clone https://github.com/Sag10122017/Movie-Revenue.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the project: `python app.py`

## Project Structure

### Data Folder
This folder contains the database used in our project.

### Classification Folder
This folder contains the code relevant to the classification models:
1. Decision Tree algorithm (`Decision_tree.ipynb`)
2. Neural Network algorithm (`neural_network.ipynb`)
3. Random Forest algorithm (`RandomForestClassify.ipynb`)
4. SVC kernel RBF algorithm (`SVC_kernal_RBF.ipynb`)
5. XGBoost algorithm (`XGBoost.ipynb`)

### Regression Folder
This folder contains the code for the regression models:
1. Ridge Regression algorithm (`boxoffice_RidgeRegression.ipynb`)
2. SVM kernel RBF algorithm (`boxoffice_SVM_rbf.ipynb`)
3. Decision Tree Regression algorithm (`decisiontree.ipynb`)
4. Gradient Boosted Decision Trees (`gbdt-model-all-features.ipynb`)
5. Random Forest algorithm (`randomforest.ipynb`)
6. XGBoost algorithm (`xgboost_model_all_features.ipynb`)

### Model Folder
This folder stores pre-trained models for both classification and regression tasks.

### app.py
To see our prediction or classification results, run `app.py`. Then execute `streamlit run <YOUR_FILE_PATH>` in the command line to see the app in action.
For example, you can run:
`streamlit run 'c:/Code/IT3190E_Group_32/Source Code/app.py'`

## Before You Run Our Project

1. Please install all necessary libraries listed in `requirements.txt`. If your `scikit-learn` library version is not 1.5.0, you will not be able to run `app.py` because all pre-trained models use `scikit-learn==1.5.0`.
2. Update all file paths to match the directories on your own system.
3. To run `SVC_kernal_RBF.ipynb`, you need to install `scikit-learn==1.2.2` and the `imbalanced-learn` library on your computer, as `imbalanced-learn` is not compatible with `scikit-learn==1.5.0`.

