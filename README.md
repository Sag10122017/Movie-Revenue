# Movie-Evaluation
Our project has two feature which include Predicition movie revenue base on their data and Classification them into one of five class bin size which have interval is 20 million dollar.
# UTILITIES
1. Referring to our git clone link: `https://github.com/Sag10122017/Movie-Revenue.git`
2. Referring to our libraries used: `pip install -q -r requirements.txt`
3. Referring to running our project: `app.py`
# Project Structure
## Data folder
Here is the folder in which we store our database for using in our project. 
## Classification folder
In Classification folder, we store code relevent to classification model:
1. Decision tree algorithm (`Decision_tree.ipynb`)
2. Neural network algorithm (`neural_network.ipynb`)
3. Random Forest algorithm (`RandomForestClassify.ipynb`)
4. SVC kernal RBF algorithm (`SVC_kernal_RBF.ipynb`)
5. XGBoost algorithm (`XGBoost.ipynb`)
## Regression folder
In Regression folder, we have 6 algorithm with 6 code consists of: 
1. Ridge Regression algorithm (`boxoffice_RidgeRegression.ipynb`)
2. SVM kernal RBF algorithm (`boxoffice_SVM_rbf.ipynb`)
3. Decision Tree Regression algorithm (`decisiontree.ipynb`)
4. Gradient Boosted Decision Trees (`gbdt-model-all-features.ipynb`)
5. Random Forest algorithm (`randomforest.iynb`)
6. XGBoost algorithm (`xgboost_model_all_features.ipynb`)
## Model folder
Here is the folder in which we store pretrain-model for Classification and Regression model. 
## app.py
If you want to see our result of prediction or classification, run `app.py`. After that, run `streamlit run <YOUR_FILE_PATH>` in command to see app in action.
(For example, you can write `streamlit run 'c:/Code/IT3190E_Group_32/Source Code/app.py'` in command to start steamlit)
# Before you run our project:
1. Please install all necessery library in requirements.txt (If your sklearn library is not equal 1.5.0, you cannot run `app.py` because all pretrain_model is use with `scikit-learn==1.5.0`)
2. Please change all directions into direction in your own system.
3. If you want to run `SVC_kernal_RBF.ipynb`, you need to install `scikit-learn==1.2.2` and `imbalance` library in your computer (Because imbalance is not compatiable with `scikit-learn==1.5.0`)
