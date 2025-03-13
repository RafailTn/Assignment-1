# Loading the libraries for the models and csv files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV

dev_set = pd.read_csv('./data/development_final_data.csv', header=0)
test_set = pd.read_csv('./data/evaluation_final_data.csv', header=0)

y = dev_set['BMI']
X = dev_set.drop(columns=['BMI'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = ['ElasticNet', 'SVR', 'BayesianRidge']
def baseline_model_fit(model, save_model=False, save_path=None):
    # Fit the model
    if model == 'ElasticNet':
        regr = ElasticNet(random_state=42)
    elif model == 'SVR':
        regr = SVR(random_state=42)
    elif model == 'BayesianRidge':
        regr = BayesianRidge(random_state=42)
    else:
        raise ValueError("Model not recognized. Choose 'ElasticNet', 'SVR', or 'BayesianRidge'.")
    regr.fit(x_train, y_train)
    predictions = regr.predict(x_test)
    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'Root Mean Squared Error of Baseline {model}: {rmse}')
    # Calculate the MAE
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error of Baseline {model}: {mae}')
    # Calculate the R2 score
    r2 = r2_score(y_test, predictions)
    print(f'R2 Score of Baseline {model}: {r2}')
    # Save the model
    if save_model:
        joblib.dump(regr, save_path)

def main():
    for model in models:
        baseline_model_fit(model, save_model=True, save_path=f'./models/baseline_{model}.pkl')
        
if __name__=="__main__":
    # Call the main function
    main()