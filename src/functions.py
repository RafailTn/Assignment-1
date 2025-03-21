# Loading the libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, r_regression
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
import optuna
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_scaler():
    # Load the scaler
    cwd = Path.cwd()
    root = cwd.parent
    scaler_path = root / "models" / "Standardscaler.pkl"
    scaler = joblib.load(scaler_path)
    return scaler

def create_pipeline(model, scaler=False, feature_selector=None):
    # Initialize the steps for the pipeline
    steps = []
    # Add the scaler if needed
    if scaler:
        # Load the scaler
        scaler = load_scaler()
        steps.append(('scaler', scaler))
    # Add the feature selector
    if feature_selector is not None:
        steps.append(('feature_selector', feature_selector))
    else:
        steps.append(('feature_selector', 'passthrough'))
    # Add the model to the pipeline
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    return pipeline

def train_model_and_predict(X_train, y_train, x_test, pipeline, root_path='', filename='', save=False, default_path=True):
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Save the pipeline
    if save:
        if default_path:
            model_path = root_path / 'models' / filename
            joblib.dump(pipeline, model_path)
        else:
            model_path = os.path.join(root_path, filename)
            joblib.dump(pipeline, model_path)
    # Make predictions
    y_pred = pipeline.predict(x_test)
    return y_pred, pipeline

def kfold(x, y_true, pipeline, n_splits=5, n_repeats=5, random_state=42):
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train_index, test_index in rkf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        y_pred, _ = train_model_and_predict(X_train, y_train, X_test, pipeline)
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return rmse_scores, mae_scores, r2_scores
    
def bootstrap(x, y_true, x_test, y_test, pipeline, n_iter=100, bstrap=False, kf=False):
    scores = {'rmse_scores': [],'mae_scores': [],'r2_scores': []}
    # Bootstrap
    if bstrap and not kf:
        pipeline.fit(x, y_true)
        for i in range(n_iter):
            x_test_boot, y_test_boot = resample(x_test, y_test, replace=True, n_samples=100, random_state=i)
            y_pred = pipeline.predict(x_test_boot)
            rmse = root_mean_squared_error(y_test_boot, y_pred)
            scores['rmse_scores'].append(rmse)
            mae = mean_absolute_error(y_test_boot, y_pred)
            scores['mae_scores'].append(mae)
            r2 = r2_score(y_test_boot, y_pred)
            scores['r2_scores'].append(r2)
    # KFold
    elif kf and not bstrap:
        rmse, mae, r2 = kfold(x, y_true, pipeline, n_splits=5, n_repeats=n_iter)
        scores['rmse_scores'].extend(rmse)
        scores['mae_scores'].extend(mae)
        scores['r2_scores'].extend(r2)
    # Train-test split
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y_true, test_size=0.1, random_state=42)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        scores['rmse_scores'].append(rmse)
        mae = mean_absolute_error(y_val, y_pred)
        scores['mae_scores'].append(mae)
        r2 = r2_score(y_val, y_pred)
        scores['r2_scores'].append(r2)
    return scores

def calculate_statistics(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    median = np.median(scores)
    return mean, std, median

def create_boxplot(scores_list, model_name, metrics, means, stds, medians):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    # If there's only one metric, ensure axes is iterable.
    if n == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=scores_list[i], color="skyblue", ax=ax)
        ax.axhline(means[i], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {means[i]:.3f}')
        ax.axhline(medians[i], color='blue', linestyle='dashed', linewidth=2, label=f'Median: {medians[i]:.3f}')
        ax.axhline(means[i] - stds[i], color='purple', linestyle='dashdot', linewidth=2, label=f'Mean - 1 Std: {(means[i]- stds[i]):.3f}')
        ax.axhline(means[i] + stds[i], color='purple', linestyle='dashdot', linewidth=2, label=f'Mean + 1 Std: {(means[i]+ stds[i]):.3f}')
        ax.set_xlabel("Samples")
        ax.set_ylabel(f"{metric} Score")
        ax.set_title(f"{metric} for {model_name}")
        ax.legend()
    plt.tight_layout()
    plt.show()

# Define the grid search function, for feature selection
# Here only 2 methods are used, PCA and KernelPCA
def grid_search(model, x, y_true, cv=5, scoring=None):
    pipeline = create_pipeline(model, scaler=False, feature_selector=None)
    N_FEATURES_OPTIONS = [10, 30, 50, 95]
    grid_params = [
        {
        'feature_selector': [PCA(), KernelPCA(kernel='poly', degree=3), KernelPCA(kernel='rbf')],
        "feature_selector__n_components": N_FEATURES_OPTIONS
        },
    ]
    grid = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring, n_jobs=1)
    grid.fit(x, y_true)
    return grid

# Optuna objective function for searching the optimal feature selection technique per model
def optuna_dim_reduction(trial, model, x, y_true):
    # Define the feature selection method
    method = trial.suggest_categorical('method', ['PCA', 'KernelPCA', 'SelectKBest'])
    if method == 'PCA':
        # For PCA, suggest the number of components
        n_components = trial.suggest_int('n_components', 1, 95)
        pipeline = create_pipeline(model, scaler=False, feature_selector=PCA(n_components=n_components))
    elif method == 'KernelPCA':
        # For KernelPCA, suggest the kernel type and number of components
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
        if kernel == 'poly':
            #  For polynomial kernel, suggest the degree
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3
        n_components = trial.suggest_int('n_components', 1, 95)
        pipeline = create_pipeline(model, scaler=False, feature_selector=KernelPCA(kernel=kernel, degree=degree, n_components=n_components))
    else:
        # For SelectKBest, suggest the scoring function and number of features
        sk_method = trial.suggest_categorical('sk_method', ['mutual_info_regression', 'r_regression'])
        if sk_method == 'mutual_info_regression':
            score_func = mutual_info_regression
        else:
            score_func = r_regression
        k = trial.suggest_int('k', 1, 95)
        pipeline = create_pipeline(model, scaler=False, feature_selector=SelectKBest(score_func=score_func, k=k))
    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_true, cv=cv, scoring="neg_root_mean_squared_error")
    return np.mean(scores)
    
# Optuna objective function for hyperparameter tuning with given feature selection technique
def optuna_objective(trial, model, x, y_true):
    # For ElasticNet, suggest the alpha and l1_ratio parameters
    if model.__class__.__name__ == 'ElasticNet':
        alpha = trial.suggest_float('alpha', 0.1, 10.0)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        # Create the pipeline with the feature selector that was suggested in the previous step
        pipeline = create_pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio), scaler=False, feature_selector=PCA(n_components=90))
    # For BayesianRidge, suggest the alpha_1, lambda_1, alpha_2, and lambda_2 parameters
    elif model.__class__.__name__ == 'BayesianRidge':
        alpha = trial.suggest_float('alpha_1', 1e-8, 1e-4, log=True)
        lambda_1 = trial.suggest_float('lambda_1', 1e-8, 1e-4, log=True)
        alpha_2 = trial.suggest_float('alpha_2', 1e-8, 1e-4, log=True)
        lambda_2 = trial.suggest_float('lambda_2', 1e-8, 1e-4, log=True)
        pipeline = create_pipeline(BayesianRidge(alpha_1=alpha, lambda_1=lambda_1, alpha_2=alpha_2, lambda_2=lambda_2), scaler=False, feature_selector=SelectKBest(score_func=mutual_info_regression, k=81))
    # For SVR, suggest the C, gamma, epsilon, and kernel parameters
    else:
        C_term = trial.suggest_float('C', 0.1, 10.0)
        gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
        epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
        # For polynomial kernel, suggest the degree and coef0 parameters
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', 0.0, 1.0)
        else:
            degree = 1
            coef0 = 0.0
        pipeline = create_pipeline(SVR(C=C_term, kernel=kernel, degree=degree, coef0=coef0, gamma=gamma, epsilon=epsilon), scaler=False, feature_selector=PCA(n_components=82))        
    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_true, cv=cv, scoring="neg_root_mean_squared_error")
    return np.mean(scores)

# Load the final model and make predictions on the defined dataframe
def bmi_pred(df_path):
    # Get the working directory
    cwd = Path.cwd()
    root = cwd.parent
    # Navigate to the models folder and find the winner model
    winner = os.path.join(root, "models/final_models/winner.pkl")
    df = pd.read_csv(df_path)
    # Use only the microbial data as features
    final_df = df.drop(columns=['Unnamed: 0', 'Experiment type', 'Disease MESH ID', 'Sex', 'Project ID', 'Host age'])
    x = final_df.drop(columns=['BMI'])
    # Load the pipeline
    pipeline = joblib.load(winner)
    # Make predictions
    y_pred = pipeline.predict(x)
    return y_pred




def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()