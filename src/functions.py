"""
This file does not contain any tests
Additional time would be needed to 
write a proper test file
"""
# Loading the libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, r_regression, mutual_info_classif, f_classif
from sklearn.linear_model import ElasticNet, BayesianRidge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.svm import SVR
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import cross_val_score
import optuna
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List

def load_scaler():
    """
    Loads the standard scaler from the models directory.
    Returns:
        scaler: The standard scaler object.
    """
    # Load the scaler
    cwd = Path.cwd()
    root = cwd.parent
    scaler_path = root / "models" / "Standardscaler.pkl"
    scaler = joblib.load(scaler_path)
    return scaler

def create_pipeline(
    model: object,
    scaler: bool = False,
    feature_selector: Optional[object] = None
):
    """
    Creates a pipeline for the model.
    Args:
        model: The model object.
        scaler: Whether to use a scaler.
        feature_selector: The feature selector object.
    Returns:
        pipeline: The pipeline object.
    """
    # Initialize the steps for the pipeline
    steps = []
    # Add the scaler if needed
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

def train_model_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    pipeline: Pipeline,
    root_path: str = '',
    filename: str = '',
    save: bool = False,
    default_path: bool = True
):
    """
    Trains the model, stores it and makes predictions.
    Args:
        X_train: The training data.
        y_train: The training labels.
        x_test: The test data.
        pipeline: The pipeline object.
        root_path: Assignment-1 folder
        filename: Name of the output file or custom path after root if default_path=False
        save: Whether to save the model
        default_path: whether to save in models (True) or another folder (False)
    """
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Save the pipeline, default path is the models folder
    if save:
        if default_path:
            model_path = root_path / 'models' / filename
            joblib.dump(pipeline, model_path)
        # Save the pipeline to the specified path, this is for the winner model
        else:
            model_path = os.path.join(root_path, filename)
            joblib.dump(pipeline, model_path)
    # Make predictions
    y_pred = pipeline.predict(x_test)
    return y_pred, pipeline

# Repeated KFold cross-validation
def kfold(
    x: pd.DataFrame,
    y_true: pd.Series,
    pipeline: Pipeline,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42
):
    """
    Performs repeated KFold cross-validation.
    Args:
        x: The input data.
        y_true: The true labels.
        pipeline: The pipeline object.
        n_splits: Number of folds
        n_repeats: Number of times kfold runs
    """
    # Initialize lists to store scores
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    # Define the repeated k-fold cross-validation class
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Iterate through the splits
    for train_index, test_index in rkf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        y_pred, _ = train_model_and_predict(X_train, y_train, X_test, pipeline)
        # Calculate the scores
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return rmse_scores, mae_scores, r2_scores

def kfold_classif(
    x: pd.DataFrame,
    y: pd.Series,
    pipeline: object,
    n_splits: int = 5,
    n_repeats: int = 200,
    random_state: int = 42
):
    """
    Same Kfold as before but for classification.
    Args:
        x: The input data.
        y_true: The true labels.
        pipeline: The pipeline object.
        n_splits: Number of folds
        n_repeats: Number of times kfold runs
    """
    accuracies = []
    precicions = []
    recalls = []
    f1s = []
    # Define the repeated k-fold cross validation class
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train_index, test_index in rkf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_pred, _ = train_model_and_predict(X_train, y_train, X_test, pipeline)
        # Calculate the scores
        accuracies.append(accuracy_score(y_test, y_pred))
        precicions.append(precision_score(y_test, y_pred, average="macro", zero_division=1))
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=1))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=1))
    return accuracies, precicions, recalls, f1s


# Function to apply RepeatedKFold cross-validation and bootstrap sampling
# If all are set to False they just run the train-test split once
def bootstrap(
    x: pd.DataFrame,
    y_true: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: Pipeline,
    n_iter: int = 100,
    bstrap: bool = False,
    kf: bool = False,
    classif: bool = False
):
    """
    Performs bootstrap sampling or k-fold cross-validation.
    Args:
        x: The input data.
        y_true: The true labels.
        x_test: The test data.
        y_test: The test labels.
        pipeline: The pipeline object.
        n_iter: The number of iterations for bootstrap sampling or number of repeats.
        bstrap: Whether to perform bootstrap sampling.
        kf: Whether to perform k-fold cross-validation.
        classif: Whether the task is for classification or regression.
    """
    if classif:
        scores = {'Accuracy':[], 'Precision':[], 'Recall':[], 'F1':[]}
    else:
        scores = {'rmse_scores': [],'mae_scores': [],'r2_scores': []}
    # Bootstrap
    if bstrap and not kf:
        pipeline.fit(x, y_true)
        for i in range(n_iter):
            x_test_boot, y_test_boot = resample(x_test, y_test, replace=True, n_samples=100, random_state=i)
            y_pred = pipeline.predict(x_test_boot)
            if classif:
                accuracy = accuracy_score(y_test_boot, y_pred)
                scores['Accuracy'].append(accuracy)
                precision = precision_score(y_test_boot, y_pred, average="macro", zero_division=1)
                scores['Precision'].append(precision)
                recall = recall_score(y_test_boot, y_pred, average="macro", zero_division=1)
                scores['Recall'].append(recall)
                f1 = f1_score(y_test_boot, y_pred, average="macro", zero_division=1)
                scores['F1'].append(f1)
            else:
                rmse = root_mean_squared_error(y_test_boot, y_pred)
                scores['rmse_scores'].append(rmse)
                mae = mean_absolute_error(y_test_boot, y_pred)
                scores['mae_scores'].append(mae)
                r2 = r2_score(y_test_boot, y_pred)
                scores['r2_scores'].append(r2)
    # KFold
    elif kf and not bstrap:
        if classif:
            accuracy, precision, recall, f1 = kfold_classif(x, y_true, pipeline, n_splits=5, n_repeats=n_iter)
            scores['Accuracy'].extend(accuracy)
            scores['F1'].extend(f1)
            scores['Precision'].extend(precision)
            scores['Recall'].extend(recall)
        else:
            rmse, mae, r2 = kfold(x, y_true, pipeline, n_splits=5, n_repeats=n_iter)
            scores['rmse_scores'].extend(rmse)
            scores['mae_scores'].extend(mae)
            scores['r2_scores'].extend(r2)
    return scores

# Function to calculate the mean, standard deviation, and median of a list of scores
def calculate_statistics(scores):
    """
    Calculates the mean, standard deviation, and median of a list of scores.
    Args:
        scores: The list of scores.
    """
    mean = np.mean(scores)
    std = np.std(scores)
    median = np.median(scores)
    return mean, std, median

# Function to create a boxplot of the scores
def create_boxplot(
    scores_list: List[List[float]],
    model_name: str,
    metrics: List[str],
    means: List[float],
    stds: List[float],
    medians: List[float]
):
    """
    Creates a boxplot of the scores.
    Args:
        scores_list: The list of scores.
        model_name: The name of the model.
        metrics: The metrics to plot.
    """
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

# Function that calls all the functions that are used regularly for training and evaluating models
def bootstrap2boxplot(
    x: pd.DataFrame,
    y_true: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: Pipeline,
    n_iter: int = 100,
    bstrap: bool = False,
    kf: bool = False,
    root_path: str = '',
    filename: str = '',
    save: bool = False,
    default_path: bool = True,
    classif: bool = False
):
    """
    Trains the model, stores it and makes predictions.
    Args:
        x: The input data.
        y_true: The true labels.
        x_test: The test data.
        pipeline: The pipeline object.
        n_iter: The number of iterations for bootstrap sampling.
        bstrap: Whether to perform bootstrap sampling.
        kf: Whether to perform k-fold cross-validation.
        root_path: The path to save the model.
        filename: The name of the model.
        save: Whether to save the model.
        default_path: Whether to save the model to the default path.
        classif: Whether the task is a classification or regression one.
    """
    # Train the model and save it to the specified path
    train_model_and_predict(x, y_true, x_test, pipeline, root_path=root_path, filename=filename, save=save, default_path=default_path)
    # Perform bootstrap sampling or k-fold cross-validation
    if classif:
        scores = bootstrap(x, y_true, x_test, y_test, pipeline, n_iter=n_iter, bstrap=bstrap, kf=kf, classif=True)
        accuracies, precisions, recalls, f1s = scores['Accuracy'], scores['Precision'], scores['Recall'], scores['F1']
        mean_accuracy, std_accuracy, median_accuracy = calculate_statistics(accuracies)
        mean_recall, std_recall, median_recall = calculate_statistics(recalls)
        mean_precisions, std_precisions, median_precisions = calculate_statistics(precisions)
        mean_f1, std_f1, median_f1 = calculate_statistics(f1s)
        model_name = pipeline.named_steps['model'].__class__.__name__
        create_boxplot([accuracies, precisions, recalls, f1s], model_name, ["Accuracy", "Precision", "Recall", 'F1'], [mean_accuracy, mean_precisions, mean_recall, mean_f1], [std_accuracy, std_precisions, std_recall, std_f1], [median_accuracy, median_precisions, median_recall, median_f1])
    else:
        scores = bootstrap(x, y_true, x_test, y_test, pipeline, n_iter=n_iter, bstrap=bstrap, kf=kf)
        rmse_scores, mae_scores, r2_scores = scores['rmse_scores'], scores['mae_scores'], scores['r2_scores']
        mean_rmse, std_rmse, median_rmse = calculate_statistics(rmse_scores)
        mean_mae, std_mae , median_mae = calculate_statistics(mae_scores)
        mean_r2, std_r2, median_r2 = calculate_statistics(r2_scores)
        model_name = pipeline.named_steps['model'].__class__.__name__
        create_boxplot([rmse_scores, mae_scores, r2_scores], model_name, ["RMSE", "MAE", "R2"], [mean_rmse, mean_mae, mean_r2], [std_rmse, std_mae, std_r2], [median_rmse, median_mae, median_r2])

# Define the grid search function, for feature selection
# Here only 2 methods are used, PCA and KernelPCA
def grid_search(
    model: object,
    x: pd.DataFrame,
    y_true: pd.Series,
    cv: int = 5,
    scoring: Optional[str] = None
):
    """
    Performs grid search for the optimal feature selection technique.
    Args:
        model: The model object.
        x: The input data.
        y_true: The true labels.
        cv: The number of folds.
        scoring: The scoring metric.
    """
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
def optuna_dim_reduction(
    trial: object,
    model: object,
    x: pd.DataFrame,
    y_true: pd.Series,
    scoring='neg_root_mean_squared_error',
    classif: bool = False
):
    """
    Performs optuna for the optimal feature selection technique per model.
    Args:
        trial: The optuna trial object.
        model: The model object.
        x: The input data.
        y_true: The true labels.
        scoring: score to optimize
    """
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
        if classif:
            # For SelectKBest, suggest the scoring function and number of features
            sk_method = trial.suggest_categorical('sk_method', ['mutual_info_classification', 'f_classification'])
            if sk_method == 'mutual_info_classification':
                score_func = mutual_info_classif
            else:
                score_func = f_classif
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
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_val_score(pipeline, x, y_true, cv=cv, scoring=scoring)
    return np.mean(scores)
    
# Optuna objective function for hyperparameter tuning with given feature selection technique
def optuna_objective(
    trial: object,
    model: object,
    x: pd.DataFrame,
    y_true: pd.Series,
    pipeline: Optional[Pipeline] = None,
    scoring: str = "neg_root_mean_squared_error",
    classif: bool = False
):
    """
    Performs optuna for the optimal hyperparameter tuning per model.
    Args:
        trial: The optuna trial object.
        model: The model object.
        x: The input data.
        y_true: The true labels.
        scoring: score to optimize
        classif: Whether the task is a classification or a regression one.
    """
    if classif:
        if model.__class__.__name__ == 'LogisticRegression':
            c = trial.suggest_float('C', 0.1, 10.0)
            solver = trial.suggest_categorical('solver', ['lbfgs', 'sag', 'saga'])
            # Initialize l1_ratio with default value
            l1_ratio = 0.0
            if solver == 'saga':
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                if penalty == 'elasticnet':
                    l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)
            else:
                penalty = 'l2'
            if pipeline is None:
                pipeline = create_pipeline(LogisticRegression(C=c, solver=solver, penalty=penalty, l1_ratio=l1_ratio), scaler=False, feature_selector=None)
            else:
                pipeline.set_params(model=LogisticRegression(C=c, solver=solver, penalty=penalty, l1_ratio=l1_ratio))
        else:
            var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
            if pipeline is None:
                pipeline = create_pipeline(GaussianNB(var_smoothing=var_smoothing), scaler=False, feature_selector=None)
            else:
                pipeline.set_params(model=GaussianNB(var_smoothing=var_smoothing))
    else:
        # For ElasticNet, suggest the alpha and l1_ratio parameters
        if model.__class__.__name__ == 'ElasticNet':
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            # If only a model is None, then create a new pipeline with only the specified model
            if pipeline is None:
                pipeline = create_pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio), scaler=False, feature_selector=None)
            # If a pipeline is passed, then set the model parameters, otherwise the model is not instantiated properly in optuna
            else:
                pipeline.set_params(model=ElasticNet(alpha=alpha, l1_ratio=l1_ratio))
        # For BayesianRidge, suggest the alpha_1, lambda_1, alpha_2, and lambda_2 parameters
        elif model.__class__.__name__ == 'BayesianRidge':
            alpha = trial.suggest_float('alpha_1', 1e-8, 1e-4, log=True)
            lambda_1 = trial.suggest_float('lambda_1', 1e-8, 1e-4, log=True)
            alpha_2 = trial.suggest_float('alpha_2', 1e-8, 1e-4, log=True)
            lambda_2 = trial.suggest_float('lambda_2', 1e-8, 1e-4, log=True)
            # If only a model is None, then create a new pipeline with only the specified model
            if pipeline is None:
                pipeline = create_pipeline(BayesianRidge(alpha_1=alpha, lambda_1=lambda_1, alpha_2=alpha_2, lambda_2=lambda_2), scaler=False, feature_selector=None)
            # If a pipeline is passed, then set the model parameters, otherwise the model is not instantiated properly in optuna
            else:
                pipeline.set_params(model=BayesianRidge(alpha_1=alpha, lambda_1=lambda_1, alpha_2=alpha_2, lambda_2=lambda_2))
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
            # If only a model is None, then create a new pipeline with only the specified model
            if pipeline is None:
                pipeline = create_pipeline(SVR(C=C_term, kernel=kernel, degree=degree, coef0=coef0, gamma=gamma, epsilon=epsilon), scaler=False, feature_selector=None)
            # If a pipeline is passed, then set the model parameters, otherwise the model is not instantiated properly in optuna
            else:
                pipeline.set_params(model=SVR(C=C_term, kernel=kernel, degree=degree, coef0=coef0, gamma=gamma, epsilon=epsilon))
    # Perform cross-validation
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_val_score(pipeline, x, y_true, cv=cv, scoring=scoring)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return np.mean(scores)

# Load the final model and make predictions on the defined dataframe
def bmi_pred(df_path: str):
    """
    Loads the final model and makes predictions on the defined dataframe.
    Args:
        df_path: The path to the dataframe.
    """
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

def add_bmi_category(df: pd.DataFrame):
    """
    Adds a BMI category column to the dataframe with two categories: normal weight and abnormal weight.
    
    Args:
        df: DataFrame containing a 'BMI' column
        
    Returns:
        DataFrame with an additional 'BMI_Category' column
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    # Define the BMI categories
    # I used chatGPT to find a way to add the second category for the abnormal BMI
    # as it includes both smaller and larger BMI than normal
    # I learned that the tilde operator works as a not and will be using it from now on
    conditions = [df['BMI'].between(18.5, 24.9), ~df['BMI'].between(18.5, 24.9)]
    categories = ['Normal weight', 'Abnormal weight']
    # Create the BMI_Category column
    df['BMI_Category'] = np.select(conditions, categories, default='Abnormal weight')
    return df

def BMI_classif(df):
    pass

def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()