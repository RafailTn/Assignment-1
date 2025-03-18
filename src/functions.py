# Loading the libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.utils import resample
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_scaler():
    # Load the scaler
    cwd = Path.cwd()
    scaler_path = cwd / "models" / "Standardscaler.pkl"
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

def train_model_and_predict(X_train, y_train, x_test, pipeline, root_path='', filename='', save=False):
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Save the pipeline
    if save:
        model_path = root_path / 'models' / filename
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

def grid_search(model, x, y_true, cv=5, scoring=None, feature_selection=True, fine_tune=False):
    pipeline = create_pipeline(model, scaler=False, feature_selector=None)
    if feature_selection and not fine_tune:
        N_FEATURES_OPTIONS = [10, 30, 50, 95]
        grid_params = [
            {
            'feature_selector': [PCA(), KernelPCA(kernel='poly', degree=3), KernelPCA(kernel='rbf')],
            "feature_selector__n_components": N_FEATURES_OPTIONS
            },
        ]
    elif fine_tune and not feature_selection:
        if model.__class__.__name__ == 'SVR':
            grid_params = [

                {
                    'model': [SVR()],
                    "model__kernel": ['linear', 'poly', 'rbf'],
                    "model__degree": [2, 3, 4, 5],
                    "model__C": np.logspace(-3, 1, 10),
                    "model__gamma": ['scale', 'auto'],
                    "model__coef0": np.logspace(-3, 1, 10),
                    "model__epsilon": np.logspace(-3, 1, 10)
                }
            ]
        elif model.__class__.__name__ == 'ElasticNet':
            grid_params = [
                {
                    'model': [ElasticNet()],
                    "model__alpha": np.logspace(-3, 3, 10),
                    "model__l1_ratio": np.linspace(0, 1, 10)
                }
            ]
        elif model.__class__.__name__ == 'BayesianRidge':
            grid_params = [
                {
                    'model': [BayesianRidge()],
                    "model__alpha_1": np.logspace(-3, 3, 10),
                    "model__alpha_2": np.logspace(-3, 3, 10),
                    "model__lambda_1": np.logspace(-3, 3, 10),
                    "model__lambda_2": np.logspace(-3, 3, 10)
                }
            ]
    grid = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring, n_jobs=1)
    grid.fit(x, y_true)
    return grid

def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()