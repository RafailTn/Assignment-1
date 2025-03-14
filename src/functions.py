# Loading the libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def create_pipeline(model, scaler=False, feature_selector=None, grid_params=None, cv=5, scoring=None):
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
    # Add the model to the pipeline
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    # If grid_params is provided, create a GridSearchCV
    if grid_params is not None:
        pipeline = GridSearchCV(pipeline, param_grid=grid_params, cv=cv, scoring=scoring)
    return pipeline

def train_model_and_predict(X_train, y_train, x_test, pipeline, filename, save=False):
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Save the pipeline
    if save:
        joblib.dump(pipeline, 'models/'+filename)
    # Make predictions
    y_pred = pipeline.predict(x_test)
    return y_pred, pipeline

def bootstrap(x, y_true, pipeline, n_iter=100):
    # Implement bootstrapping to calculate the confidence intervals
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    for _ in range(n_iter):
        X_boot, y_boot = resample(x, y_true, replace=True, random_state=42)
        # Identify OOB samples (samples not in X_boot)
        oob_mask = ~X.index.isin(X_boot.index)
        X_oob, y_oob = X[oob_mask], y[oob_mask]      
        # Train model on bootstrapped data
        pipeline.fit(X_boot, y_boot)
        # Predict on OOB samples
        if not X_oob.empty:
            y_pred = model.predict(X_oob)
            # Evaluate on OOB samples
            rmse = mean_squared_error(y_oob, y_pred, squared=False)
            rmse_scores.append(rmse)
            mae = mean_absolute_error(y_oob, y_pred)
            mae_scores.append(mae)
            r2 = r2_score(y_oob, y_pred)
            r2_scores.append(r2)
    return rmse_scores, mae_scores, r2_scores

def caclulate_statitstics(scores):
    # Calculate the mean, median and standard deviation of the scores
    mean = np.mean(scores)
    std = np.std(scores)
    median = np.median(scores)
    # Calculate the confidence intervals
    lower_bound, upper_bound = np.percentile(scores, [2.5, 97.5])
    return mean, std, median, lower_bound, upper_bound

def create_boxplot(scores, model_name, metric, mean, std, median, ci_lower, ci_upper):
    # Create a boxplot
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=scores, color='lightblue', width=0.4)
    # Add mean, median, and CI as separate markers
    plt.axhline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axhline(median, color='blue', linestyle='-', label=f'Median: {median:.2f}')
    plt.axhline(ci_lower, color='green', linestyle='dotted', label=f'95% CI Lower: {ci_lower:.2f}')
    plt.axhline(ci_upper, color='green', linestyle='dotted', label=f'95% CI Upper: {ci_upper:.2f}')
    plt.axhline(mean - std, color='purple', linestyle='dashdot', label=f'Mean - 1 Std: {mean - std:.2f}')
    plt.axhline(mean + std, color='purple', linestyle='dashdot', label=f'Mean + 1 Std: {mean + std:.2f}')
    # Customize the plot
    plt.ylabel(f"{metric} ({model_name})")
    plt.title(f"Boxplot of {metric} ({model_name}) with CI, Mean, Median, and Std")
    plt.legend()
    plt.show()

def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()