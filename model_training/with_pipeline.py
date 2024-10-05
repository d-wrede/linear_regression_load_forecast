import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib

# to avoid memory issues
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# Custom transformer for feature engineering
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df, window_size=7, agg1_window_size=2,
                 agg2_window_size=2, lag1_count=5, lag_agg1_count=5,
                 lag_agg2_count=5):
        self.window_size = window_size
        self.agg1_window_size = agg1_window_size
        self.agg2_window_size = agg2_window_size
        self.lag1_count = lag1_count
        self.lag_agg1_count = lag_agg1_count
        self.lag_agg2_count = lag_agg2_count
        # aggr_count = 4, resample_count = 4
        # self.aggr_count = aggr_count
        # self.resample_count = resample_count
        self.df = df

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # define Z as empty dataframe
        # X_new = pd.DataFrame()
        # X = X_new

        X['hour_sine'] = self.df['hour_sine']
        X['hour_cosine'] = self.df['hour_cosine']
        X['weekday_sine'] = self.df['weekday_sine']
        X['weekday_cosine'] = self.df['weekday_cosine']
        X['day_of_year_sine'] = self.df['day_of_year_sine']
        X['day_of_year_cosine'] = self.df['day_of_year_cosine']

        # Select appropriate rolling mean and std columns
        X['rolling_mean'] = self.df[f'rolling_mean_{self.window_size}']
        X['rolling_std'] = self.df[f'rolling_std_{self.window_size}']

        # rearrange so that 'consumption [W]' follows the cyclic features
        consumption = X.pop('consumption [W]')
        X['consumption [W]'] = consumption

        # Select lag features based on lag1_count
        lagged_features = {f'lag_{lag}': self.df[f'lag_{lag}'] for lag in
                           range(1, self.lag1_count + 1)}

        # Select lagged aggregated features based on lag_agg1_count
        lagged_agg1_features = {f'lag_agg1_{lag}': self.df[f'lag_agg1_' \
                                f'{lag}_window_{self.agg1_window_size}'] for
                                lag in range(0, self.lag_agg1_count + 1)}
        lagged_agg2_features = {f'lag_agg2_{lag}': self.df[f'lag_agg2_' \
                                f'{lag}_window_{self.agg2_window_size}'] for
                                lag in range(0, self.lag_agg2_count + 1)}

        # Concatenate the lagged features to the transformed dataframe
        X = pd.concat([X, pd.DataFrame(lagged_features, index=X.index),
                       pd.DataFrame(lagged_agg1_features, index=X.index),
                       pd.DataFrame(lagged_agg2_features, index=X.index)],
                      axis=1)
        # ,
        # pd.DataFrame(lagged_agg1_features, index=X.index)]
        # pd.DataFrame(lagged_agg2_features, index=X.index)],

        # print("columns in X:\n", X.columns)

        return X

# Function to load and preprocess the data
def load_and_preprocess_data(file_path, max_lag1, max_agg1_lag,
                             max_agg2_lag, agg1_windows, agg2_windows,
                             rolling_windows):

    df = pd.read_csv(file_path, parse_dates=True, index_col=0)

    # Filter out flagged holidays, atypical days, and weeks
    # df = df[~(df['holiday'] | df['atypical_day'] | df['atypical_week'])]
    #
    # # Drop unnecessary columns
    # df.drop(['holiday', 'atypical_day', 'atypical_week'], axis=1, inplace=True)

    # Add lagged features up to max_lag1 all at once
    lagged_features = {f'lag_{lag}': df['consumption [W]'].shift(lag) for
                       lag in range(1, max_lag1 + 1)}

    df = pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)

    # Add lagged aggregated features for different aggregation windows
    lagged_agg1_features = {}
    for window in agg1_windows:
        for lag in range(0, max_agg1_lag + 1):
            lagged_agg1_features[f'lag_agg1_{lag}_window_{window}'] = \
                df['consumption [W]'].shift(lag).rolling(window=window).mean()

    # Concatenate the lagged features to the dataframe
    df = pd.concat([df, pd.DataFrame(lagged_agg1_features, index=df.index)], axis=1)

    lagged_agg2_features = {}
    for window in agg2_windows:
        for lag in range(0, max_agg2_lag + 1):
            lagged_agg2_features[f'lag_agg2_{lag}_window_{window}'] = \
                df['consumption [W]'].shift(lag).rolling(window=window).mean()

    # Concatenate the lagged features to the dataframe
    df = pd.concat([df, pd.DataFrame(lagged_agg2_features, index=df.index)],
                   axis=1)

    # Precompute rolling means and stds for all rolling windows and concatenate them
    rolling_features = {}
    for window in rolling_windows:
        rolling_features[f'rolling_mean_{window}'] = df[
            'consumption [W]'].rolling(window=window).mean()
        rolling_features[f'rolling_std_{window}'] = df[
            'consumption [W]'].rolling(window=window).std()

    # Concatenate the rolling mean and std columns to the dataframe
    df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)],
                   axis=1)

    # Create target columns for the next 24 intervals (6 hours ahead) all at once
    forecast_hours = 12
    target_columns = {f'target_{i}': df['consumption [W]'].shift(-i) for i in
                      range(1, (forecast_hours * 4 + 1))}

    # Concatenate the new target columns to the existing dataframe
    df = pd.concat([df, pd.DataFrame(target_columns, index=df.index)], axis=1)

    # Drop rows with NaNs after creating the features and target columns
    df.dropna(inplace=True)

    # Split features (X) and targets (y)
    X = df[['consumption [W]']]  # Add any additional features here if needed
    y = df[[f'target_{i}' for i in range(1, (forecast_hours * 4 + 1))]]

    return X, y, df


# Split into training and test sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42,
                                                        shuffle=False)


# Function to define the pipeline
def create_pipeline(df):
    # linear regression
    return Pipeline(steps=[
        ('feature_engineering', FeatureEngineeringTransformer(df=df)),
        ('scaler', StandardScaler()),
        # ('poly', PolynomialFeatures(degree=3, interaction_only=True)),
        # elastic net model:
        # ('model', ElasticNet())
        # ('model', Ridge())
        # ('model', Lasso(warm_start=True))
        # ('model', LinearRegression())
        ('model', RandomForestRegressor(random_state=42))
    ])

    # return Pipeline(steps=[
    #     ('feature_engineering', FeatureEngineeringTransformer(df=df)),
    #     ('scaler', StandardScaler()),
    #     ('model', RandomForestRegressor(random_state=42))
    # ])


def asymmetric_loss(y_true, y_pred, under_penalty=2.0, over_penalty=1.0):
    # Calculate the error
    error = y_true - y_pred

    # Apply different penalties for under- and over-estimations
    loss = np.where(error > 0, under_penalty * error**2,
                    over_penalty * error**2)

    return np.mean(loss)


def peak_load_loss(y_true, y_pred, peak_threshold):
    # Get the top 10% of the actual load values
    high_load_indices = y_true >= 200000
    # np.percentile(y_true, peak_threshold)

    # Calculate the error only on those periods
    error = np.abs(y_true[high_load_indices] - y_pred[high_load_indices])

    return np.mean(error)

# Function to perform Grid Search with cross-validation
def perform_grid_search(pipeline, param_grid, X_train, y_train):
    # Create a custom scorer
    custom_scorer = make_scorer(asymmetric_loss, greater_is_better=False)
    custom_peak_scorer = make_scorer(peak_load_loss, peak_threshold=90)

    grid_search = GridSearchCV(pipeline, param_grid, cv=2,
                               scoring='r2',
                               verbose=2, n_jobs=10)
    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best R² score:", grid_search.best_score_)

    # Get the results into a pandas DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Sort the results by mean test score in descending order
    sorted_results = results_df.sort_values(by='mean_test_score',
                                            ascending=False)

    # Print the top 5 results with their parameters and scores
    top_10_results = sorted_results[['mean_test_score', 'params']].head(10)

    # Ensure full parameter information is shown
    pd.set_option('display.max_colwidth', None)

    print("\nTop 10 results:")
    print(top_10_results)

    # Reset display option after printing
    pd.reset_option('display.max_colwidth')

    return grid_search.best_estimator_, grid_search.best_params_


def visualize_ridge_coefficients(best_model, feature_names):
    """
    Visualize the coefficients of the Ridge regression model.
    The feature_names correspond to the lagged features, and the coefficients
    will show how the model weights each lag.

    :param best_model: The trained Ridge model (best estimator from GridSearchCV).
    :param feature_names: A list of feature names (lagged feature names).
    """

    # Extract coefficients from the model (assuming it's Ridge)
    coefficients = best_model.named_steps['model'].coef_

    # If y has multiple targets (for multi-output regression), take the first output
    if coefficients.ndim > 1:
        coefficients = coefficients[0]

    # statistics_coefficients(coefficients)

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(coefficients)), coefficients, marker='o', linestyle='-', color='b')
    plt.title('Ridge Regression Coefficients for Lagged Features')
    plt.xlabel('Lagged Feature Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    # plt.yscale('log')

    # Highlight the zero-coefficients (features dropped by Lasso)
    zero_coeffs = np.where(coefficients == 0)[0]
    if len(zero_coeffs) > 0:
        plt.scatter(zero_coeffs, coefficients[zero_coeffs], color='red', label='Zero Coefficients')

    plt.legend()
    # plt.show()


def visualize_random_forest_feature_importance(best_model, feature_names):
    # Extract feature importances from the best random forest model
    importances = best_model.named_steps['model'].feature_importances_

    # Sort feature importances in descending order and get the corresponding feature names
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = np.array(feature_names)[indices]

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.barh(sorted_feature_names[:20], sorted_importances[:20], color="b",
             align="center")  # top 20 features
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()  # To display highest importance at the top


def statistics_coefficients(coefficients):
    # Statistical analysis
    total_coefficients = len(coefficients)
    non_zero_coefficients = np.sum(coefficients != 0)
    zero_coefficients = total_coefficients - non_zero_coefficients
    median_coefficients = np.median(coefficients)
    std_coefficients = np.std(coefficients)
    min_coefficients = np.min(coefficients)
    max_coefficients = np.max(coefficients)
    print(f"Total number of coefficients: {total_coefficients}")
    print(f"Number of zero coefficients: {zero_coefficients}")
    print(f"Number of non-zero coefficients: {non_zero_coefficients}")
    print(
        f"Percentage of zero coefficients: {100 * zero_coefficients / total_coefficients:.2f}%")
    print(f"Mean of all coefficients: {np.mean(coefficients):.4f}")
    print(f"Median of all coefficients: {median_coefficients:.4f}")
    print(f"Standard deviation of all coefficients: {std_coefficients:.4f}")
    print(f"Range of coefficients: {min_coefficients:.4f} to {max_coefficients:.4f}")




def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set using MSE and R² score for each predicted interval.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Initialize lists to store metrics for each interval
    mse_list = []
    r2_list = []

    # Loop through each interval (there are 24 target intervals)
    for i in range(y_test.shape[1]):
        # Calculate metrics for the current interval
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

        # Store the results
        mse_list.append(mse)
        r2_list.append(r2)

        # Print metrics for the current interval
        # print(f"Interval {i + 1}:")
        # print(f"  Test MSE: {mse:.4f}")
        # print(f"  Test R²: {r2:.4f}")
        # print('-----------------------------')

    # Optionally: Calculate the mean metrics across all intervals
    mean_mse = sum(mse_list) / len(mse_list)
    mean_r2 = sum(r2_list) / len(r2_list)
    print("Mean Test MSE across all intervals:", mean_mse)
    print("Mean Test R² across all intervals:", mean_r2)


def compare_actual_vs_predicted(X_test, y_test, y_test_pred):
    # global y_test
    # Ensure the index is a DatetimeIndex (this is likely already the case)
    y_test = pd.DataFrame(y_test,
                          index=X_test.index)  # in case y_test doesn't have the correct index
    y_test_pred_df = pd.DataFrame(y_test_pred, index=X_test.index)
    # Choose a specific week from the test set
    subset_start = pd.Timestamp('2024-08-01')  # Replace with an actual date in
    # your dataset
    subset_end = subset_start + pd.Timedelta(days=14)
    # Filter the test data and predictions based on this week
    y_test_subset = y_test.loc[subset_start:subset_end]
    y_test_pred_subset = y_test_pred_df.loc[subset_start:subset_end]
    # Define the step for selecting every 3 hours (3 hours * 4 intervals per hour = 12 intervals)
    step_size = 12
    # Select points every 3 hours from the test set
    selected_indices = np.arange(0, len(y_test_subset), step_size)
    # Plot actual vs predicted for all 24 intervals for each selected point
    # Create a new figure
    plt.figure(figsize=(14, 8))
    # Plot actual values for the entire subset (i.e., one week)
    plt.plot(y_test_subset.index, y_test_subset['target_1'], label='Actual',
             color='blue')
    # Now loop through the selected points every 3 hours and plot the predicted values
    for idx in selected_indices:
        # Ensure there are enough data points to plot 24 intervals
        if idx + 24 <= len(y_test_subset):
            # Get the time range for the next 24 intervals
            pred_time_range = y_test_subset.index[idx:idx + 24]

            # Plot predicted values for the next 24 intervals using the actual time range
            plt.plot(pred_time_range, y_test_pred_subset.iloc[idx, :24].values,
                     label=f'Predicted (Starting at {y_test_subset.index[idx]})',
                     linestyle='--')
    # Add labels, title, legend, and grid
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (W)')
    plt.title('Actual vs Predicted Power Consumption Over One Week')
    # Formatting the x-axis to show the date properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45)
    # plt.legend()
    plt.grid(True)
    # Show the plot
    plt.tight_layout()


def perform_tests(X_test, best_model, best_params, df, y_test):
    best_window_size = best_params['feature_engineering__window_size']
    best_lag_count = best_params['feature_engineering__lag1_count']
    # Manually set the best parameters to the feature transformer for X_test
    transformer = FeatureEngineeringTransformer(df=df,
                                                window_size=best_window_size,
                                                lag_count=best_lag_count)
    # Transform X_test with best hyperparameters
    X_test_transformed = transformer.transform(X_test)
    y_test_aligned = y_test.loc[X_test_transformed.index]
    # Evaluate the best model on the transformed test set
    evaluate_model(best_model, X_test_transformed, y_test_aligned)


# Main workflow function
def main():
    # param_grid = {
    #     'feature_engineering__window_size': [40, 50],
    #     'feature_engineering__lag1_count': [200, 500, 750],
    #     'feature_engineering__lag2_count': [50, 100, 200],
    #     # for Ridge
    #     'model__alpha': [100, 300, 500],
    #
    #     # for RandomForestRegressor
    #     # 'model__n_estimators': [50],
    #     # # Number of trees in the forest
    #     # 'model__max_depth': [10],  # Maximum depth of each tree
    #     # 'model__min_samples_split': [2],
    #     # # Minimum samples required to split a node
    #     # 'model__min_samples_leaf': [2],
    #     # Minimum samples required at each leaf node
    # }

    param_grid = {
        'feature_engineering__window_size': [40],
        'feature_engineering__agg1_window_size': [2],
        'feature_engineering__agg2_window_size': [6],
        'feature_engineering__lag1_count': [500, 750],
        'feature_engineering__lag_agg1_count': [25],
        'feature_engineering__lag_agg2_count': [13],
        # for Ridge
        # 'model__alpha': [700, 850, 1000, 1200],  # ElasticNet uses alpha for
        # regularization strength
        # for ElasticNet
        # 'model__l1_ratio': [0.1, 0.5, 0.9],  # l1_ratio to control balance between Lasso and Ridge

        # for Random Forest
        'model__n_estimators': [100],
        # Number of trees in the forest
        'model__max_depth': [10, 15],  # Maximum depth of each tree
        'model__min_samples_split': [5],
        # Minimum samples required to split a node
        'model__min_samples_leaf': [5, 10],
        # 'max_features'
    }

    rolling_windows = param_grid['feature_engineering__window_size']
    agg1_windows = param_grid['feature_engineering__agg1_window_size']
    agg2_windows = param_grid['feature_engineering__agg2_window_size']
    max_lag1 = max(param_grid['feature_engineering__lag1_count'])
    max_agg1_lag = max(param_grid['feature_engineering__lag_agg1_count'])
    max_agg2_lag = max(param_grid['feature_engineering__lag_agg2_count'])

    # Load and preprocess data
    filepath = 'data/Energy_23_24.csv'
    X, y, df = load_and_preprocess_data(filepath, max_lag1, max_agg1_lag,
                                        max_agg2_lag, agg1_windows,
                                        agg2_windows, rolling_windows)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    # X_train = X
    # y_train = y

    # Create the pipeline
    pipeline = create_pipeline(df)

    # Perform grid search
    best_model, best_params = perform_grid_search(pipeline, param_grid,
                                                 X_train, y_train)

    # save the best model and best parameters
    model_name = best_model.named_steps['model'].__class__.__name__
    # get current date
    today = date.today()
    filename = f"trained_models/{model_name}_{today}"
    joblib.dump(best_model, f'{filename}_model.pkl')
    joblib.dump(best_params, f'{filename}_params.pkl')

    # Example feature names for 750 lagged features
    feature_names = [f'lag_{i}' for i in range(1, 751)]

    # Visualize the coefficients
    # visualize_ridge_coefficients(best_model, feature_names)
    visualize_random_forest_feature_importance(best_model, feature_names)

    # compare with predicted values
    y_test_pred = best_model.predict(X_test)
    compare_actual_vs_predicted(X_test, y_test, y_test_pred)
    plt.show()

    # perform_tests(X_test, best_model, best_params, df, y_test)


if __name__ == "__main__":
    main()