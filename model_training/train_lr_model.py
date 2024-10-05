import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

def result_metrics():
    # global i
    # Initialize variables to accumulate the sum of MSE and R² scores
    total_train_mse = 0
    total_test_mse = 0
    total_train_r2 = 0
    total_test_r2 = 0
    # Calculate performance metrics for each horizon (1 to 24) for both train and test sets
    for i in range(24):
        train_mse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])
        test_mse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])

        train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

        # Accumulate the sum for each metric
        total_train_mse += train_mse
        total_test_mse += test_mse
        total_train_r2 += train_r2
        total_test_r2 += test_r2

        print(
            f"Interval {i + 1} - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
        print(
            f"Interval {i + 1} - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")
        print('-----------------------------------')
    # Calculate the average MSE and R² across all intervals
    avg_train_mse = total_train_mse / 24
    avg_test_mse = total_test_mse / 24
    avg_train_r2 = total_train_r2 / 24
    avg_test_r2 = total_test_r2 / 24
    # Print the summary
    print(
        f"Average Train MSE: {avg_train_mse:.2f}, Average Test MSE: {avg_test_mse:.2f}")
    print(
        f"Average Train R²: {avg_train_r2:.2f}, Average Test R²: {avg_test_r2:.2f}")


def compare_actual_vs_predicted(y_test, y_test_pred):
    # global y_test
    # Ensure the index is a DatetimeIndex (this is likely already the case)
    y_test = pd.DataFrame(y_test,
                          index=X_test.index)  # in case y_test doesn't have the correct index
    y_test_pred_df = pd.DataFrame(y_test_pred, index=X_test.index)
    # Choose a specific week from the test set
    subset_start = pd.Timestamp('2024-08-01')  # Replace with an actual date in
    # your dataset
    subset_end = subset_start + pd.Timedelta(days=7)  # Define a 7-day range
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
    plt.show()

def add_lagged_features(df, max_lag_count):
    # Instead of inserting lags one by one, create all lagged columns at once
    lags = [df['consumption [W]'].shift(lag) for lag in
            range(1, max_lag_count + 1)]

    # Create column names for the lags
    lag_col_names = [f'lag_{lag}' for lag in range(1, max_lag_count + 1)]

    # Concatenate the lagged columns with the original dataframe all at once
    df_lags = pd.concat(lags, axis=1)
    df_lags.columns = lag_col_names

    # Concatenate the original dataframe with the lagged dataframe
    df = pd.concat([df, df_lags], axis=1)
    return df

# Custom transformer for feature engineering (rolling mean, lag features, sine/cosine transformations)
# class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, window_size=7, lag1_count=5):
#         self.window_size = window_size
#         self.lag1_count = lag1_count
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         # Add rolling mean and std
#         X['rolling_mean'] = X['consumption [W]'].rolling(window=self.window_size).mean()
#         X['rolling_std'] = X['consumption [W]'].rolling(window=self.window_size).std()
#
#         # Add lagged features
#         for lag in range(1, self.lag1_count + 1):
#             X[f'lag_{lag}'] = X['consumption [W]'].shift(lag)
#
#         # Drop any NaNs introduced by rolling or lagging
#         X.dropna(inplace=True)
#
#         # Sine and cosine transformations for hour and weekday
#         X['hour_sine'] = np.sin(2 * np.pi * X.index.hour / 24)
#         X['hour_cosine'] = np.cos(2 * np.pi * X.index.hour / 24)
#         X['weekday_sine'] = np.sin(2 * np.pi * X.index.weekday / 7)
#         X['weekday_cosine'] = np.cos(2 * np.pi * X.index.weekday / 7)
#
#         return X

# Define the pipeline
# pipeline = Pipeline(steps=[
#     ('feature_engineering', FeatureEngineeringTransformer(window_size=7, lag1_count=130)),
#     ('scaler', StandardScaler()),
#     ('model', LinearRegression())
# ])

# Load the dataset
df = pd.read_csv('../data/Energy_23_24.csv', parse_dates=True, index_col=0)

# Filter out flagged holidays, atypical days, and atypical weeks
# df = df[~(df['holiday'] | df['atypical_day'] | df['atypical_week'])]

# drop the columns that are no longer needed for training the model
# df.drop(['holiday', 'atypical_day', 'atypical_week'], axis=1, inplace=True)

# Define ranges for rolling window sizes and lag features
# rolling_window_sizes = [7, 8, 9, 11]  # Example rolling window sizes
# lag_feature_counts = [110, 120, 130, 140]  # Example lag feature
rolling_window_sizes = [3, 5, 6, 7, 9, 11]
lag_feature_counts = [50, 70, 90, 110, 130, 140, 150]
# sets

start_time = time.time()

# Store results for each combination
results = []

for window_size in rolling_window_sizes:
    # calculate a rolling mean of the filtered data
    df['rolling_mean'] = df['consumption [W]'].rolling(
        window=window_size).mean()
    # rolling std deviation
    df['rolling_std'] = df['consumption [W]'].rolling(
        window=window_size).std()

    for lag_count in lag_feature_counts:
        start_run_time = time.time()

        # Add lagged features
        df_run = add_lagged_features(df, lag_count)

        # Drop any rows with missing values (due to rolling mean/rolling std)
        df_run.dropna(inplace=True)

        # Create target columns for the next 24 intervals
        target_columns = []
        for i in range(1, 25):  # Predict the next 24 intervals (15 minutes * 24 = 6 hours ahead)
            df_run[f'target_{i}'] = df_run['consumption [W]'].shift(-i)
            target_columns.append(f'target_{i}')

        # Define the features (including dynamically generated lag features)
        features = ['hour_sine', 'hour_cosine', 'weekday_sine',
            'weekday_cosine', 'rolling_mean', 'rolling_std'] + \
            [f'lag_{i}' for i in range(1, lag_count + 1)]

        # Split the data into training and test sets
        X = df_run[features]
        y = df_run[target_columns]

        # Standardize (normalize) the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Drop rows where y contains NaN values and align X accordingly
        y = y.dropna()
        X = X[:len(y)]  # Ensure corresponding rows in X are dropped

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=42, shuffle=False)

        # Initialize and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Initialize lists to store metrics for each interval
        train_mse_list = []
        test_mse_list = []
        train_r2_list = []
        test_r2_list = []

        # Loop through each of the 24 prediction intervals
        for i in range(24):  # There are 24 target columns
            train_mse = mean_squared_error(y_train.iloc[:, i],
                                           y_train_pred[:, i])
            test_mse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

            # Append the results for each interval
            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)
            train_r2_list.append(train_r2)
            test_r2_list.append(test_r2)

        # Calculate the mean performance metrics across all intervals
        mean_train_mse = np.mean(train_mse_list)
        mean_test_mse = np.mean(test_mse_list)
        mean_train_r2 = np.mean(train_r2_list)
        mean_test_r2 = np.mean(test_r2_list)

        # Store the results
        results.append({
            'window_size': window_size,
            'lag1_count': lag_count,
            'mean_train_mse': mean_train_mse,
            'mean_test_mse': mean_test_mse,
            'mean_train_r2': mean_train_r2,
            'mean_test_r2': mean_test_r2
        })

        print(f"Window Size Mean: {window_size}, Lag Count: {lag_count}")
        print(f"run time: {time.time() - start_run_time} seconds")

print(f"Total run time: {time.time() - start_time} seconds")

# Convert results to a DataFrame for easier analysis
resultsss = pd.DataFrame(results)
# sort the results by test MSE
resultsss = resultsss.sort_values('mean_test_mse')

print("results:\n", resultsss)


# result_metrics()
# compare_actual_vs_predicted(y_test, y_test_pred)
