import pandas as pd
import numpy as np
import holidays

def sine_cosine_transform():
    def cyclic_transform(values, period):
        sine = np.sin(2 * np.pi * values / period)
        cosine = np.cos(2 * np.pi * values / period)
        return sine, cosine

    # Assuming df_resampled has a datetime index
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['weekday'] = df_resampled.index.weekday  # Monday=0, Sunday=6
    df_resampled['day_of_year'] = df_resampled.index.dayofyear  # 1-365

    # Sine/Cosine transformation for hour (24-hour cycle)
    df_resampled['hour_sine'], df_resampled['hour_cosine'] = cyclic_transform(
        df_resampled['hour'], 24)

    # Sine/Cosine transformation for weekday (7-day cycle)
    df_resampled['weekday_sine'], df_resampled[
        'weekday_cosine'] = cyclic_transform(df_resampled['weekday'], 7)

    # Sine/Cosine transformation for day of year (365-day cycle)
    df_resampled['day_of_year_sine'], df_resampled[
        'day_of_year_cosine'] = cyclic_transform(df_resampled['day_of_year'],
                                                 365)

    # Dropping the original columns (optional)
    df_resampled.drop(['hour', 'weekday', 'day_of_year'], axis=1, inplace=True)

# Define the file paths
file_2023 = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2023_imputed.csv"
file_2024 = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2024_imputed.csv"

# Read the CSV files into dataframes
df_2023 = pd.read_csv(file_2023, delimiter=';', decimal=',')
df_2024 = pd.read_csv(file_2024, delimiter=';', decimal=',')

# Convert the timestamp to datetime
df_2023['timestamp'] = pd.to_datetime(df_2023['timestamp']) #, dayfirst=True)
df_2024['timestamp'] = pd.to_datetime(df_2024['timestamp']) #, dayfirst=True)

# Concatenate both years into one dataframe
df_combined = pd.concat([df_2023, df_2024])
# reorganize columns
# df_combined.drop(columns=['Zeitstempel'], inplace=True)
df_combined.rename(columns={'electricity_consumption':
                                'consumption [W]'}, inplace=True)

# Set timestamp as index
df_combined.set_index('timestamp', inplace=True)

# Resample the data to 15-minute intervals, taking the mean of the values
df_resampled = df_combined.resample('15T').mean().ffill()

# Apply the sine/cosine transform to the timestamp
sine_cosine_transform()

## mark holidays
# Get German holidays for Baden-Württemberg
german_holidays = holidays.Germany(state='BW')

# Flag holidays in your data
df_resampled['holiday'] = df_resampled.index.to_series().apply(lambda x: x in
                                                                   german_holidays)

# Save the resampled data to a new CSV file
output_file = "data/Energy_23_24.csv"
df_resampled.to_csv(output_file)
