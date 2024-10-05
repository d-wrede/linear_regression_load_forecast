import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mark_atypical_days(window_size, atypical_threshold):
    # global daily_consumption, global_std_dev
    # Resample the data to daily intervals and calculate the mean for each day
    daily_consumption = df.resample('D').agg(
        {'consumption [W]': 'mean', 'holiday': 'max'})
    # Add a weekday column (0=Monday, 6=Sunday)
    daily_consumption['weekday'] = daily_consumption.index.weekday
    # Step 1: Calculate rolling mean for each weekday separately (with min_periods=1 to handle edge cases)
    # window_size = 30  # You can adjust this window size based on the granularity of your data
    daily_consumption['rolling_mean'] = daily_consumption.groupby('weekday')[
        'consumption [W]'].transform(
        lambda x: x.rolling(window=window_size, center=True,
                            min_periods=1).mean()
    )
    # Step 2: Center the data by subtracting the rolling mean
    daily_consumption['centered_data'] = daily_consumption[
                                             'consumption [W]'] - \
                                         daily_consumption['rolling_mean']
    # Step 3: Calculate the global standard deviation of the centered data (this is the Z-score denominator)
    global_std_dev = daily_consumption['centered_data'].std()
    # Step 4: Apply the Z-score method (abs(centered_data / global_std_dev))
    daily_consumption['z_score'] = np.abs(
        daily_consumption['centered_data'] / global_std_dev)
    # Step 5: Define a threshold for marking atypical days (e.g., Z-score > 2)
    # atypical_threshold = 1.8  # You can adjust this value
    daily_consumption['atypical_day'] = daily_consumption[
                                        'z_score'] > atypical_threshold
    # Now map the atypical days back to the original data (keep the original frequency)
    df['atypical_day'] = df.index.normalize().isin(
        daily_consumption[daily_consumption['atypical_day']].index)
    # df['weekday'] = df.index.weekday
    df['rolling_mean'] = daily_consumption['rolling_mean']

    # return daily_consumption, global_std_dev, atypical_threshold
    return global_std_dev


def mark_atypical_weeks(window_size, atypical_threshold):
    # Step 1: Resample data to weekly intervals and calculate the mean for each week
    weekly_consumption = df.resample('W').agg(
        {'consumption [W]': 'mean'})

    # Step 2: Calculate the rolling mean for weekly data to smooth the trend
    # window_size = 12
    weekly_consumption['rolling_mean_week'] = weekly_consumption[
        'consumption [W]'].rolling(window=window_size,
                                        center=True, min_periods=1).mean()
    df['rolling_mean_week'] = weekly_consumption['rolling_mean_week'].reindex(
        df.index, method='ffill')

    # Step 3: Center the weekly data by subtracting the rolling mean
    weekly_consumption['centered_data_week'] = weekly_consumption[
                                  'consumption [W]'] - \
                                  weekly_consumption['rolling_mean_week']
    df['centered_data_week'] = weekly_consumption['centered_data_week'].reindex(
        df.index, method='ffill')

    # Step 4: Calculate the global standard deviation based on centered weekly data
    global_weekly_std_dev = weekly_consumption['centered_data_week'].std()

    # Step 5: Apply the Z-score method to the weekly data
    weekly_consumption['z_score'] = np.abs(
        weekly_consumption['centered_data_week'] / global_weekly_std_dev)

    # Step 6: Define a threshold to mark atypical weeks
    weekly_consumption['atypical_week'] = weekly_consumption[
                                         'z_score'] > atypical_threshold

    # Step 7: Mark days that belong to atypical weeks
    df['week_start'] = df.index.to_period('W').start_time
    weekly_consumption.index = weekly_consumption.index.to_period(
        'W').start_time
    atypical_week_map = weekly_consumption['atypical_week'].to_dict()
    df['atypical_week'] = df['week_start'].map(atypical_week_map)
    # drop week_start column
    df.drop(columns=['week_start'], inplace=True)
    # df['atypical_week'] = df['week_start'].isin(
    #     weekly_consumption[weekly_consumption['atypical_week']].index)

    return global_weekly_std_dev


def plot_load_data(global_daily_std_dev, global_weekly_std_dev,
                    day_window_size, week_window_size,
                   atypical_day_threshold, atypical_week_threshold):
    fig, axs = plt.subplots(3, 3, figsize=(18, 10))  # Adjust the grid size as needed

    # Plot weekly consumption with rolling mean and band
    weekly_consumption = df.resample('W').agg({
        'consumption [W]': 'mean',
        'rolling_mean_week': 'mean',
        'centered_data_week': 'mean',
        'atypical_week': 'max'
    })
    axs[0, 0].plot(weekly_consumption.index,
                   weekly_consumption['consumption [W]'] / 1000,
                   label='Weekly Consumption')

    # Plot the rolling mean
    axs[0, 0].plot(weekly_consumption.index,
                   weekly_consumption['rolling_mean_week'] / 1000, linestyle='--',
                   label='Rolling Mean')

    # Plot the band: rolling mean ± global standard deviation
    axs[0, 0].fill_between(weekly_consumption.index,
                           (weekly_consumption['rolling_mean_week'] - global_weekly_std_dev * atypical_week_threshold) / 1000,
                           (weekly_consumption['rolling_mean_week'] + global_weekly_std_dev * atypical_week_threshold) / 1000,
                           color='blue', alpha=0.2, label='± Global Std Dev')

    # Mark "squares" for atypical weeks
    atypical_week_data = weekly_consumption[weekly_consumption['atypical_week']]
    axs[0, 0].scatter(atypical_week_data.index,
                      atypical_week_data['consumption [W]'] / 1000,
                      color='black', marker='s', facecolors='none', s=100,
                      label='Atypical Week')

    # Add labels, title, legend, and grid for the weekly consumption plot
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Power Consumption (kW)')
    axs[0, 0].set_title('Weekly Power Consumption and Atypical Weeks')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot for daily consumption based on weekdays
    daily_data = df.resample('D').agg(
        {'consumption [W]': 'mean',
         'holiday': 'max',
         'atypical_day': 'max',  # Keep 'atypical_day' flag
         'atypical_week': 'max',  # Keep 'atypical_week' flag
         'rolling_mean': 'mean'  # Keep 'rolling_mean' for daily data
         })
    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in range(7):  # Loop through all 7 weekdays
        ax = axs[((i + 1) // 3), (i + 1) % 3]
        day_data = daily_data[daily_data.index.weekday == i]

        # Plot the original daily power consumption data
        ax.plot(day_data.index,
                day_data['consumption [W]'] / 1000,
                label=weekday_labels[i])

        # Plot the rolling mean
        ax.plot(day_data.index, day_data['rolling_mean'] / 1000,
                linestyle='--', label=f'{weekday_labels[i]} Rolling Mean')

        # Plot the band: rolling mean ± global standard deviation
        # global_daily_std_dev = day_data['centered_data'].std()  # Recalculate from df
        ax.fill_between(day_data.index,
                        (day_data['rolling_mean'] - global_daily_std_dev *
                         atypical_day_threshold) / 1000,
                        (day_data['rolling_mean'] + global_daily_std_dev *
                         atypical_day_threshold) / 1000,
                        color='blue', alpha=0.2,
                        label=f'{weekday_labels[i]} ± Global Std Dev')

        # Plot "x" markers for holidays
        holiday_data = day_data[day_data['holiday']]
        ax.scatter(holiday_data.index,
                   holiday_data['consumption [W]'] / 1000,
                   color='black', marker='x', s=100,
                   label=f"{weekday_labels[i]} (Holiday)")

        # Plot "o" markers for atypical days
        atypical_data = day_data[day_data['atypical_day']]
        ax.scatter(atypical_data.index,
                   atypical_data['consumption [W]'] / 1000,
                   color='black', marker='o', facecolors='none', s=100,
                   label=f"{weekday_labels[i]} (Atypical Day)")

        # Flag days that are part of atypical weeks using the 'atypical_week' flag
        atypical_week_days = day_data[day_data['atypical_week']]
        ax.scatter(atypical_week_days.index,
                   atypical_week_days['consumption [W]'] / 1000,
                   color='black', marker='s', facecolors='none', s=100,
                   label=f"{weekday_labels[i]} (Atypical Week)")

        # Set labels and titles for each subplot
        ax.set_xlabel('Date')
        ax.set_ylabel('Power Consumption (kW)')
        ax.set_title(f'Daily Power Consumption for {weekday_labels[i]}')
        ax.legend()
        ax.grid(True)

    # Hide empty subplot if needed
    fig.delaxes(axs[2, 2])
    plt.tight_layout()
    plt.show()


# Load the data
df = pd.read_csv('data/Energy_23_24.csv', parse_dates=True, index_col=0)

day_window_size = 30
day_atypical_threshold = 1.8
global_daily_std_dev = mark_atypical_days(day_window_size,
                                        day_atypical_threshold)

week_window_size = 20
week_atypical_threshold = 1.65
global_weekly_std_dev = mark_atypical_weeks(week_window_size,
                                          week_atypical_threshold)

plot_load_data(global_daily_std_dev, global_weekly_std_dev,
                day_window_size, week_window_size,
               day_atypical_threshold, week_atypical_threshold)

drop_columns = ['rolling_mean', 'rolling_mean_week', 'centered_data_week']
df.drop(columns=drop_columns, inplace=True)

# Save the original data with both holiday and atypical flags
df.to_csv('data/Energy_23_24.csv')