import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# Define the file paths
file_2023 = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2023.csv"
file_2024 = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2024.csv"

# Read the CSV files into dataframes
df = pd.read_csv(file_2023, parse_dates=['timestamp'], dayfirst=True,
                 delimiter=';', decimal=',')
# df_2024 = pd.read_csv(file_2024, parse_dates=True, delimiter=';', decimal=',')
# parse_dates=True, index_col=0

# Rename the columns
df.columns = ['timestamp', 'electricity_consumption']

# Find duplicate timestamps
# duplicate_timestamps = df[df.duplicated(subset='timestamp', keep=False)]
# print(duplicate_timestamps)

# Drop duplicate timestamps
df = df.drop_duplicates(subset='timestamp')

# Set 'timestamp' as the index
df = df.set_index('timestamp')

def detect_file_frequency(df):
    # only consider the first 1000 rows for frequency detection
    df_short = df.head(100)

    # Calculate the time differences between consecutive timestamps
    time_diffs = df_short.index.to_series().diff().dropna()

    # Find the most common time difference (the mode)
    most_common_diff = time_diffs.mode()[0]

    # Print the detected frequency for debugging
    print(f"Detected file frequency: {most_common_diff}")

    return most_common_diff


def detect_gaps(df):
    # Detect the frequency of the file
    freq = detect_file_frequency(df)
    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_reindexed = df.reindex(full_time_index)

    # Create a DataFrame to store gap information
    gaps = []
    is_gap = False
    gap_start = None

    # Loop through reindexed data to detect gaps
    for i, val in enumerate(df_reindexed.index):
        if pd.isna(df_reindexed['electricity_consumption'].iloc[i]):
            if not is_gap:
                # Set gap_start to the last valid point before the gap
                gap_start = df_reindexed.index[i - 1] if i > 0 else df_reindexed.index[i]
                is_gap = True
        else:
            if is_gap:
                # Set gap_end to the first valid point after the gap
                gap_end = df_reindexed.index[i]
                gaps.append({
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration': gap_end - gap_start
                })
                is_gap = False

    gaps_df = pd.DataFrame(gaps)
    return gaps_df, df_reindexed



def plot_gaps(gaps_df):
    # Plotting gaps on a calendar scale and gap length on the y-axis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(gaps_df['gap_start'], gaps_df[
        'duration'].dt.total_seconds() / 3600)  # duration in hours
    plt.title('Gaps over Time (Calendar)')
    plt.xlabel('Gap Start Date')
    plt.ylabel('Gap Duration (Hours)')
    plt.grid(True)

    # Sorting gaps by length for the second plot
    gaps_df_sorted = gaps_df.sort_values('duration', ascending=False)
    # reduce to the first 50 gaps
    gaps_df_sorted = gaps_df_sorted.head(50)

    # Plotting gaps ordered by length
    plt.subplot(1, 2, 2)
    plt.bar(range(len(gaps_df_sorted)),
            gaps_df_sorted['duration'].dt.total_seconds() / 3600)
    plt.title('Gap Duration Ordered by Length')
    plt.xlabel('Gap Number (Ordered by Duration)')
    plt.ylabel('Gap Duration (Hours)')
    plt.tight_layout()
    plt.show()

def plot_gap_fixes(df, gap_end, gap_length, gap_start):
    # Plot 2*gap_length before and after the gap
    plot_start = gap_start - pd.Timedelta(minutes=2 * gap_length)
    plot_end = gap_end + pd.Timedelta(minutes=2 * gap_length)
    plot_data = df.loc[plot_start:plot_end,
                ['electricity_consumption', 'gap_fix']]
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data.index, plot_data['electricity_consumption'],
             label='Original Data', color='blue')
    plt.plot(plot_data.index, plot_data['gap_fix'], label='Imputed Data',
             color='red', linestyle='--')
    plt.axvspan(gap_start, gap_end, color='yellow', alpha=0.3, label='Gap')
    plt.plot(df.loc[gap_start:gap_end, 'orgig_line'], label='Original Line',
                color='green', linestyle='-.')
    plt.plot(df.loc[gap_start:gap_end, 'ref_line'], label='Reference Line',
                color='purple', linestyle='-.')

    # set min max according to orginal_consumption
    plt.ylim(df['orig_consumption'].min(), df['orig_consumption'].max())
    plt.title(f'Gap from {gap_start} to {gap_end}')
    plt.xlabel('Time')
    plt.ylabel('Electricity Consumption')
    plt.legend()
    plt.tight_layout()
    plt.show()


def scale_reference_data(ref_values, gap_start, gap_end, df_imp):
    """Linearly scale the reference values so that they match the start and end points of the gap."""

    # Get the values at the start and end of the gap
    gap_start_value = df_imp.loc[gap_start, 'electricity_consumption']
    gap_end_value = df_imp.loc[gap_end, 'electricity_consumption']

    # Number of points in the original line
    num_points = len(ref_values)

    # Generate linear interpolation for original line
    orig_line = np.linspace(gap_start_value, gap_end_value, num_points)

    # Generate linear interpolation for reference line
    ref_line = np.linspace(ref_values.iloc[0], ref_values.iloc[-1], num_points)

    transfer_value = abs(df_imp['electricity_consumption'].min()) * 2

    # Create the scaling line using the ratio of the two lines
    scaling_line = (orig_line + transfer_value) / (ref_line + transfer_value)

    # Scale the reference values based on the ratio of the two lines
    scaled_ref_values = (ref_values + transfer_value) * scaling_line - transfer_value

    # Check if the scaled values are within the bounds of the original data
    # around the gap
    duration = gap_end - gap_start
    min_value = df_imp.loc[gap_start - 2 * duration:gap_end + 2 * duration, 'electricity_consumption'].min()
    max_value = df_imp.loc[gap_start - 2 * duration:gap_end + 2 * duration, 'electricity_consumption'].max()
    scaled_ref_values = np.clip(scaled_ref_values, min_value, max_value)

    # pass the original line and reference line to the dataframe
    df_imp.loc[gap_start:gap_end, 'orgig_line'] = orig_line
    df_imp.loc[gap_start:gap_end, 'ref_line'] = ref_line

    return scaled_ref_values

def find_valid_reference(gap_start, gap_end, df_imp):
    """Find valid reference data for imputing a gap in the data."""

    def generate_alternating_weeks(n):
        """Generate alternating integers up to ±n for week shifts."""
        return [(-1) ** i * ((i + 1) // 2) for i in range(2 * n)]
    # k = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]  # Alternating
    # between future and
    # past weeks
    k = generate_alternating_weeks(12)
    i = 0

    while i < len(k):
        print(f"i: {i}")
        # Get reference week based on k[i]
        ref_start = gap_start + pd.Timedelta(days=7 * k[i])
        ref_end = gap_end + pd.Timedelta(days=7 * k[i])

        # Check if ref_start and ref_end are within the bounds of the data index
        if ref_start in df_imp.index and ref_end in df_imp.index:
            # Get the reference values
            ref_values = df_imp.loc[ref_start:ref_end, 'electricity_consumption']

        # If valid reference values found, return them
        if not ref_values.isnull().any():
            return ref_values

        print(f"ref_values:\n{ref_values}")

        # Increment counter to move to next week
        i += 1

    # typical restart time of the agent is around 3 am. Hence check for other
    # timeslots in the same day
    i = 0
    while i < len(k):
        print(f"i: {i}")
        # Get reference week based on k[i]
        ref_start = gap_start + pd.Timedelta(hours=1 * k[i])
        ref_end = gap_end + pd.Timedelta(hours=1 * k[i])

        # Check if ref_start and ref_end are within the bounds of the data index
        if ref_start in df_imp.index and ref_end in df_imp.index:
            # Get the reference values
            ref_values = df_imp.loc[ref_start:ref_end, 'electricity_consumption']

        # If valid reference values found, return them
        if not ref_values.isnull().any():
            return ref_values

        print(f"ref_values:\n{ref_values}")

        # Increment counter to move to next week
        i += 1

    # If no valid reference is found, raise an error or handle it
    raise ValueError(
        f"No valid reference data found for gap from {gap_start} to {gap_end}")


def check_reindexing(df_reindexed, freq):
    # Calculate the difference between consecutive timestamps
    time_diffs = df_reindexed.index.to_series().diff().dropna()

    # Check if all time differences match the expected frequency
    expected_diff = pd.Timedelta(seconds=freq.total_seconds())
    if not all(time_diffs == expected_diff):
        print("Warning: The DataFrame has irregular intervals.")
    else:
        print("The DataFrame has consistent 30-second intervals.")

    # Check if there are NaN values, indicating gaps
    if df_reindexed.isnull().any().any():
        print("There are NaN values in the DataFrame (indicating gaps).")
    else:
        print("No NaN values found in the DataFrame.")

    # Show a sample of the DataFrame to inspect
    print(df_reindexed.head(10))
    print(df_reindexed.tail(10))

def remove_long_gaps(df_imp, gaps_df, max_gap_length):
    removed_intervals = []  # Store the intervals that are removed

    for _, gap in gaps_df.iterrows():
        gap_start = gap['gap_start']
        gap_end = gap['gap_end']

        # Gap length in minutes
        gap_length = (gap_end - gap_start)
        # check if the gap is longer than 3.5 days
        if gap_length > pd.Timedelta(days=max_gap_length):
            print(f"Gap from {gap_start} to {gap_end} is too long to impute.")
            removal_start = gap_start.replace(hour=3, minute=0, second=0)
            if removal_start > gap_start:  # If removal_start is after gap_start, go back a day
                removal_start -= pd.Timedelta(days=1)
            loc_removal_start = df_imp.index.get_loc(removal_start)

            # End the cut one week later at 3:00 AM
            removal_end = removal_start + pd.Timedelta(days=7)
            loc_removal_end = df_imp.index.get_loc(removal_end)
            print(f"Cutting data from {removal_start} to {removal_end}")

            # Drop this section of data from the DataFrame
            df_imp.drop(df_imp.loc[removal_start:removal_end].index,
                        inplace=True)

            # Add the removed interval to the list for updating gaps_df
            removed_intervals.append((removal_start, removal_end, loc_removal_start, loc_removal_end))

    # Update gaps_df to remove gaps that fall into the removed intervals
    gaps_df_updated = gaps_df.copy()

    for values in removed_intervals:
        removal_start, removal_end, loc_removal_start, loc_removal_end = values

        # Adjust gaps that are partly contained in the removed interval
        for i, gap in gaps_df_updated.iterrows():
            gap_start = gap['gap_start']
            gap_end = gap['gap_end']

            # Get the previous valid index before removal_start
            previous_valid_index = df_imp.index[loc_removal_start - 1]
            #if loc_removal_start > 0 else None

            # Get the next valid index after removal_end
            next_valid_index = df_imp.index[
                loc_removal_end + 1] if loc_removal_end < len(
                df_imp) - 1 else None

            # If the gap starts before the removal but ends inside it, adjust the gap_end
            if gap_start < removal_start and gap_end > removal_start and gap_end <= removal_end:
                if previous_valid_index:
                    gaps_df_updated.at[i, 'gap_end'] = previous_valid_index

            # If the gap ends after the removal but starts inside it, adjust the gap_start
            elif gap_start >= removal_start and gap_start < removal_end and gap_end > removal_end:
                if next_valid_index:
                    gaps_df_updated.at[i, 'gap_start'] = next_valid_index

        # Remove gaps that are fully contained within the removed interval
        gaps_df_updated = gaps_df_updated[
            ~(
                    (removal_start <= gaps_df_updated['gap_start']) &
                    (gaps_df_updated['gap_end'] <= removal_end)
            )
        ]

    return df_imp, gaps_df_updated


def impute_gaps_with_weekly_seasonality(df_imp, gaps_df):

    # Create new columns
    df_imp['orig_consumption'] = df_imp['electricity_consumption']
    df_imp[['gap_fix', 'orgig_line', 'ref_line']] = np.nan

    # Loop through each gap in the gaps_df
    for _, gap in gaps_df.iterrows():
        gap_start = gap['gap_start']
        gap_end = gap['gap_end']

        # Gap length in minutes
        gap_length = (gap_end - gap_start).total_seconds() / 60

        # Look for data from the week after the gap
        try:
            ref_values = find_valid_reference(gap_start, gap_end, df_imp)

            # Ensure ref_values has enough data
            if len(ref_values) == 0:
                print(
                    f"Reference data is insufficient to fill the gap from "
                    f"{gap_start} to {gap_end}")
                sys.exit(1, "Reference data is insufficient to fill the gap.")

            # Linearly scale the reference values to match the start and end points of the gap
            scaled_ref_values = scale_reference_data(ref_values, gap_start,
                                                     gap_end, df_imp)

            # if NaT values in scaled_ref_values
            if pd.isnull(scaled_ref_values).any():
                print(
                    f"Reference data is insufficient to fill the gap from "
                    f"{gap_start} to {gap_end}")
                continue

            # Impute the missing values
            # Get the positional index of gap_start and gap_end
            start_index = df_imp.index.get_loc(gap_start)
            end_index = df_imp.index.get_loc(gap_end)

            # Use iloc to assign scaled_ref_values from start_index to end_index (inclusive)
            df_imp.iloc[start_index:end_index + 1,
            df_imp.columns.get_loc('gap_fix')] = scaled_ref_values

            # imputing directly in the electricity_consumption column potentially
            # leads to reusing reference data for multiple gaps
            df_imp.iloc[start_index + 1:end_index,
            df_imp.columns.get_loc('electricity_consumption')] = \
                scaled_ref_values[1:-1]

        except KeyError:
            # If no reference week is available, leave the gap as NaN
            print(
                f"No data available to impute gap from {gap_start} to {gap_end}")
            continue

        # filter for gaps with a duration of at least 10 minutes
        if gap_length >= 5:
            # freq = pd.Timedelta(seconds=30)
            # check_reindexing(df_imp, freq)
            plot_gap_fixes(df_imp, gap_end, gap_length, gap_start)

    return df_imp


# Get the gaps
gaps_df, df_reindexed = detect_gaps(df)

print(gaps_df.head())

# Summary statistics
gap_statistics = {
    'total_gaps': len(gaps_df),
    'average_gap_duration': gaps_df['duration'].mean(),
    'longest_gap_duration': gaps_df['duration'].max(),
    'shortest_gap_duration': gaps_df['duration'].min()
}

# Display gap statistics
# print(gap_statistics)

# plot_gaps(gaps_df)

# Remove gaps longer than 3.5 days
max_gap_length = 3.5
df_removed, gaps_df = remove_long_gaps(df_reindexed, gaps_df, max_gap_length)
df_imputed = impute_gaps_with_weekly_seasonality(df_removed, gaps_df)

# drop gap_fix column
df_imputed.drop(columns=['gap_fix'], inplace=True)
# reset index and name the timestamp column
df_imputed.reset_index(inplace=True)
df_imputed.rename(columns={'index': 'timestamp'}, inplace=True)

# safe to csv
file_2023_imputed = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2023_imputed.csv"
file_2024_imputed = "/home/daniel/energy_data/PROJECT_NAME/PROJECT_NAME_2024_imputed.csv"
df_imputed.to_csv(file_2023_imputed, sep=';', decimal=',')
print("Imputed data saved to CSV file.")