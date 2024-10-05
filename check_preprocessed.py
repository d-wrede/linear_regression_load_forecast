import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Energy_23_24.csv', parse_dates=True, index_col=0)

# Resample the data to weekly intervals and calculate the mean for each week
weekly_consumption = df.resample('W').mean()

# Calculate the average consumption across all weeks
average_weekly_consumption = weekly_consumption['Übergabezähler Ges. Wirkleistung [W]'].mean()

# Resample the data to daily intervals and calculate the mean for each day
# daily_consumption = df.resample('D').mean()
daily_consumption = df.resample('D').agg({'Übergabezähler Ges. Wirkleistung '
                                          '[W]': 'mean', 'holiday': 'max',
                                          'atypical': 'max'})

# Add a weekday column (0=Monday, 6=Sunday)
daily_consumption['weekday'] = daily_consumption.index.weekday

# Group by weekday and calculate the mean consumption for each weekday
weekday_avg_consumption = daily_consumption.groupby('weekday')['Übergabezähler Ges. Wirkleistung [W]'].mean()

# Define labels for weekdays
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create subplots grid, e.g., 2 rows, 2 columns for weekly and daily consumption plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot weekly consumption on the first subplot (top-left)
axs[0, 0].plot(weekly_consumption.index, weekly_consumption['Übergabezähler Ges. Wirkleistung [W]'], label='Weekly Consumption')
axs[0, 0].axhline(y=average_weekly_consumption, color='r', linestyle='--', label='Average Weekly Consumption')
axs[0, 0].set_title('Weekly Power Consumption and Average')
axs[0, 0].set_xlabel('Week')
axs[0, 0].set_ylabel('Power Consumption (W)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot weekday combinations on the remaining subplots (top-right, bottom-left, bottom-right)
weekday_combinations = [[0, 1, 2], [3, 4], [5, 6]]  # Example: Monday-Wednesday, Thursday-Friday, Saturday-Sunday
subplot_positions = [(0, 1), (1, 0), (1, 1)]  # Positions in the subplot grid

for idx, combination in enumerate(weekday_combinations):
    ax = axs[subplot_positions[idx]]

    # Plot data for each weekday in the combination
    for i in combination:
        day_data = daily_consumption[daily_consumption['weekday'] == i]
        ax.plot(day_data.index, day_data['Übergabezähler Ges. Wirkleistung [W]'], label=weekday_labels[i])
        ax.axhline(y=weekday_avg_consumption[i], color='r', linestyle='--', label=f'Average {weekday_labels[i]} Consumption')

        # Plot "x" markers for holidays
        holiday_data = day_data[day_data['holiday']]
        ax.scatter(holiday_data.index,
                   holiday_data['Übergabezähler Ges. Wirkleistung [W]'],
                   color='black', marker='x', s=100,
                   label=f"{weekday_labels[i]} (Holiday)")

        # Plot "o" markers for atypical days
        atypical_data = day_data[day_data['atypical']]
        ax.scatter(atypical_data.index,
                   atypical_data['Übergabezähler Ges. Wirkleistung [W]'],
                   color='black', marker='o', facecolors='none', s=100,
                   label=f"{weekday_labels[i]} (Atypical)")

    # Set labels and titles for each subplot
    ax.set_xlabel('Date')
    ax.set_ylabel('Power Consumption (W)')
    ax.set_title(f'Daily Power Consumption for {", ".join([weekday_labels[i] for i in combination])}')
    ax.legend()
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
