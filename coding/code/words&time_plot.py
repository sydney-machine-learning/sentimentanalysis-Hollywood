import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/mnt/data/data_csv.csv'
data = pd.read_csv(file_path)

data['time'] = data['time'].str.replace('mins', '').astype(float)

data['decade'] = (data['year'] // 10) * 10

decade_summary = data.groupby('decade').agg({'numberofwords': 'mean', 'time': 'mean'}).reset_index()

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(14, 14))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Words', color=color)
sns.lineplot(x='year', y='numberofwords', data=data, marker='o', ax=ax1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time (minutes)', color=color)
sns.scatterplot(x='year', y='time', data=data, ax=ax2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_title('Number of Words and Time by Year')
ax1.grid(True)

color = 'tab:blue'
ax3.set_xlabel('Decade')
ax3.set_ylabel('Average Number of Words', color=color)
sns.lineplot(x='decade', y='numberofwords', data=decade_summary, marker='o', ax=ax3, color=color, ci=None)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax3.twinx()
color = 'tab:red'
ax4.set_ylabel('Average Time (minutes)', color=color)
sns.lineplot(x='decade', y='time', data=decade_summary, marker='o', ax=ax4, color=color, ci=None)
ax4.tick_params(axis='y', labelcolor=color)
ax3.set_title('Average Number of Words and Time by Decade')
ax3.grid(True)

fig.tight_layout()
plt.show()
