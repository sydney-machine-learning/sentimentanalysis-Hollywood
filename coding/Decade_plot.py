decade_summary = data.groupby('decade').agg({'numberofwords': 'mean', 'time': 'mean'}).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('Decade')
ax1.set_ylabel('Average Number of Words', color=color)
sns.lineplot(x='decade', y='numberofwords', data=decade_summary, marker='o', ax=ax1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Time (minutes)', color=color)
sns.lineplot(x='decade', y='time', data=decade_summary, marker='o', ax=ax2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Average Number of Words and Time by Decade')
fig.tight_layout()
plt.grid(True)
plt.show()
