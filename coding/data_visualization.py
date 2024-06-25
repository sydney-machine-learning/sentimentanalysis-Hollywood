import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("C:/Users/Admin/Desktop/data_csv.csv")
data['year'] = pd.to_datetime(data['year'], format='%Y')

data['time'] = data['time'].str.replace('mins', '').astype(float)


data['decade'] = (data['year'].dt.year // 10) * 10


grouped = data.groupby('decade').agg(
    mean_words=('numberofwords', 'mean'),
    var_words=('numberofwords', 'var'),
    mean_time=('time', 'mean'),
    var_time=('time', 'var')
).reset_index()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(grouped['decade'], grouped['mean_words'], marker='o')
axs[0, 0].set_title('Mean of Number of Words per Decade')
axs[0, 0].set_xlabel('Decade')
axs[0, 0].set_ylabel('Mean Number of Words')

axs[0, 1].plot(grouped['decade'], grouped['var_words'], marker='o', color='orange')
axs[0, 1].set_title('Variance of Number of Words per Decade')
axs[0, 1].set_xlabel('Decade')
axs[0, 1].set_ylabel('Variance of Number of Words')

axs[1, 0].plot(grouped['decade'], grouped['mean_time'], marker='o', color='green')
axs[1, 0].set_title('Mean of Movie Time per Decade')
axs[1, 0].set_xlabel('Decade')
axs[1, 0].set_ylabel('Mean Movie Time (minutes)')

axs[1, 1].plot(grouped['decade'], grouped['var_time'], marker='o', color='red')
axs[1, 1].set_title('Variance of Movie Time per Decade')
axs[1, 1].set_xlabel('Decade')
axs[1, 1].set_ylabel('Variance of Movie Time (minutes)')

plt.tight_layout()
plt.show()

output_path = 'C:/Users/Admin/Desktop/decade_movie_statistics.png'
fig.savefig(output_path)
