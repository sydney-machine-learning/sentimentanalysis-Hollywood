import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data from the CSV file
data = pd.read_csv('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/roberta_df.csv')
data2 = pd.read_csv('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/data_csv.csv')

# Extract relevant columns and sum the counts for each emotion
emotion_columns = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Official report', 'Joking']
emotion_counts = data[emotion_columns].sum()

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(emotion_counts.index, emotion_counts.values, color='blue')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.title('Counts of Different Emotions')


weights = {
    'Optimistic': 3,
    'Thankful': 2,
    'Empathetic': 0,
    'Pessimistic': -3,
    'Anxious': -1,
    'Sad': -2,
    'Annoyed': -1,
    'Denial': -4,
    'Official report': 0,
    'Joking': 1
}

data['weighted_score'] = (
    data['Optimistic'] * weights['Optimistic'] +
    data['Thankful'] * weights['Thankful'] +
    data['Empathetic'] * weights['Empathetic'] +
    data['Pessimistic'] * weights['Pessimistic'] +
    data['Anxious'] * weights['Anxious'] +
    data['Sad'] * weights['Sad'] +
    data['Annoyed'] * weights['Annoyed'] +
    data['Denial'] * weights['Denial'] +
    data['Official report'] * weights['Official report'] +
    data['Joking'] * weights['Joking']
)

data['year']=data2['year']
# Group by year and calculate the average weighted score for each year
average_scores_by_year = data.groupby('year')['weighted_score'].mean().reset_index()

# Plot the average weighted score trend over the years
plt.figure(figsize=(10, 6))
plt.plot(average_scores_by_year['year'], average_scores_by_year['weighted_score'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Weighted Score')
plt.title('Average Weighted Sentiment Score Trend Over Years')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# heatmap
heatdata=data.iloc[:,3:12]

# plt.figure(figsize=(10, 8))
# sns.heatmap(data.iloc[:,3:12], annot=True, cmap='coolwarm', cbar=True, square=True)
# plt.title('Heatmap of Emotions Data')
# plt.show()
columns = data.iloc[:,3:12].columns

# 初始化共现矩阵
co_occurrence_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)

# 计算每对情绪的共现次数
for i in range(len(heatdata)):
    for emotion1 in columns:
        for emotion2 in columns:
            if heatdata.at[i, emotion1] > 0 and heatdata.at[i, emotion2] > 0:
                co_occurrence_matrix.at[emotion1, emotion2] += 1

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title('Co-occurrence Heatmap of Emotions')
plt.show()