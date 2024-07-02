import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data from the CSV file
data = pd.read_csv('C:/Users/Admin/Desktop/roberta_df.csv')
data2 = pd.read_csv('C:/Users/Admin/Desktop/data_csv.csv')

# Extract relevant columns and sum the counts for each emotion
emotion_columns = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Official report', 'Joking']
emotion_counts = data[emotion_columns].sum()

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(emotion_counts.index, emotion_counts.values, color='blue')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.title('Counts of Different Emotions')
plt.show()


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
    'Joking': -1
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

# Save the plot to a file
plt.show()

heatmap_data = data[['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Official report', 'Joking']]
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap="Blues", cbar=True, linewidths=0.5, linecolor='grey')
plt.title('Sentiment Heatmap')
plt.xlabel('Sentiment')
plt.ylabel('Movie Index')
plt.show()