import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('C:/Users/Admin/Desktop/roberta_df.csv')


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