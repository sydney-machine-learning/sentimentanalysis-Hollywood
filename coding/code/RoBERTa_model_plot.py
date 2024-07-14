import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('C:/Users/Admin/Desktop/data_csv.csv')

# Process the data: group by year and calculate the required statistics
data_grouped = data.groupby('year').agg({
    'movie': 'count',  # Count of movies
    'numberofwords': 'mean'  # Average number of words
}).reset_index()

# Create separate plots for each decade
decades = list(range(1950, 2030, 10))  # List of decades

for start_year in decades:
    end_year = start_year + 9
    decade_data = data_grouped[(data_grouped['year'] >= start_year) & (data_grouped['year'] <= end_year)]
    
    if not decade_data.empty:
        years = decade_data['year']
        movie_count = decade_data['movie']
        average_number_of_words = decade_data['numberofwords']

        x = np.arange(len(years))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax1 = plt.subplots()

        # Plot the count of movies
        rects1 = ax1.bar(x - width/2, movie_count, width, label='Count of Movies', color='green')

        # Instantiate a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot the average number of words
        rects2 = ax2.bar(x + width/2, average_number_of_words, width, label='Average Number of Words', color='black')

        # Add labels, title, and legend
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Count of Movies', color='green')
        ax2.set_ylabel('Average Number of Words', color='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years, rotation=45)

        # Combine legends
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
        
        plt.title(f'{start_year}-{end_year} Decade Data')
        plt.show()


