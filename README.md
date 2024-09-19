# Sentiment analysis of movie scripts from Hollywood

This repository provides code and supplementary materials for the paper titled 'Longitudinal Abuse and Sentiment Analysis of Hollywood Oscar and Blockbuster Movie Dialogues using LLMs'.

## Seminar

## Publication

## Task Description

This project explores the trends in abusive language and sentiment in Hollywood movies from 1950 to 2024, with a focus on Oscar-nominated films and top 10 box-office hits. We utilize modern NLP models such as RoBERTa to conduct multi-label classification on movie subtitles, analyzing shifts in emotions and the use of abusive language across time and genres.

It examines changes in sentiment and abusive language in movie dialogues over 75 years, focusing on the influence of social and cultural shifts. Using large-scale language models (LLMs) fine-tuned on movie subtitles, we analyze various genres and decades to identify emotional trends in Hollywood.

### Datasets

Movies Subtitles:
Subtitles from over 1,000 films, including Oscar-nominated films and the top 10 box-office hits, were collected. These films were categorized into four genres: Action, Comedy, Drama, and Thriller.

SenWave Dataset:
This dataset includes sentiment-labeled tweets from the COVID-19 pandemic period. It is used to fine-tune our sentiment classification model for multi-label classification across emotions like optimism, anxiety, and anger. Additionally, the SenWave dataset from GitHub was utilised: 
[SenWave Dataset](https://github.com/gitdevqiang/SenWave?tab=readme-ov-file#senwave-a-fine-grained-sentiment-analysis-dataset-for-covid-19-tweets)

RAL-E Dataset:
A Reddit-based dataset used for detecting abusive language, focusing on offensive, hateful, or violent content. The dataset was crucial for fine-tuning our abuse detection models. The dataset we used comes from Tommaso Caselli's HateBERT paper:[RAL-E Dataset](https://osf.io/tbd58/?view_only=%20cb79b3228d4248ddb875eb1803525ad8)

### Models

N-Gram Analysis:
We conducted an N-Gram analysis (bigrams, trigrams) to visualize the most frequent word sequences in movie dialogues over time. This helped identify thematic trends and shifts in sentiment.

BERT-based Models:
We used pre-trained RoBERTa and HateBERT models for sentiment analysis and abuse detection. RoBERTa was fine-tuned using the SenWave dataset for sentiment analysis, while HateBERT was used to detect abusive language in movie dialogues.

### Results

1. Sentiment Analysis Over Time
   
We performed sentiment analysis on movie dialogues from 1950 to 2024, identifying significant changes in emotional expression.

Sentiment Polarity Trends (1950-2024)
The graph below shows the trend of sentiment polarity in movie dialogues over time, with sentiment polarity scores ranging from -1 to 1, where positive numbers represent positive emotions and negative numbers represent negative emotions.

Sentiment Weights by Decade
The sentiment weights chart highlights the relative contribution of different emotions over the decades. Emotions like optimism, anger, and humor fluctuate in prominence across different time periods.
![image](https://github.com/user-attachments/assets/fa7e93b1-ef34-4d89-96a0-b5118c5a6bc3)


2. Abusive Language Detection
   
Abusive Word Frequency by Decade
Abusive language frequency peaked in the 2000s and has since declined, reflecting changing societal norms.

Abusive Content Across Genres
Action films show a low level of abusive content, while thrillers in the 1950s had the highest abusive word count.
![image](https://github.com/user-attachments/assets/4c34ac95-fb90-4c18-8dec-e49dcab9cd34)


3. Emotional Sentiment Co-occurrence
   
The heatmap below shows frequent co-occurrences of humor with anger, especially in comedies, reflecting the use of satire and conflict.
![image](https://github.com/user-attachments/assets/958371b5-3b73-49b5-8da3-b4734c0e9816)



