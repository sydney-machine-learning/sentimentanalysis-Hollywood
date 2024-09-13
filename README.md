## Task Description

This project explores the trends in abusive language and sentiment in Hollywood movies from 1950 to 2024, with a focus on Oscar-nominated films and top 10 box-office hits. We utilize modern NLP models such as RoBERTa to conduct multi-label classification on movie subtitles, analyzing shifts in emotions and the use of abusive language across time and genres.

It examines changes in sentiment and abusive language in movie dialogues over 75 years, focusing on the influence of social and cultural shifts. Using large-scale language models (LLMs) fine-tuned on movie subtitles, we analyze various genres and decades to identify emotional trends in Hollywood.

### Datasets

Movies Subtitles
Subtitles from over 1,000 films, including Oscar-nominated films and the top 10 box-office hits, were collected. These films were categorized into four genres: Action, Comedy, Drama, and Thriller.

SenWave Dataset
This dataset includes sentiment-labeled tweets from the COVID-19 pandemic period. It is used to fine-tune our sentiment classification model for multi-label classification across emotions like optimism, anxiety, and anger.

RAL-E Dataset
A Reddit-based dataset used for detecting abusive language, focusing on offensive, hateful, or violent content. The dataset was crucial for fine-tuning our abuse detection models.

### Models

N-Gram Analysis
We conducted an N-Gram analysis (bigrams, trigrams) to visualize the most frequent word sequences in movie dialogues over time. This helped identify thematic trends and shifts in sentiment.

BERT-based Models
We used pre-trained RoBERTa and HateBERT models for sentiment analysis and abuse detection. RoBERTa was fine-tuned using the SenWave dataset for sentiment analysis, while HateBERT was used to detect abusive language in movie dialogues.

### Results

Sentiment Analysis
Sentiment analysis revealed that humorous and positive emotions like joking are highly frequent in movie dialogues, especially in comedies.
Over time, the overall emotional polarity of movies became more complex, with the highest positivity found in earlier decades (1950s-1980s).

Abusive Language Detection
The overall frequency of abusive language has declined since the 1950s, particularly in the Action and Thriller genres.
The analysis of abusive words showed notable changes in societal acceptance of violent and abusive content in films.
