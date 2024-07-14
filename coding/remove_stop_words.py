import nltk
from nltk.corpus import stopwords
import pandas as pd 


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df=pd.read_csv("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/data_csv.csv")


df['bodyContent'] = df['bodyContent'].astype(str)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def collect_stopwords(text):
    words = text.split()
    removed_words = [word for word in words if word.lower() in stop_words]
    return ' '.join(removed_words)

df['movie_filtered'] = df['bodyContent'].apply(remove_stopwords)
stop=df['removed_stopwords'] = df['bodyContent'].apply(collect_stopwords)


df.to_csv("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/cleaned_data.csv", index=False)
print(stop)
