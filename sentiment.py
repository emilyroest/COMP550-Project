import reader
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# returns sentiment df - same format as csv
# new column orders are neg, neu, pos, compound
def sentiment_df(df):
    sia = SentimentIntensityAnalyzer()
    df = df.dropna(subset=['comment']).copy() 
    scores = df['comment'].apply(sia.polarity_scores)

    # normalize dic to df from polarity
    scores_df = pd.DataFrame(scores.tolist(), index=df.index)
    df = pd.concat([df, scores_df], axis=1)
    return df

# outputs a csv file with the sentiments attached
def sentiment_from_csv(input_csv):
    output_csv = 'test_output_sentiment.csv'
    df = reader.process_file_df(input_csv)
    df = sentiment_df(df)
    df.to_csv(output_csv, index=False)

def main():
    # example usage
    df = reader.process_file_df('test_output.csv')
    df_scores = sentiment_df(df)
    print(df_scores.head())
    sentiment_from_csv('test_output.csv')

if __name__ == "__main__":
    main()
