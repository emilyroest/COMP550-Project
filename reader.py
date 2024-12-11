import pandas as pd
from pathlib import Path

p = Path(__file__).parent / 'data' / 'train-balanced-sarcasm.csv'
columns = ['label', 'comment', 'parent_comment']

# call to create csv output
def process_file_csv(csv_file):
    output_file = 'test_output.csv'
    with open(p, mode='r', encoding='utf-8') as file:
        df = pd.read_csv(csv_file, usecols=columns)
        df.to_csv(output_file, index=False)

# call for df version
def process_file_df(csv_file):
    with open(p, mode='r', encoding='utf-8') as file:
        df = pd.read_csv(csv_file, usecols=columns)
    return df
