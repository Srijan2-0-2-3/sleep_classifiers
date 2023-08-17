import pandas as pd
import numpy as np
from source.analysis.setup.subject_builder import SubjectBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor


def create_csv():
    SubjectBuilder.get_subject_dictionary()


def update_csv():
    subject_ids = SubjectBuilder.get_all_subject_ids()
    for subject_id in subject_ids:
        print(subject_id)
        df = pd.read_csv(f'{subject_id}.csv', index_col=0)
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        columns = ['count', 'hr', 'time', 'cosine', 'sleep_label']
        for column in columns:
            df[column] = df[column].apply(lambda x: str(x.split()[0].replace('[', '')))
            df[column] = df[column].apply(lambda x: str(x.split()[0].replace(']', '')))
            df[column] = df[column].astype(float)
        print(df.head())
        df.to_csv(f'{subject_id}.csv')
        # df = df.apply(pd.to_numeric)
        print(df.dtypes)


def load_csv():
    subject_ids = SubjectBuilder.get_all_subject_ids()
    for subject_id in subject_ids:
        df = pd.read_csv(f'{subject_id}.csv', index_col=0)
        print(df.columns)
        print(df.shape)
        array = df.to_numpy()
        X = np.concatenate([array[0], array[1], array[2], array[3]])
        print(X)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        print(f'{subject_id} standardised')
        Y = df['sleep_label']
        Y = np.around(Y)
        Y = abs(Y.astype(np.int_))
        print(f'{subject_id} labelled')


if __name__ == '__main__':
    load_csv()
