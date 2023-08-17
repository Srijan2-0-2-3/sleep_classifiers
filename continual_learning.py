import pandas as pd

from source.analysis.setup.subject_builder import SubjectBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor

def create_csv():
    SubjectBuilder.get_subject_dictionary()

def load_csv():
    subject_ids = SubjectBuilder.get_all_subject_ids()
    for subject_id in subject_ids:
        print(subject_id)
        df = pd.read_csv(f'{subject_id}.csv',index_col=0)
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        # df['count'] = df['count'].astype(float)
        df = df.apply(pd.to_numeric)
        print(df.dtypes)

if __name__=='__main__':
    create_csv()