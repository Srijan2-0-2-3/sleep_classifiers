import glob
import pandas as pd
from source.analysis.setup.subject_builder import SubjectBuilder
import os

from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService

subject_ids = SubjectBuilder.get_all_subject_ids()
print(len(subject_ids))

subjs = [("S2", "S3"), ("S5", 'S7'), ("S8", "S9"), ("S10", "S11"), ("S13", "S14"), ("S15", "S16"), ("S17", "")]
subjs = []
for i in range(0, 30, 2):
    subjs.append((subject_ids[i], subject_ids[i + 1]))

print(len(subjs))

for subject_id in subject_ids:
    # joined_files = os.path.join('/home/srijan/PycharmProjects/sleep_classifiers/outputs/features', subject_id + '*.out')
    # joined_list = glob.glob(joined_files)
    # print(joined_list)
    # for file in joined_list:
    #     fd = open(file,'r')
    #     lines = fd.readlines()
    #     df = pd.read
    #     # new_lines = []
    #     # for line in lines:
    #     #     line.strip('\n')
    #     #     print(line)
    #     #     new_lines.append(line)
    #     # lines = new_lines
    #     # print(lines)
    #     print(df)
    # subject = SubjectBuilder.build(subject_id)
    # print(subject.subject_id)
    # df_feature = pd.DataFrame([subject.feature_dictionary])
    # print(df_feature)
    # print(subject.labeled_sleep)
    feature_count = ActivityCountFeatureService.load(subject_id)
    feature_count = [feature for feature in feature_count]
    print(feature_count)
    feature_hr = HeartRateFeatureService.load(subject_id)
    feature_time = TimeBasedFeatureService.load_time(subject_id)
    feature_cosine = TimeBasedFeatureService.load_cosine(subject_id)
    labeled_sleep = PSGLabelService.load(subject_id)
    columns = ['count', 'hr', 'time', 'cosine', 'sleep']
    user_dataframe = pd.DataFrame(
        list(zip(feature_count, feature_hr, feature_time, feature_cosine, labeled_sleep)),
        columns=columns)
    user_dataframe.to_csv(f'{subject_id}.csv')
    # print(subject.feature_dictionary)
    # df = pd.concat(map(pd.read_csv, joined_list), ignore_index=False,axis=0)
    # df = pd.DataFrame()
    # df =df.merge(pd.read_csv,joined_list)
    # print(df)
