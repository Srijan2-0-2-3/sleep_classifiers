from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter

subject_ids = SubjectBuilder.get_all_subject_ids()

for subject_id in subject_ids:
    subject = SubjectBuilder.build(subject_id)
    print(len(subject.feature_dictionary[FeatureType.count]))
