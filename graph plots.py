import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
datas = pickle.load(open('naive_1_sleep_classifier.pkl', 'rb'), encoding='latin1')

print(datas)

def traverse_dict_and_lists(data, indent=""):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}{key}:")
            traverse_dict_and_lists(value, indent + "  ")
    elif isinstance(data, list):
        for item in data:
            print(f"{indent}- (List Item):")
            traverse_dict_and_lists(item, indent + "  ")
    else:
        print(f"{indent}{data}")

traverse_dict_and_lists(datas)