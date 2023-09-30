import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

methods = {'replay': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'cumulative': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'lwf': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'ewc': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'episodic': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []}}


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


for key in methods.keys():
    data = pickle.load(open(f'{key}_29_sleep_classifier.pkl', 'rb'), encoding='latin1')
    for i in range(30):
        methods[key]['loss'].append(data[i]['loss'])
        methods[key]['acc'].append(data[i]['acc'])
        methods[key]['cpu_usage'].append(data[i]['cpu_usage'])
        methods[key]['disk_usage'].append(data[i]['disk_usage'])
        methods[key]['forg'].append(data[i]['forg'])

traverse_dict_and_lists(methods['replay']['forg'])
exp = [i for i in range(0, 30)]
plt.plot(exp, methods['cumulative']['loss'], label='cumulative')
plt.plot(exp, methods['replay']['loss'], label='replay')
plt.plot(exp, methods['lwf']['loss'], label='lwf')
plt.plot(exp, methods['ewc']['loss'], label='ewc')
plt.plot(exp, methods['episodic']['loss'], label='episodic')
plt.legend()
plt.grid()
plt.show()
plt.savefig('loss_sl.png')

