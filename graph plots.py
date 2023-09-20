import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

methods = {'replay': {},
           'cumulative': {},
           'lwf': {},
           'ewc': {}}


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


loss = []
cpu_usage = []
acc = []
forg = []
disk_usage = []
time = []
for i in range(0, 30):
    datas = pickle.load(open('replay_' + str(i) + '_sleep_classifier.pkl', 'rb'), encoding='latin1')
    # print(datas)

    loss.append(datas[i]['loss'])
    acc.append(datas[i]['acc'])
    forg.append(datas[i]['forg'])
    cpu_usage.append(datas[i]['cpu_usage'])
    disk_usage.append(datas[i]['disk_usage'])

exp = [i for i in range(0, 30)]

methods['replay']['loss'] = loss
methods['replay']['acc'] = acc
methods['replay']['forg'] = forg
methods['replay']['cpu_usage'] = cpu_usage
methods['replay']['disk_usage'] = disk_usage
print(methods)
plt.plot(exp, disk_usage)
plt.savefig('disk_usage_sl.png')
