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

# loss = []
# cpu_usage = []
# acc = []
# forg = []
# disk_usage = []
# time = []
#
#
# datas = pickle.load(open('replay_' + str(29) + '_sleep_classifier.pkl', 'rb'), encoding='latin1')
# traverse_dict_and_lists(datas)
# for i in range(0, 30):
#     loss.append(datas[i]['loss'])
#     acc.append(datas[i]['acc'])
#     forg.append(datas[i]['forg'])
#     cpu_usage.append(datas[i]['cpu_usage'])
#     disk_usage.append(datas[i]['disk_usage'])
#
#
#
# methods['replay']['loss'] = loss
# methods['replay']['acc'] = acc
# methods['replay']['forg'] = forg
# methods['replay']['cpu_usage'] = cpu_usage
# methods['replay']['disk_usage'] = disk_usage
# print(len(methods['replay']['loss']))
# # plt.plot(exp, disk_usage)
# # plt.savefig('disk_usage_sl.png')
#
# loss = []
# cpu_usage = []
# acc = []
# forg = []
# disk_usage = []
# time = []
#
# datas = pickle.load(open('cumulative_' + str(29) + '_sleep_classifier.pkl', 'rb'), encoding='latin1')
# print(datas)
#
# for i in range(0, 30):
#     loss.append(datas[i]['loss'])
#     acc.append(datas[i]['acc'])
#     forg.append(datas[i]['forg'])
#     cpu_usage.append(datas[i]['cpu_usage'])
#     disk_usage.append(datas[i]['disk_usage'])
#
#
#
# methods['cumulative']['loss'] = loss
# methods['cumulative']['acc'] = acc
# methods['cumulative']['forg'] = forg
# methods['cumulative']['cpu_usage'] = cpu_usage
# methods['cumulative']['disk_usage'] = disk_usage
# print(len(methods['cumulative']['loss']))
# loss = []
# cpu_usage = []
# acc = []
# forg = []
# disk_usage = []
# time = []
# datas = pickle.load(open('lwf_' + str(29) + '_sleep_classifier.pkl', 'rb'), encoding='latin1')
# print(datas)
# for i in range(0, 30):
#     loss.append(datas[i]['loss'])
#     acc.append(datas[i]['acc'])
#     forg.append(datas[i]['forg'])
#     cpu_usage.append(datas[i]['cpu_usage'])
#     disk_usage.append(datas[i]['disk_usage'])
#
# methods['lwf']['loss'] = loss
# methods['lwf']['acc'] = acc
# methods['lwf']['forg'] = forg
# methods['lwf']['cpu_usage'] = cpu_usage
# methods['lwf']['disk_usage'] = disk_usage
# print(len(methods['lwf']['loss']))
# m = methods['lwf']['loss'][0:30]

#
# figure, axis = plt.subplots(2, 3)
#
# axis[0, 0].plot(exp, methods['cumulative']['loss'],label = 'cumulative')
# axis[0, 0].plot(exp, methods['replay']['loss'],label='replay')
# axis[0, 0].plot(exp, methods['lwf']['loss'],label='lwf')
# axis[0, 0].set_title("Loss")
#
# axis[0, 1].plot(exp, methods['cumulative']['acc'],label = 'cumulative')
# axis[0, 1].plot(exp, methods['replay']['acc'],label='replay')
# axis[0, 1].plot(exp, methods['lwf']['acc'],label='lwf')
# axis[0, 1].set_title("Accuracy")
#
# axis[1, 0].plot(exp, methods['cumulative']['forg'],label = 'cumulative')
# axis[1, 0].plot(exp, methods['replay']['forg'],label='replay')
# axis[1, 0].plot(exp, methods['lwf']['forg'],label='lwf')
# axis[1, 0].set_title("Forgetting")
#
# axis[1, 1].plot(exp, methods['cumulative']['cpu_usage'],label = 'cumulative')
# axis[1, 1].plot(exp, methods['replay']['cpu_usage'],label='replay')
# axis[1, 1].plot(exp, methods['lwf']['cpu_usage'],label='lwf')
# axis[1, 1].set_title("Cpu usage")
#
# axis[1, 2].plot(exp, methods['cumulative']['disk_usage'],label = 'cumulative')
# axis[1, 2].plot(exp, methods['replay']['disk_usage'],label='replay')
# axis[1, 2].plot(exp, methods['lwf']['disk_usage'],label='lwf')
# axis[1, 2].set_title("disk usage")
#
# # Combine all the operations and display
# plt.show()
# plt.savefig('sleep_classifier.png')
