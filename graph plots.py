import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import matplotlib.ticker as mticker

# datas = pickle.load(open('sleep_classifiernaive_results.pkl', 'rb'), encoding='latin1')

methods = {'replay': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'cumulative': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'lwf': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'ewc': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'episodic': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []},
           'naive': {'loss': [], 'acc': [], 'forg': [], 'cpu_usage': [], 'disk_usage': []}}


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
    print(key)
    for i in range(30):
        data = pickle.load(open(f'results/{key}_{i}_sleep_classifier.pkl', 'rb'), encoding='latin1')
        # print(data)
        if i < 10:
            methods[key]['disk_usage'].append(data[0][f'DiskUsage_Exp/eval_phase/train_stream/Task000/Exp00{i}'])
            methods[key]['loss'].append(data[0][f'Loss_Exp/eval_phase/train_stream/Task000/Exp00{i}'])
            methods[key]['acc'].append(data[0][f'Top1_Acc_Exp/eval_phase/train_stream/Task000/Exp00{i}'] * 100)
            methods[key]['forg'].append(data[0][f'StreamForgetting/eval_phase/test_stream'] * 100)
            methods[key]['cpu_usage'].append(data[0][f'CPUUsage_Exp/eval_phase/train_stream/Task000/Exp00{i}'])

        else:
            methods[key]['disk_usage'].append(data[0][f'DiskUsage_Exp/eval_phase/train_stream/Task000/Exp0{i}'])
            methods[key]['loss'].append(data[0][f'Loss_Exp/eval_phase/train_stream/Task000/Exp0{i}'])
            methods[key]['acc'].append(data[0][f'Top1_Acc_Exp/eval_phase/train_stream/Task000/Exp0{i}'] * 100)
            methods[key]['forg'].append(data[0][f'StreamForgetting/eval_phase/test_stream'] * 100)
            methods[key]['cpu_usage'].append(data[0][f'CPUUsage_Exp/eval_phase/train_stream/Task000/Exp0{i}'])

    # traverse_dict_and_lists(data)

    print(len(methods[key]['disk_usage']))

strategies = ['disk_usage', 'loss', 'acc', 'forg', 'cpu_usage']
exp = [i for i in range(30)]
tick_spacing = 1

method_styles = {
    'naive': {'label': 'naive', 'linestyle': '-', 'marker': 'v', 'color': 'cyan'},
    'cumulative': {'label': 'cumulative', 'linestyle': '-', 'marker': 'o', 'color': 'blue'},
    'replay': {'label': 'replay', 'linestyle': '-', 'marker': 'x', 'color': 'red'},
    'lwf': {'label': 'lwf', 'linestyle': '-', 'marker': 's', 'color': 'green'},
    'ewc': {'label': 'ewc', 'linestyle': '-', 'marker': 'D', 'color': 'purple'},
    'episodic': {'label': 'episodic', 'linestyle': '-', 'marker': '^', 'color': 'orange'}
}

y_labels = {
    'disk_usage': 'Disk Usage',
    'loss': 'Loss',
    'acc': 'Accuracy (%)',
    'forg': 'Forgetting (%)',
    'cpu_usage': 'CPU Usage',
}
for strategy in strategies:
    plt.clf()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    for key, style in method_styles.items():
        plt.plot(exp, methods[key][strategy], label=style['label'], linestyle=style['linestyle'],
                 marker=style['marker'], color=style['color'])
    plt.xlabel('Experiences')
    plt.ylabel(y_labels[strategy])
    plt.xticks(rotation = 90)
    plt.legend()
    plt.grid()

    plt.savefig(f'graphs/{strategy}_sleep_classifier.png')
