import time

from avalanche.benchmarks.generators import dataset_benchmark, nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, make_classification_dataset
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, Cumulative, LwF, EWC, JointTraining, GEM, Replay
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
import pickle
import torch.nn as nn
import torch
import numpy as np
import sys
import time
import tensorflow as tf
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import copy
from source.analysis.setup.subject_builder import SubjectBuilder


class WESADTsSet(Dataset):
    def __init__(self, transform=None):
        self.root_dir = ""
        self.transform = transform
        Xts = pickle.load(open(self.root_dir + "Xts.pkl", 'rb'), encoding='latin1')
        yts = pickle.load(open(self.root_dir + "yts.pkl", 'rb'), encoding='latin1')
        if self.transform:
            Xts = self.transform(Xts)
        self.data = Xts
        self.targets = yts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        return data, target


class WESADTrSet(Dataset):
    def __init__(self, pair, transform=None):
        self.root_dir = ""
        self.transform = transform
        self.pair = pair
        self.subject_ids = SubjectBuilder.get_all_subject_ids()
        subjs = [('3509524', '5132496'), ('1066528', '5498603'),
                 ('2638030', '2598705'), ('5383425', '1455390'),
                 ('4018081', '9961348'), ('1449548', '8258170'),
                 ('781756', '9106476'), ('8686948', '8530312'),
                 ('3997827', '4314139'), ('1818471', '4426783'),
                 ('8173033', '7749105'), ('5797046', '759667'),
                 ('8000685', '6220552'), ('844359', '9618981'),
                 ('1360686', '46343'), ('8692923', '')]

        print(len(subjs))
        couple = subjs[self.pair]

        X, y = None, None
        for S in couple:
            if S != "":
                Xs = pickle.load(open(self.root_dir + "X" + S + ".pkl", 'rb'), encoding='latin1')
                ys = pickle.load(open(self.root_dir + "y" + S + ".pkl", 'rb'), encoding='latin1')
                if (X is None):
                    X = copy.deepcopy(Xs)
                    y = copy.deepcopy(ys)
                else:
                    X = np.concatenate([X, Xs], axis=0)
                    y = np.concatenate([y, ys], axis=0)
                del Xs
                del ys
        if self.transform:
            X = self.transform(X)

        self.data = torch.tensor(X)
        self.targets = torch.tensor(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        return data, target


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=1):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden_dim, layers, batch_first=True)
        self.clf = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # batch_size first
        x, _ = self.rnn(x)  # _ to ignore state
        x = x[:, -1]  # last timestep for classfication
        return self.clf(x)


import warnings

warnings.filterwarnings("ignore")


def train_wesad(strat, i=""):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    scenario = dataset_benchmark(
        [WESADTrSet(pair=0), WESADTrSet(pair=1), WESADTrSet(pair=2), WESADTrSet(pair=3), WESADTrSet(pair=4),
         WESADTrSet(pair=5), WESADTrSet(pair=6), WESADTrSet(pair=7), WESADTrSet(pair=8), WESADTrSet(pair=9),
         WESADTrSet(pair=10),
         WESADTrSet(pair=11), WESADTrSet(pair=12), WESADTrSet(pair=13), WESADTrSet(pair=14), WESADTrSet(pair=15)],
        # AvalancheDataset(WESADTrSet(pair=7), task_labels=8)],
        [(WESADTsSet())]
    )
    # scenario = make_classification_dataset([AvalancheDataset(WESADTrSet(pair=0)),
    # 	AvalancheDataset(WESADTrSet(pair=1)),
    # 	AvalancheDataset(WESADTrSet(pair=2)),
    # 	AvalancheDataset(WESADTrSet(pair=3)),
    # 	AvalancheDataset(WESADTrSet(pair=4)),
    # 	AvalancheDataset(WESADTrSet(pair=5)),
    # 	AvalancheDataset(WESADTrSet(pair=6))
    # 	# AvalancheDataset(WESADTrSet(pair=7), task_labels=8)
    # 										])
    # [AvalancheDataset(WESADTsSet())])
    # ds = WESADTrSet(pair=0)
    # data = ds.data
    # targets = ds.targets
    # dataset = tf.data.Dataset.from_tensor_slices((ds.data, ds.targets))
    # scenario = dataset_benchmark(dataset,WESADTsSet())
    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open('wesadlog.txt', 'a'))
    int_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        gpu_usage_metrics(0, experience=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[text_logger]
    )

    es = EarlyStoppingPlugin(patience=25, val_stream_name="train_stream")

    results = []
    model = Classifier(in_dim=4, hidden_dim=4, out_dim=1, layers=2)

    if (strat == "naive"):
        print("Naive continual learning")
        strategy = Naive(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                         train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "offline"):
        print("Offline learning")
        strategy = JointTraining(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                                 train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "replay"):
        print("Replay training")
        strategy = Replay(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                          train_epochs=10, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                          mem_size=70, train_mb_size=70)  # 25% of WESAD
    elif (strat == "cumulative"):
        print("Cumulative continual learning")
        strategy = Cumulative(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                              train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "lwf"):
        print("LwF continual learning")
        strategy = LwF(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, alpha=0.5,
                       temperature=1)
    elif (strat == "ewc"):
        print("EWC continual learning")
        torch.backends.cudnn.enabled = False
        strategy = EWC(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=1, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                       ewc_lambda=0.99)
    elif (strat == "episodic"):
        print("Episodic continual learning")
        strategy = GEM(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                       patterns_per_exp=70)

    thisresults = []

    print(i + ".")
    start = time.time()
    if strat == "offline":
        res = strategy.train(scenario.train_stream)
        r = strategy.eval(scenario.test_stream)
        thisresults.append({"loss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                            "acc": (float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"]) * 100),
                            "forg": r["StreamForgetting/eval_phase/test_stream"],
                            "all": r})
        results.append({"strategy": strat,
                        "finalloss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                        "finalacc": r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
                        "results": thisresults})
    else:
        for experience in scenario.train_stream:
            print(start)
            res = strategy.train(experience)
            r = strategy.eval(scenario.test_stream)

            print(f"loss:{r['Loss_Exp/eval_phase/test_stream/Task000/Exp000']}")
            thisresults.append({"loss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                                "acc": (float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"]) * 100),
                                "forg": r["StreamForgetting/eval_phase/test_stream"],
                                "all": r})
        results.append({"strategy": strat,
                        "finalloss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                        "finalacc": r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
                        "results": thisresults})
    elapsed = time.time() - start
    results.append({"time": elapsed})
    with open("wesad_" + strat + "_results" + i + ".pkl", "wb") as outfile:
        pickle.dump(results, outfile)
    print("\t" + str(elapsed) + " seconds")


strat = 'ewc'
for i in range(5):
    # train_wesad(sys.argv[1].strip(), str(i))
    train_wesad(strat, str(i))
if __name__ == '__main__':
    strat = 'ewc'
    for i in range(5):
        # train_wesad(sys.argv[1].strip(), str(i))
        train_wesad(strat, str(i))
