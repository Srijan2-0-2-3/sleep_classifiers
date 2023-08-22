import time

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
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

from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter


class FeatureDataset(Dataset):
    def __init__(self, subject_id):
        subject = SubjectBuilder.build(subject_id)
        print(subject.subject_id)
        # print(subject.feature_dictionary)
        # print(subject.labeled_sleep)
        feature_dictionary = torch.Tensor(subject.feature_dictionary)
        labeled_sleep = torch.Tensor(subject.labeled_sleep)
        print(type(labeled_sleep))
    #     x = df.iloc[1:num_rows, 0:3].values
    #     y = df.iloc[1:num_rows, 4].values
    #
    #     sc = StandardScaler()
    #     x_train = sc.fit_transform(x)
    #     y_train = y
    #
    #     self.data = torch.tensor(x_train)
    #     self.targets = torch.tensor(y_train)
    #
    # def __len__(self):
    #     return len(self.targets)
    #
    # def __getitem__(self, idx):
    #     return self.data[idx], self.targets[idx]


def avalanche_method(strat, i):
    device = torch.device('cpu')
    subject_ids = SubjectBuilder.get_all_subject_ids()
    data_splits = TrainTestSplitter.leave_one_out(subject_ids)
    train_set = data_splits[0].training_set
    # training_dataset =
    # print(training_dataset)
    test_set = data_splits[0].testing_set
    # testing_set =
    scenario = dataset_benchmark(train_datasets=[FeatureDataset(subject_id) for subject_id in train_set],
                                 test_datasets=[FeatureDataset(subject_id) for subject_id in test_set])

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
    model = SimpleMLP(num_classes=6)

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

    print(str(i) + ".")
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


if __name__ == '__main__':
    avalanche_method(strat='naive', i=0)