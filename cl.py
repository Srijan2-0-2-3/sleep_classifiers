import time

import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from avalanche.benchmarks.generators import dataset_benchmark, nc_benchmark
from avalanche.training.supervised import Naive, Cumulative, LwF, EWC, JointTraining, GEM, Replay
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
import pickle
import torch.nn as nn
import torch

import warnings

warnings.filterwarnings("ignore")
from avalanche.benchmarks.utils import make_classification_dataset

from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter


class FeatureDataset(Dataset):
    def __init__(self, subject_id):
        subject = SubjectBuilder.build(subject_id)
        x = []
        for feature in subject.feature_dictionary.keys():
            x.append(subject.feature_dictionary[feature])
        self.data = torch.tensor(np.transpose((np.array(x))), dtype=torch.float32)
        print(self.data.shape)
        self.targets = torch.tensor(subject.labeled_sleep, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # print(self.targets[idx])
        return self.data[idx], self.targets[idx]


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier, self).__init__()
        # print('init')
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)

        x = self.sigmoid(x)

        x = self.output(x)

        x = self.softmax(x)

        return x


def cl_method():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subject_ids = SubjectBuilder.get_all_subject_ids()
    data_splits = TrainTestSplitter.leave_one_out(subject_ids)
    train_set = data_splits[0].training_set
    test_set = data_splits[0].testing_set
    train_set = [data_splits[i].training_set for i in range(len(subject_ids))]
    test_set = [data_splits[i].testing_set for i in range(len(subject_ids))]


    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open('sleep_classifier_log.txt', 'a'))
    int_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        gpu_usage_metrics(0, experience=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[int_logger]
    )

    es = EarlyStoppingPlugin(patience=25, val_stream_name="train_stream")

    start2 = time.time()
    strats = ['naive', 'replay', 'cumulative', 'lwf', 'ewc', 'episodic']
    # for strat in strats:
    strat = 'naive'
    input_dim = 4
    hidden_dim = 8
    output_dim = 6

    model = Classifier(input_dim, hidden_dim, output_dim).to(device)
    if (strat == "naive"):
        print("Naive continual learning")
        strategy = Naive(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                         train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                         train_mb_size=64, eval_mb_size=64)
    elif (strat == "offline"):
        print("Offline learning")
        strategy = JointTraining(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                                 train_epochs=30, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "replay"):
        print("Replay training")
        strategy = Replay(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                          train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                          mem_size=100,
                          train_mb_size=50, eval_mb_size=50)
    elif (strat == "cumulative"):
        print("Cumulative continual learning")
        strategy = Cumulative(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                              train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device,
                              train_mb_size=50, eval_mb_size=50)
    elif (strat == "lwf"):
        print("LwF continual learning")
        strategy = LwF(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=100,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, alpha=0.5, temperature=1,
                       train_mb_size=50, eval_mb_size=50)
    elif (strat == "ewc"):
        print("EWC continual learning")
        torch.backends.cudnn.enabled = False
        strategy = EWC(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=100,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, ewc_lambda=0.99,
                       train_mb_size=50, eval_mb_size=50)
    elif (strat == "episodic"):
        print("Episodic continual learning")
        strategy = GEM(model, Adam(model.parameters(), lr=0.001, betas=(0.99, 0.99)), CrossEntropyLoss(),
                       train_epochs=100,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, patterns_per_exp=70,
                       train_mb_size=50, eval_mb_size=50)

    results = []
    start1 = time.time()
    i = 0
    for training_set,testing_set in zip(train_set,test_set):
        scenario = dataset_benchmark([make_classification_dataset(FeatureDataset(subject)) for subject in train_set],
                                     [make_classification_dataset(FeatureDataset(subject)) for subject in test_set])
        for experience in scenario.train_stream:
            start = time.time()
            thisresults = []
            print(experience)
            print(start)
            res = strategy.train(experience)
            r = strategy.eval(scenario.test_stream)

            thisresults.append(r)
            print('time_taken', time.time() - start)

            with open(f'results/{strat}_{i}_sleep_classifier.pkl', 'ab') as f:
                pickle.dump(thisresults, f)
            i += 1

    print(time.time() - start2)


cl_method()
