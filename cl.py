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

from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter


class FeatureDataset(Dataset):
    def __init__(self, subject_id):
        subject = SubjectBuilder.build(subject_id)
        x = []
        for feature in subject.feature_dictionary.keys():
            x.append(subject.feature_dictionary[feature])

        self.data = torch.tensor(np.transpose((np.array(x))), dtype=torch.float32)
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


input_dim = 4
hidden_dim = 8
output_dim = 6

model = Classifier(input_dim, hidden_dim, output_dim)

# dataset = FeatureDataset('46343')
#
# val_size = int(0.2 * len(dataset))
# train_size = len(dataset) - val_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# batch_size = 1
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 1
# for epoch in range(num_epochs):
#     model.train()
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         val_correct = 0
#         total_samples = 0
#         for batch_X, batch_y in val_loader:
#             val_outputs = model(batch_X)
#             val_loss += criterion(val_outputs, batch_y).item()
#
#             _, predicted = torch.max(val_outputs, 1)
#             print(predicted)
#             val_correct += (predicted == batch_y).sum().item()
#             total_samples += len(batch_y)
#
#         val_accuracy = val_correct / total_samples
#         print(
#             f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

scenario = dataset_benchmark([FeatureDataset('46343')],
                             [FeatureDataset('46343')])

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
device = torch.device('cpu')
strats = ['naive','offline','replay','cumulative','lwf','ewc','episodic']
for strat in strats:
    if (strat == "naive"):
        print("Naive continual learning")
        strategy = Naive(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                         train_epochs=1, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "offline"):
        print("Offline learning")
        strategy = JointTraining(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                                 train_epochs=1, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "replay"):
        print("Replay training")
        strategy = Replay(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                          train_epochs=1, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, mem_size=70,
                          train_mb_size=70)  # 25% of WESAD
    elif (strat == "cumulative"):
        print("Cumulative continual learning")
        strategy = Cumulative(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                              train_epochs=1, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
    elif (strat == "lwf"):
        print("LwF continual learning")
        strategy = LwF(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=1,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, alpha=0.5, temperature=1)
    elif (strat == "ewc"):
        print("EWC continual learning")
        torch.backends.cudnn.enabled = False
        strategy = EWC(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=1,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, ewc_lambda=0.99)
    elif (strat == "episodic"):
        print("Episodic continual learning")
        strategy = GEM(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=1,
                       eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, patterns_per_exp=70)

    thisresults = []

    start = time.time()
    for experience in scenario.train_stream:
        print(start)
        res = strategy.train(experience)
        r = strategy.eval(scenario.test_stream)

        print(f"loss:{r['Loss_Exp/eval_phase/test_stream/Task000/Exp000']}")
        thisresults.append({"loss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                            "acc": (float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"]) * 100),
                            "forg": r["StreamForgetting/eval_phase/test_stream"],
                            "all": r})
    results.append({"strategy": 'replay',
                    "finalloss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                    "finalacc": r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
                    "results": thisresults})
    elapsed = time.time() - start
    results.append({"time": elapsed})
    with open("sleep_classifier" + strat + "_results" + ".pkl", "wb") as outfile:
        pickle.dump(results, outfile)


results = pickle.load(open("wesad_replay_results.pkl", 'rb'), encoding='latin1')
for result in results:
    print(result)
