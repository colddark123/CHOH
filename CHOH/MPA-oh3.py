import numpy
import torch as th
import torch.nn as nn
import argparse
import numpy as np
from trainer2 import Trainer
import datetime

th.manual_seed(10)
parser = argparse.ArgumentParser()


parser.add_argument('--out_dim', type=int, default=1, help='output layer')
parser.add_argument('--lr', type=float, default='1e-4')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default='100')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--plttitle', type=str, default='OH3-MPA')
args = parser.parse_args()

all_inputs = np.float32(numpy.load("data/oh3_x.npy"))
all_targets = numpy.load("data/oh3_y.npy")
all_targets = np.float32(numpy.squeeze(all_targets))

if th.cuda.is_available():
    device = args.device
else:
    device = 'cpu'

from data.ohData import MyGraphDataset
Dataset=MyGraphDataset(all_inputs,all_targets,device)

#split 13449 1681 1682
train_size = int(len(Dataset)*0.8)
validate_size = int(len(Dataset)*0.1)
test_size = len(Dataset) - validate_size - train_size
print(train_size,validate_size,test_size)
train_dataset, validate_dataset, test_dataset = th.utils.data.random_split(Dataset, [train_size, validate_size, test_size])

from dgl.dataloading import GraphDataLoader
# create dataloaders
batch_size=args.batch_size

train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = GraphDataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=True)


all=[]
starttime = datetime.datetime.now()
for graph,labels in train_dataset:
    #labels=labels.numpy()
    node_num=len(graph.nodes())
    all.append(labels/node_num)
endtime = datetime.datetime.now()
print("get mean and std time:"+str(endtime - starttime))
all=np.array(all)
print(np.mean(all))
print(np.var(all))
mean=np.mean(all)
std=np.var(all)


n_epochs=args.epoch
device=args.device
from model.MPA import Representation



embedding_dim=64
hidden_dim=64

mlp_dim=32
output_dim=1
n_interactions=3
num_heads=20
start=0.0
stop=5.0
n_gaussians=300
weight_decay=0.99
decrease_steps=100000
decrease_epoch=5
decrease_rate=0.8
net= Representation(embedding_dim,hidden_dim,mlp_dim,
                    output_dim,device,num_heads
                    ,start,stop,n_gaussians,mean,std,n_interactions)
net.to(device)
from torch import optim
loss_func = nn.L1Loss()
evaluate_fn = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


trainer = Trainer(
    model=net,
    loss_fn=loss_func,
    evaluate_fn=evaluate_fn,
    optimizer=optimizer,
    train_loader=train_dataloader,
    validation_loader=validate_dataloader,
)
trainer.train(device=device,
              n_epochs=n_epochs,
              title=args.plttitle,
              decrease_epoch=decrease_epoch,
              decrease_rate=decrease_rate)

endtime = datetime.datetime.now()
print("training use:"+str(endtime - starttime))

