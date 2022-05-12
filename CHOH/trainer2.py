import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import datetime
import os
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        evaluate_fn,
        optimizer,
        train_loader,
        validation_loader,

    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self._model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluate_fn = evaluate_fn
        #self.scheduler = CosineAnnealingLR(self.optimizer,)

    def _optimizer_to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def lr_scheduler(self,optimizer, decrease_rate):
        """Decay learning rate by a factor."""
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decrease_rate

        return optimizer
    def train(self, device, n_epochs,title,decrease_epoch,decrease_rate):
        self._model.to(device)
        self._optimizer_to(device)
        train_epoch_loss = []
        eval_epoch_loss = []
        bestloss=0.0

        localtime = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        dir_name = localtime+"_"+title
        os.makedirs(dir_name)
        lr_epoch = []
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs)
        for epoch in range(n_epochs):
            lr_scheduler.step()
            it=0
            self._model.train()
            train_loss = 0.0
            train_iter = self.train_loader
            print("epoch is "+str(epoch+1))
            train_times=0
            for train_batch,labels in train_iter:
                self.optimizer.zero_grad()
                train_batch=train_batch.to(device)
                result = self._model(train_batch)
                labels = labels.to(device)
                result = result.squeeze()
                labels = labels.squeeze()
                loss = self.loss_fn(labels, result)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
                self.optimizer.step()
                # if epoch % decrease_epoch == 0 and it==0 and epoch!=0:
                #     self.optimizer = self.lr_scheduler(self.optimizer, decrease_rate=decrease_rate)
                #print(loss.item())
                #print(self.optimizer)
                train_loss = train_loss + loss.item()
                train_times = train_times+1
                it = it + 1
            train_avg_loss = train_loss / train_times
            print("train_loss is "+str(train_avg_loss))
            train_epoch_loss.append(train_avg_loss)

            self._model.eval()
            val_loss = 0.0
            val_times = 0
            for val_batch,labels in self.validation_loader:
                val_batch = val_batch.to(device)
                val_result = self._model(val_batch)
                labels = labels.to(device)
                val_result = val_result.squeeze()
                labels = labels.squeeze()
                loss = self.evaluate_fn(labels, val_result)
                val_loss = val_loss+loss.item()
                val_times = val_times+1
            val_avg_loss=val_loss/val_times
            print("val_loss is "+str(val_avg_loss))
            print("lr is " + str(self.optimizer.state_dict()['param_groups'][0]['lr']))
            lr_epoch.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            eval_epoch_loss.append(val_avg_loss)
            if (epoch == 0):
                bestloss = val_avg_loss
            if(val_avg_loss<bestloss):
                bestloss = val_avg_loss
                torch.save(self._model.state_dict(),dir_name+"/"+title+ ".pt")
        dict = {'train_loss': train_epoch_loss, 'val_loss': eval_epoch_loss, 'lr': lr_epoch}
        df = pd.DataFrame(dict)
        df.to_csv(dir_name+'/data.csv')
        np.save(dir_name + "/" + title + "_train_loss", train_epoch_loss)
        np.save(dir_name + "/" + title + "_val_loss", eval_epoch_loss)
        x = np.arange(1, n_epochs + 1)
        print("bestloss is "+str(bestloss))
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, train_epoch_loss, label="train", color='red')
        plt.plot(x, eval_epoch_loss, label="val", color='blue')
        plt.legend()

        plt.savefig(dir_name+"/"+title, dpi=1000)
        plt.show()