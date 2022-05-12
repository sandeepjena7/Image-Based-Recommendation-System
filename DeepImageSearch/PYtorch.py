import torch
import numpy as np
from DeepImageSearch.utils.allutils import util
import torchvision.transforms as  T
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import Linear, LeakyReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import os
from time import time
import copy
import argparse
import tensorflow as tf




class _seed:
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class Net(nn.Module):
    def __init__(self, inputchannel=3, output_class=None):
        super(Net, self).__init__()

        self.cnn_layer = Sequential(
            Conv2d(inputchannel, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(4, stride=4),
            Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(4, stride=4))

        self.fc = nn.Sequential(
            nn.Flatten(),
            Linear(1960, output_class))

    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.fc(x)
        return x

class CustoModelPytorch:
    def __init__(self,config,params):
        params = util.read_yaml(params)
        config = util.read_yaml(config)

        self.device = 'cpu'
        artifact_dir = config['artifacts']['artifactdir']
        imagedir_name = config['artifacts']["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        self.train_dir = config['artifacts']["train_dir"]
        self.val_dir = config['artifacts']["val_dir"]
        metadata_dir = config['artifacts']["meta_data_dir"]
        model_name = config['artifacts']['pytorchmodel']
        historyfile = config['artifacts']['pytorchhistory']

        self.image_dir = os.path.join(artifact_dir,imagedir_name,preprocessed)
        self.modelfilename = os.path.join(artifact_dir,metadata_dir,model_name)
        self.historypath = os.path.join(artifact_dir,metadata_dir,historyfile)

        self.batchsize = params['batch_size']
        self.epoch = params['epoch']
        self.no_of_class = params['class']
        self.lr = params['lr']
        random_state = params['random_state']
        _seed.set_seed(random_state)

    @classmethod
    def _dataaugmentation(cls):
        dataaug = {
            "train": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the data base on imagenet datasets
            ]),
            "val": T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        return dataaug

    @classmethod
    def _dataloader(cls,imagedir,traindir,valdir,batchsize,shuffle=True,numworkers=1):

        dataset = {x: ImageFolder(os.path.join(imagedir,x),cls._dataaugmentation()[x] ) for x in [traindir,valdir]}
        dataloader = {x: DataLoader(dataset[x],batch_size=batchsize,shuffle=shuffle) for x in [traindir,valdir]}
        del dataset
        return dataloader

    def __call__(self):

        dataloader = self._dataloader(self.image_dir,self.train_dir,self.val_dir,self.batchsize)
        model = Net(3,self.no_of_class).to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.9, nesterov=True)
        criterion = CrossEntropyLoss()
        history = {'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}
        since = time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0       

        for epoch in range(self.epoch):
            print(f"Epoch {epoch + 1}/{self.epoch}")
            print("-" * 30)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_correct = 0

                run_time = time()
                for i,(inputs, labels) in  enumerate( dataloader[phase]):
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(inputs)
                        loss = criterion(preds, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    running_correct += preds.argmax(dim=1).eq(labels).sum().item()

                epoch_loss = running_loss / (len(dataloader[phase]) * self.batchsize)
                epoch_acc = running_correct / (len(dataloader[phase]) * self.batchsize)
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time() - run_time:.2f}")
                if phase == 'train':
                    history['loss'].append(epoch_loss)
                    history['accuracy'].append(epoch_acc)
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_accuracy'].append(epoch_acc)


                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # torch.cuda.empty_cache()  # if not used gpu please comment this

            print()
        time_elapsed = time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s Epoch: {self.epoch}")
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(),self.modelfilename)
        util.dump_pickle(self.historypath,history)


def maintorch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',"-c", type=str, default="config/config.yaml", help='ROOT/config/config.yaml')
    parser.add_argument('--params',"-p", type=str, default="params.yaml", help='ROOT/params.yaml')
    opt = parser.parse_args()
    model = CustoModelPytorch(**vars(opt))
    model()


if __name__ == "__main__":
    maintorch()



