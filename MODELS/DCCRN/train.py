import torch
import os
import time
import pandas as pd
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import *
from model import *

class WavDataset(Dataset):
    def __init__(self, paths):
        self.pairs_paths = [os.path.join(paths, filename) for filename in os.listdir(paths)]

    def __getitem__(self, item):
        pair = np.load(self.pairs_paths[item], allow_pickle=True)

        return torch.from_numpy(pair[0]).type(torch.FloatTensor), torch.from_numpy(pair[1]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.pairs_paths)

def test_epoch(model, test_iter, device, criterion, batch_size, test_all=False):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        i = 0
        for _, (x, y) in enumerate(test_iter):
            x = x.view(x.size(0) * x.size(1), x.size(2)).float().squeeze(0).to(device)
            y = y.view(y.size(0) * y.size(1), y.size(2)).float().squeeze(0).to(device)
            
            y_p = model(x, train=False)
            loss = criterion(source=y.unsqueeze(1), estimate_source=y_p)
            loss_sum += loss.item()
            i += 1
    return loss_sum / i

def train(model, optimizer, criterion, train_iter, test_iter, max_epoch, device, batch_size, log_path):
    train_losses = []
    test_losses = []
    for epoch in range(max_epoch):
        loss_sum = 0
        i = 0
        for step, (x, y) in enumerate(train_iter):
            x = x.view(x.size(0) * x.size(1), x.size(2)).float()
            y = y.view(y.size(0) * y.size(1), y.size(2)).float()
            shuffle = torch.randperm(x.size(0))
            x = x[shuffle]
            y = y[shuffle]
            for index in range(0, x.size(0) - batch_size + 1, batch_size):
                model.train()
                x_item = x[index:index + batch_size, :].squeeze(0).to(device)
                y_item = y[index:index + batch_size, :].squeeze(0).to(device)
                optimizer.zero_grad()
                y_p = model(x_item)
                loss = criterion(source=y_item.unsqueeze(1), estimate_source=y_p)
                if step == 0 and index == 0 and epoch == 0:
                    loss.backward()
                    loss_sum += loss.item()
                    i += 1
                    test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
                    print(
                        "first test step:%d,ind:%d,train loss:%.5f,test loss:%.5f" % (
                            step, index, loss_sum / i, test_loss)
                    )
                    train_losses.append(loss_sum / i)
                    test_losses.append(test_loss)
                else:
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    i += 1

        with torch.no_grad():
            test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
        print(
            "epoch:%d,step:%d,train loss:%.5f,test loss:%.5f,time:%s" % (
                epoch, step, loss_sum / i, test_loss, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
            )
        )
        
        train_losses.append(loss_sum / i)
        test_losses.append(test_loss)
        
        torch.save(model.state_dict(), os.path.join(log_path, "parameter_epoch%d.pth" % (epoch)))
    
    print(train_losses)
    print(test_losses)

if __name__ == '__main__':
    TRAIN_DATA_LABELS = "/scratch/lustre/home/dost2904/Corrupted_speech/labels.csv"
    SERIALIZED_DATA_FOLDER_TRAIN = "/scratch/lustre/home/dost2904/serialized_train_dccrn/"
    SERIALIZED_DATA_FOLDER_TEST = "/scratch/lustre/home/dost2904/serialized_test_dccrn/"

    df_labels = pd.read_csv(TRAIN_DATA_LABELS)
    df_train, df_test = train_test_split(df_labels, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    cut_wav(SERIALIZED_DATA_FOLDER_TRAIN, df_train, 37.5, 16000)
    cut_wav(SERIALIZED_DATA_FOLDER_TEST, df_test, 37.5, 16000)

    data_verify(SERIALIZED_DATA_FOLDER_TRAIN)
    data_verify(SERIALIZED_DATA_FOLDER_TEST)

    batch_size = 5
    max_epoch = 200
    device = torch.device("cuda:0")
    lr = 0.00001

    train_dataset = WavDataset(SERIALIZED_DATA_FOLDER_TRAIN)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = WavDataset(SERIALIZED_DATA_FOLDER_TEST)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dccrn = DCCRN_(
        n_fft=512, hop_len=int(6.25 * 16000 / 1000), net_params=get_net_params(), batch_size=batch_size,
        device=device, win_length=int((25 * 16000 / 1000))).to(device)

    optimizer = torch.optim.Adam(dccrn.parameters(), lr=lr)
    criterion = SiSnr()

    train_losses = []
    test_losses = []
    train_bar = tqdm(range(max_epoch))
    
    test_loss = 0
    min_loss = 100
    lr_patience = 0

    for epoch in train_bar:
        loss_sum = 0
        i = 0

        for step, (x, y) in enumerate(train_dataloader):
            x = x.view(x.size(0) * x.size(1), x.size(2)).float().squeeze(0).to(device)
            y = y.view(y.size(0) * y.size(1), y.size(2)).float().squeeze(0).to(device)

            dccrn.train()
            optimizer.zero_grad()
            y_p = dccrn(x)
            loss = criterion(source=y.unsqueeze(1), estimate_source=y_p)

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            i += 1

        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        dccrn.eval()
        with torch.no_grad():
            test_loss_sum = 0
            i_test = 0
            for step, (x, y) in enumerate(test_dataloader):
                x = x.view(x.size(0) * x.size(1), x.size(2)).float().squeeze(0).to(device)
                y = y.view(y.size(0) * y.size(1), y.size(2)).float().squeeze(0).to(device)
                
                y_p = dccrn(x)
                loss = criterion(source=y.unsqueeze(1), estimate_source=y_p)
                test_loss_sum += loss.item()
                i_test += 1

        train_bar.set_description(
            'Epoch {} out of {} step {}: train loss {:.4f}, test loss {:.4f}'
                .format(epoch + 1, max_epoch, step, loss_sum / i, test_loss_sum / i_test))
        
        train_losses.append(loss_sum / i)
        test_losses.append(test_loss_sum / i_test)
        
        torch.save(dccrn.state_dict(), os.path.join('./', "weights/parameter_epoch%d.pth" % (epoch)))

        # check if loss improved
        if min_loss>=loss_sum/i:
            #assign new loss
            min_loss = test_loss_sum/i_test
            lr_patience = 0
        else:
            lr_patience += 1

        # if loss is not improved for 10 epochs in a row, stop training
        if lr_patience == 10:
            break
    
    # printing because it will be stored into logs file
    print(train_losses)
    print(test_losses)

