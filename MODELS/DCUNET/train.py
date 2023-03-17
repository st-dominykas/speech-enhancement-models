import gc
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data_load import *
from model import *


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    x_ = torch.squeeze(x_, 1)
    y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0
    for noisy_x, clean_x in train_loader:

        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item() 
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

def test_epoch(net, test_loader, loss_fn):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss

def train_(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    # Early stopping
    last_loss = 1
    patience = 3
    triggertimes = 0

    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        # first evaluating for comparison
        if e == 0:
            with torch.no_grad():
                test_loss = test_epoch(net, test_loader, loss_fn)
                
            test_losses.append(test_loss)
            print("Loss before training:{:.6f}".format(test_loss))
          

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        scheduler.step()
        with torch.no_grad():
          test_loss = test_epoch(net, test_loader, loss_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e+1, epochs),
                      "Loss: {:.6f}...".format(train_loss),
                      "Test Loss: {:.6f}".format(test_loss))

        # Early stopping from second epoch
        if len(test_losses)>1:
            current_loss = test_losses[-1]
            last_loss = test_losses[-2]
            print(f'Current Loss: {current_loss}, Last Loss: {last_loss}')

            if current_loss > last_loss:
                triggertimes += 1

                if triggertimes >= patience:
                    print('Early stopping!')
                    return train_losses, test_losses

            else:
                print('trigger times: 0')
                triggertimes = 0

    return train_losses, test_losses

if __name__ == "__main__":
    
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

    # audio parameters
    SAMPLE_RATE = 48000
    N_FFT = SAMPLE_RATE * 64 // 1000 + 4 # SAMPLE_RATE/N_FFT = FFT SAMPLE RATE
    HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4

    # read audio mapping file
    df_clean_noisy = pd.read_csv("/scratch/lustre/home/dost2904/DCUNET_data/DCUNET_files.csv")

    # split data to train and test samples
    #df_train, df_test = train_test_split(df_clean_noisy.sample(50), test_size=0.2)
    df_train, df_test = train_test_split(df_clean_noisy, test_size=0.2)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    test_dataset = SpeechDataset(df_train['Noisy'], df_train['Clean'], N_FFT, HOP_LENGTH)
    train_dataset = SpeechDataset(df_test['Noisy'], df_test['Clean'], N_FFT, HOP_LENGTH)  

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    gc.collect()
    torch.cuda.empty_cache()

    dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)

    loss_fn = wsdr_fn
    optimizer = torch.optim.Adam(dcunet20.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    train_losses, test_losses = train_(dcunet20, train_loader, test_loader, loss_fn, optimizer, scheduler, 30)

    torch.save(dcunet20.state_dict(), 'DCUNET_20__weights.pth')

    print(f'Train losses: {train_losses}')
    print(f'Test losses {test_losses}')
    