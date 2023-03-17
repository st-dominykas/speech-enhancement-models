import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import gc
from torch.autograd import Variable

from data import *
from model import *

if __name__ == "__main__":
    DTLN = Pytorch_DTLN()
    optimizer = torch.optim.Adam(DTLN.parameters(), lr=1e-4)
    criterion = SiSnr()

    SAMPLE_RATE = 16000
    NUMBER_OF_EPOCHS = 200
    BATCH_SIZE = 32
    WINDOW_SIZE = 2**15

    TRAIN_DATA_LABELS = "/scratch/lustre/home/dost2904/Corrupted_speech/labels.csv"
    SERIALIZED_DATA_FOLDER_TRAIN = "/scratch/lustre/home/dost2904/serialized_train_dtln/"
    SERIALIZED_DATA_FOLDER_TEST = "/scratch/lustre/home/dost2904/serialized_test_dtln/"

    df_labels = pd.read_csv(TRAIN_DATA_LABELS)

    # for debuging purposes
    df_train, df_test = train_test_split(df_labels, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print('Training data processing...')
    process_and_serialize(WINDOW_SIZE, SAMPLE_RATE, 'train', df_train, SERIALIZED_DATA_FOLDER_TRAIN)
    print('Test data processing...')
    process_and_serialize(WINDOW_SIZE, SAMPLE_RATE, 'test', df_test, SERIALIZED_DATA_FOLDER_TEST)

    train_dataset = AudioDataset(data_path=SERIALIZED_DATA_FOLDER_TRAIN)
    test_dataset = AudioDataset(data_path=SERIALIZED_DATA_FOLDER_TEST)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        print('Training on GPU.')
        DTLN.cuda()

    # train
    train_losses = []
    test_losses = []
    min_loss = 100
    lr_patience = 0

    print('Star training...')
    train_bar = tqdm(range(NUMBER_OF_EPOCHS))
    for epoch in train_bar:
        train_batch_counter = 0
        loss_train = 0
        test_batch_counter = 0
        loss_test = 0
        for train_clean, train_noisy in train_data_loader:
            DTLN.train()
            
            if torch.cuda.is_available():
                train_clean, train_noisy = train_clean.cuda(), train_noisy.cuda()
            train_clean, train_noisy = Variable(train_clean), Variable(train_noisy)

            DTLN.zero_grad()

            output = DTLN(train_noisy.squeeze(1))
            loss = criterion(source=train_clean.squeeze(1), estimate_source=output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(DTLN.parameters(), 3)
            optimizer.step()
            loss_train += loss.item()
            train_batch_counter += 1

        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        DTLN.eval()
        with torch.no_grad():
            for test_clean, test_noisy in test_data_loader:           
                if torch.cuda.is_available():
                    test_clean, test_noisy = test_clean.cuda(), test_noisy.cuda()
                test_clean, test_noisy = Variable(test_clean), Variable(test_noisy)

                output = DTLN(test_noisy.squeeze(1))
                loss = criterion(source=test_clean.squeeze(1), estimate_source=output)

                loss_test += loss.item()
                test_batch_counter += 1

        train_bar.set_description(
                'Epoch {} out of {}: train loss {:.4f}, test loss {:.4f}'
                    .format(epoch + 1, NUMBER_OF_EPOCHS, loss_train / train_batch_counter, loss_test / test_batch_counter))
            
        train_losses.append(loss_train / train_batch_counter)
        test_losses.append(loss_test / test_batch_counter)
        
        torch.save(DTLN.state_dict(), os.path.join('./', "weights/parameter_epoch%d.pth" % (epoch)))

        # check if loss improved
        if min_loss>=loss_train/train_batch_counter:
            #assign new loss
            min_loss = loss_train/train_batch_counter
            lr_patience = 0
        else:
            lr_patience += 1

        # if loss is not improved for 3 epoch in a row, lower LR by half
        if lr_patience == 3:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5

        # if loss is not improved for 10 epochs in a row, stop training
        if lr_patience == 10:
            break

    print(train_losses)
    print(test_losses)