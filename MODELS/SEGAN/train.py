import torch
import torch.nn as nn

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator

def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, _ = torchaudio.load(file)
    wav = torchaudio.transforms.Resample(48000,16000)(wav).flatten()
    wav = wav.numpy()
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices

def process_and_serialize(window_size, sample_rate, data_type, df_labels, serialized_folder):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5
    clean_files = df_labels['File Name']
    noisy_files = df_labels['Corrupted File Name']

    # walk through the path, slice the audio file, and save the serialized result
    for index in range(len(clean_files)):
        clean_sliced = slice_signal(clean_files[index], window_size, stride, sample_rate)
        noisy_sliced = slice_signal(noisy_files[index], window_size, stride, sample_rate)
        # serialize - file format goes [original_file]_[slice_number].npy
        # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
        file_name = re.split(r'/',clean_files[index])[-1]
        for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
            pair = np.array([slice_tuple[0], slice_tuple[1]])
            np.save(os.path.join(serialized_folder, '{}_{}'.format(file_name, idx)), arr=pair)

def data_verify(data_type, serialized_folder, window_size):
    """
    Verifies the length of each data after pre-process.
    """
    for root, _, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break

def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.
    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient
        pre: pre-emphasis or de-emphasis signals
    Returns:
        result: pre-emphasized or de-emphasized signal batch
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result

class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.
        Args:
            batch_size(int): batch size
        Returns:
            ref_batch: reference batch
        """
        ref_file_names = np.random.choice(self.file_names, batch_size)
        ref_batch = np.stack([np.load(f) for f in ref_file_names])

        ref_batch = emphasis(ref_batch, emph_coeff=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        noisy = pair[1].reshape(1, -1)
        clean = pair[0].reshape(1, -1)
        return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
            torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)

if __name__ == "__main__":
    
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    WINDOW_SIZE = 2 ** 14  # about 1 second of samples
    SAMPLE_RATE = 16000
    TRAIN_DATA_LABELS = "/scratch/lustre/home/dost2904/Corrupted_speech/labels.csv"
    SERIALIZED_DATA_FOLDER_TRAIN = "/scratch/lustre/home/dost2904/serialized_train/"
    SERIALIZED_DATA_FOLDER_TEST = "/scratch/lustre/home/dost2904/serialized_test/"
    PROCESS_DATA = False

    df_labels = pd.read_csv(TRAIN_DATA_LABELS)

    # for debuging purposes
    df_train, df_test = train_test_split(df_labels, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if PROCESS_DATA:
        process_and_serialize(WINDOW_SIZE, SAMPLE_RATE, 'train', df_train, SERIALIZED_DATA_FOLDER_TRAIN)
        process_and_serialize(WINDOW_SIZE, SAMPLE_RATE, 'test', df_test, SERIALIZED_DATA_FOLDER_TEST)

        data_verify('train', SERIALIZED_DATA_FOLDER_TRAIN, WINDOW_SIZE)
        data_verify('test', SERIALIZED_DATA_FOLDER_TEST, WINDOW_SIZE)

    train_dataset = AudioDataset(data_type='train', data_path=SERIALIZED_DATA_FOLDER_TRAIN)
    test_dataset = AudioDataset(data_type='test', data_path=SERIALIZED_DATA_FOLDER_TEST)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)
    ref_batch_test = test_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()

    if torch.cuda.is_available():
        print('Training on GPU.')
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
        ref_batch_test = ref_batch_test.cuda()
    else:
        print('Training on CPU')
    ref_batch = Variable(ref_batch)
    ref_batch_test = Variable(ref_batch_test)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.00002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002)

    #TBA
    discriminator_loss_train = []
    generator_loss_train = []
    discriminator_loss_test = []
    generator_loss_test = []

    train_bar = tqdm(range(NUM_EPOCHS))
    for epoch in train_bar:
        train_batch_counter = 0
        d_loss_train = 0
        g_loss_train = 0
        for train_batch, train_clean, train_noisy in train_data_loader:

            # latent vector - normal distribution
            z = nn.init.normal_(torch.Tensor(train_batch.size(0), 1024, 8))

            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            noisy_loss.backward()
            with torch.no_grad():
                d_loss = clean_loss * 0.5 + noisy_loss * 0.5

            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)

            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()

            train_batch_counter += 1
            d_loss_train += d_loss.data
            g_loss_train += g_loss.data

        # TEST
        test_batch_counter = 0
        d_loss_test = 0
        g_loss_test = 0

        for test_batch, test_clean, test_noisy in test_data_loader:

            # latent vector - normal distribution
            z = nn.init.normal_(torch.Tensor(test_batch.size(0), 1024, 8))

            if torch.cuda.is_available():
                test_batch, test_clean, test_noisy = test_batch.cuda(), test_clean.cuda(), test_noisy.cuda()
                z = z.cuda()
            test_batch, test_clean, test_noisy = Variable(test_batch), Variable(test_clean), Variable(test_noisy)
            z = Variable(z)

            # discriminator test loss
            outputs = discriminator(test_batch, ref_batch_test)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1

            generated_outputs = generator(test_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, test_noisy), dim=1), ref_batch_test)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            d_loss = clean_loss * 0.5 + noisy_loss * 0.5

            # generator test loss
            generated_outputs = generator(test_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, test_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch_test)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)

            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(test_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            test_batch_counter += 1
            d_loss_test += d_loss.data
            g_loss_test += g_loss.data

        train_bar.set_description(
            'Epoch {} out of {}: d_loss_train {:.4f}, g_loss_train {:.4f}, d_loss_test {:.4f}, g_loss_test {:.4f}'
                .format(epoch + 1, NUM_EPOCHS, d_loss_train/train_batch_counter, g_loss_train/train_batch_counter, d_loss_test/test_batch_counter, g_loss_test/test_batch_counter))
        

        discriminator_loss_train.append(d_loss_train/train_batch_counter)
        generator_loss_train.append(g_loss_train/train_batch_counter)
        discriminator_loss_test.append(d_loss_test/test_batch_counter)
        generator_loss_test.append(g_loss_test/test_batch_counter)

        # save the model parameters for each epoch
        torch.save(generator.state_dict(), 'weights/SEGAN_generator_weights_e{}.pkl'.format(epoch + 1))
        torch.save(discriminator.state_dict(), 'weights/SEGAN_discriminator_weights_e{}.pkl'.format(epoch + 1))

    print(discriminator_loss_train)
    print(generator_loss_train)
    print(discriminator_loss_test)
    print(generator_loss_test)
    