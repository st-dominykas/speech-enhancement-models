import torchaudio
import numpy as np
from tqdm import tqdm
import os

# function loads the signal from the path with given sample resolution and cuts into frames that equal to given duration
def load_wav(path, frame_dur, sr=16000):
    wav, _ = torchaudio.load(path)
    wav = torchaudio.transforms.Resample(48000,sr)(wav).flatten()
    wav = wav.numpy()
    win = int(frame_dur / 1000 * sr)
    wav = wav[0:win*(len(wav)//win)]
    return np.split(wav, int(len(wav) / win), axis=0)

# function saves a pair of clean and noisy signal to given directory
def cut_wav(dir, df_labels, frame_dur, sr=16000):
    clean_frames = np.array([])
    noisy_frames = np.array([])
    saver_counter = 1

    # loop through all files
    for item in range(df_labels.shape[0]):
        if len(clean_frames)==0:
            clean_frames = load_wav(df_labels['File Name'][item], frame_dur=frame_dur, sr=sr)
            noisy_frames = load_wav(df_labels['Corrupted File Name'][item], frame_dur=frame_dur, sr=sr)
        else:
            clean_frames = np.concatenate((clean_frames, load_wav(df_labels['File Name'][item], frame_dur=frame_dur, sr=sr)))
            noisy_frames = np.concatenate((noisy_frames, load_wav(df_labels['Corrupted File Name'][item], frame_dur=frame_dur, sr=sr)))

        if len(clean_frames) >= 100:
            # save list to folder
            pair = np.array([clean_frames[:100], noisy_frames[:100]])
            np.save(dir+'/'+str(saver_counter), arr=pair)

            clean_frames = clean_frames[100:]
            noisy_frames = noisy_frames[100:]
            saver_counter += 1

    while len(clean_frames)>=100:
        pair = np.array([clean_frames[:100], noisy_frames[:100]])
        np.save(dir+'/'+str(saver_counter), arr=pair)

        clean_frames = clean_frames[100:]
        noisy_frames = noisy_frames[100:]
        saver_counter += 1

# function loops through all files in the directory and check if all they have desired length
def data_verify(path):
    for root, dirs, files in os.walk(path):
        for filename in tqdm(files):
            data_pair = np.load(os.path.join(root, filename), allow_pickle=True)
            if data_pair.shape != (2, 100, 600):
                print('Snippet length not {} : {} instead'.format(600, data_pair.shape))
                print(filename)
                break
