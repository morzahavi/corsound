import os
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow_io as tfio
import csfunctions
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from csfunctions.csfunctions import load_audio, get_spec, listen_audio, plot_audio, plot_spec, write_tfrecord, \
    display_batch, get_dataset

print("1")
# Parser
def get_argparser():
    parser = argparse.ArgumentParser()
    # Datset Options
    parser.add_argument("--new_run", type=str,
                        help="Whether to create the tf files. Enter 'yes' if it is the first time you run this script, and 'no' otherwise")
    return parser

    #
    ##
    BASE_PATH = 'asvspoof/LA'
    FOLDS = 10
    SEED = 101
    DEBUG = True
    # Audio params
    SAMPLE_RATE = 16000
    DURATION = 5.0 # duration in second
    AUDIO_LEN = int(SAMPLE_RATE * DURATION)

    # Spectrogram params
    N_MELS = 128 # freq axis
    N_FFT = 2048
    SPEC_WIDTH = 256 # time axis
    HOP_LEN = AUDIO_LEN//(SPEC_WIDTH - 1) # non-overlap region
    FMAX = SAMPLE_RATE//2 # max frequency
    SPEC_SHAPE = [SPEC_WIDTH, N_MELS] # output spectrogram shape


    train_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                           sep=" ", header=None)
    train_df.columns =['speaker_id','filename','system_id','null','class_name']
    train_df.drop(columns=['null'],inplace=True)
    train_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
    train_df['target'] = (train_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real
    if DEBUG:
        train_df = train_df.groupby(['target']).sample(2500).reset_index(drop=True)


    valid_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                           sep=" ", header=None)
    valid_df.columns =['speaker_id','filename','system_id','null','class_name']
    valid_df.drop(columns=['null'],inplace=True)
    valid_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_dev/flac/'+valid_df.filename+'.flac'
    valid_df['target'] = (valid_df.class_name=='spoof').astype('int32')
    if DEBUG:
        valid_df = valid_df.groupby(['target']).sample(2000).reset_index(drop=True)


    test_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                          sep=" ", header=None)
    test_df.columns =['speaker_id','filename','system_id','null','class_name']
    test_df.drop(columns=['null'],inplace=True)
    test_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_eval/flac/'+test_df.filename+'.flac'
    test_df['target'] = (test_df.class_name=='spoof').astype('int32')
    if DEBUG:
        test_df = test_df.groupby(['target']).sample(2000).reset_index(drop=True)


    row = train_df[train_df.target==0].iloc[10]
    audio, sr = load_audio(row.filepath, sr=None)
    audio = audio[:AUDIO_LEN]
    spec = get_spec(audio)

    plt.figure(figsize=(12*2,5))

    plt.subplot(121)
    plot_audio(audio)
    plt.title("Waveform",fontsize=17)

    plt.subplot(122)
    plot_spec(spec);
    plt.title("Spectrogram",fontsize=17)

    plt.tight_layout()
    plt.savefig('wave_specttest')
    plt.close()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
