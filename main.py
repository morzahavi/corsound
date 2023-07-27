import librosa.display
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import os
import librosa
import librosa.display

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'DejaVu Sans'
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import tensorflow as tf

import re, math
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow_io as tfio
import tensorflow_probability as tfp

import yaml
from IPython import display as ipd
import json
from datetime import datetime

from glob import glob
from tqdm.notebook import tqdm
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from IPython import display as ipd

import itertools
import scipy
import warnings

# Show less log messages
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Set true to show less logging messages
os.environ["WANDB_SILENT"] = "true"
import wandb

class CFG:
    wandb = True
    project = "fake-speech-detection"
    debug = False
    exp_name = "v0"
    comment = "Conformer-128x80-cosine-no_aug-no_fc"

    # Use verbose=0 for silent, 1 for interactive
    verbose = 0
    display_plot = True

    # Device for training
    device = None  # device is automatically selected

    # Model & Backbone
    model_name = "Conformer"

    # Seeding for reproducibility
    seed = 101

    # Audio params
    sample_rate = 16000
    duration = 3.5 # duration in second
    audio_len = int(sample_rate * duration)
    normalize = True

    # Spectrogram params
    spec_freq = 128 # freq axis
    n_fft = 2048
    spec_time = 256 # time axis
    hop_len = audio_len//(spec_time - 1) # non-overlap region
    fmin = 20
    fmax = sample_rate//2 # max frequency
    spec_shape = [spec_time, spec_freq] # output spectrogram shape

    # Audio Augmentation
    timeshift_prob = 0.0
    gn_prob = 0.0

    # Spectrogram Augmentation
    time_mask = 20
    freq_mask = 10
    cutmix_prob = 0.0
    cutmix_alpha = 2.5
    mixup_prob = 0.0
    mixup_alpha = 2.5

    # Batch Size & Epochs
    batch_size = 32
    drop_remainder = False
    epochs = 12
    steps_per_execution = None

    # Loss & Optimizer & LR Scheduler
    loss = "binary_crossentropy"
    optimizer = "Adam"
    lr = 1e-4
    lr_schedule = "cosine"

    # Augmentation
    augment = False

    # Clip values to [0, 1]
    clip = False


def seeding(SEED):
    """
    Sets all random seeds for the program (Python, NumPy, and TensorFlow).
    """
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    #     os.environ["TF_CUDNN_DETERMINISTIC"] = str(SEED)
    tf.random.set_seed(SEED)
    print("seeding done!!!")


seeding(CFG.seed)

def configure_device():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # connect to tpu cluster
        strategy = tf.distribute.TPUStrategy(tpu) # get strategy for tpu
        print('> Running on TPU ', tpu.master(), end=' | ')
        print('Num of TPUs: ', strategy.num_replicas_in_sync)
        device='TPU'
    except: # otherwise detect GPUs
        tpu = None
        gpus = tf.config.list_logical_devices('GPU') # get logical gpus
        ngpu = len(gpus)
        if ngpu: # if number of GPUs are 0 then CPU
            strategy = tf.distribute.MirroredStrategy(gpus) # single-GPU or multi-GPU
            print("> Running on GPU", end=' | ')
            print("Num of GPUs: ", ngpu)
            device='GPU'
        else:
            print("> Running on CPU")
            strategy = tf.distribute.get_strategy() # connect to single gpu or cpu
            device='CPU'
    return strategy, device, tpu

strategy, CFG.device, tpu = configure_device()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


BASE_PATH = '/asvpoof/LA'
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





train_df = pd.read_csv(f'asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                       sep=" ", header=None)
train_df.columns =['speaker_id','filename','system_id','null','class_name']
train_df.drop(columns=['null'],inplace=True)
train_df['filepath'] = f'asvspoof/LA/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
train_df['target'] = (train_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real
if DEBUG:
    train_df = train_df.groupby(['target']).sample(2500).reset_index(drop=True)
print(f'Train Samples: {len(train_df)}')
train_df.head(2)

valid_df = pd.read_csv(f'asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                       sep=" ", header=None)
valid_df.columns =['speaker_id','filename','system_id','null','class_name']
valid_df.drop(columns=['null'],inplace=True)
valid_df['filepath'] = f'asvspoof/LA/ASVspoof2019_LA_dev/flac/'+valid_df.filename+'.flac'
valid_df['target'] = (valid_df.class_name=='spoof').astype('int32')
if DEBUG:
    valid_df = valid_df.groupby(['target']).sample(2000).reset_index(drop=True)
print(f'Valid Samples: {len(valid_df)}')
valid_df.head(2)

test_df = pd.read_csv(f'asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                      sep=" ", header=None)
test_df.columns =['speaker_id','filename','system_id','null','class_name']
test_df.drop(columns=['null'],inplace=True)
test_df['filepath'] = f'asvspoof/LA/ASVspoof2019_LA_eval/flac/'+test_df.filename+'.flac'
test_df['target'] = (test_df.class_name=='spoof').astype('int32')
if DEBUG:
    test_df = test_df.groupby(['target']).sample(2000).reset_index(drop=True)
print(f'Test Samples: {len(test_df)}')
test_df.head(2)


def load_audio(path, sr=16000):
    """load audio from .wav file
    Args:
        path: file path of .wav file
        sr: sample rate
    Returns:
        audio, sr
    """
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr

def plot_audio(audio, sr=16000):
    fig = librosa.display.waveshow(audio,
                                   x_axis='time',
                                   sr=sr)
    return fig

def listen_audio(audio, sr=16000):
    display(ipd.Audio(audio, rate=sr))

def get_spec(audio):
    spec = librosa.feature.melspectrogram(audio, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)
    return spec

def plot_spec(spec, sr=16000):
    fig = librosa.display.specshow(spec,
                                   x_axis='time',
                                   y_axis='hz',
                                   hop_length=HOP_LEN,
                                   sr=SAMPLE_RATE,
                                   fmax=FMAX,)
    return fig


row = train_df[train_df.target==0].iloc[10]
print(f'> Filename: {row.filename} | Label: {row.class_name}')
audio, sr= load_audio(row.filepath, sr=None)
audio = audio[:AUDIO_LEN]
spec = get_spec(audio)

# print('# Listen')
# listen_audio(audio, sr=16000)

print("# Plot\n")
plt.figure(figsize=(12*2,5))

plt.subplot(121)
plot_audio(audio)
plt.title("Waveform",fontsize=17)
plt.tight_layout()
plt.savefig('wave_form_0')
plt.close()


plt.subplot(122)
plot_spec(spec);
plt.title("Spectrogram",fontsize=17)
plt.tight_layout()
plt.savefig('wave_spect_0')
plt.close()

row = train_df[train_df.target==1].iloc[10]
print(f'Filename: {row.filename} | Label: {row.class_name}')
audio, sr= load_audio(row.filepath, sr=None)
audio = audio[:AUDIO_LEN]
spec = get_spec(audio)

print('# Listen')
listen_audio(audio, sr=16000)

print("# Plot\n")
plt.figure(figsize=(12*2,5))

plt.subplot(121)
plot_audio(audio)
plt.title("Waveform",fontsize=17)
plt.tight_layout()
plt.savefig('wave_form_1')
plt.close()

plt.subplot(122)
plot_spec(spec);
plt.title("Spectrogram",fontsize=17)
plt.tight_layout()
plt.savefig('wave_spect_1')
plt.close()



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# Split valid data into folds
for fold, (_, val_idx) in enumerate(skf.split(valid_df, y=valid_df['target'])):
    valid_df.loc[val_idx, 'fold'] = fold

# Split test data into folds
for fold, (_, val_idx) in enumerate(skf.split(test_df, y=test_df['target'])):
    test_df.loc[val_idx, 'fold'] = fold
display(test_df.groupby(['fold','target']).size())





import re
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
## TFRecord Data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def train_serialize_example(feature0, feature1, feature2,
                            feature3, feature4,feature5,feature6):
    feature = {
        'audio':_bytes_feature(feature0),
        'id':_bytes_feature(feature1),
        'speaker_id':_bytes_feature(feature2),
        'system_id':_bytes_feature(feature3),
        'class_name':_bytes_feature(feature4),
        'audio_len':_int64_feature(feature5),
        'target':_int64_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()





## Write TFRecord
os.makedirs('/tmp/asvspoof', exist_ok=True)
##
def write_tfrecord(df, split='train', show=True):
    df = df.copy()
    folds = sorted(df.fold.unique().tolist())
    for fold in tqdm(folds): # create tfrecord for each fold
        fold_df = df.query("fold==@fold").sample(frac=1.0)
        if show:
            print(); print('Writing %s TFRecord of fold %i :'%(split,fold))
        with tf.io.TFRecordWriter('/tmp/asvspoof/%s%.2i-%i.tfrec'%(split,fold,fold_df.shape[0])) as writer:
            samples = fold_df.shape[0] # samples = 200
            it = tqdm(range(samples)) if show else range(samples)
            for k in it: # images in fold
                row = fold_df.iloc[k,:]
                audio, sr = load_audio(row['filepath'])
                audio_id = row['filename']
                speaker_id = row['speaker_id']
                system_id = row['system_id']
                class_name = row['class_name']
                target = row['target']
                example  = train_serialize_example(
                    tf.audio.encode_wav(audio[...,None],sample_rate=sr),
                    str.encode(audio_id),
                    str.encode(speaker_id),
                    str.encode(system_id),
                    str.encode(class_name),
                    len(audio),
                    int(target),
                )
                writer.write(example)
            if show:
                filepath = '/tmp/asvspoof/%s%.2i-%i.tfrec'%(split,fold,fold_df.shape[0])
                filename = filepath.split('/')[-1]
                filesize = os.path.getsize(filepath)/10**6
                print(filename,':',np.around(filesize, 2),'MB')

##
write_tfrecord(train_df,split='train', show=True)
write_tfrecord(valid_df,split='valid', show=True)
write_tfrecord(test_df,split='test', show=True)


TRAIN_FILENAMES = tf.io.gfile.glob(BASE_PATH + '/asvspoof/train*.tfrec')
VALID_FILENAMES = tf.io.gfile.glob(BASE_PATH + '/asvspoof/valid*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(BASE_PATH + '/asvspoof/test*.tfrec')

print('# NUM TRAIN: {:,}'.format(count_data_items(TRAIN_FILENAMES)))
print('# NUM VALID: {:,}'.format(count_data_items(VALID_FILENAMES)))
print('# NUM TEST: {:,}'.format(count_data_items(TEST_FILENAMES)))