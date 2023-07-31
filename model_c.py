#!/usr/bin/env python3
########################################################################
# Model C
########################################################################
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import librosa
import librosa.display
import re
import warnings
warnings.filterwarnings("ignore")
########################################################################
# Meta Data
# * `speaker_id` :        LA_****, a 4-digit speaker ID
# * `filename` :  LA_****, name of the audio file
# * `system_id` :         ID of the speech spoofing system `(A01 - A19)`,  or, for **real** speech SYSTEM-ID is left blank ('-')
# * `class_name` :        **bonafide** for genuine speech, or, **spoof** for fake/spoof speech
# * `target` : `1` for **fake/spoof**  and `0` for **real/genuine**
########################################################################
BASE_PATH = "/Users/morzahavi/"
TF_PATH = BASE_PATH
DEBUG = True
FOLDS = 10
SEED = 101
train_df = pd.read_csv(f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                       sep=" ", header=None)
train_df.columns = ['speaker_id', 'filename', 'system_id', 'null', 'class_name']
train_df.drop(columns=['null'], inplace=True)
train_df['filepath'] = f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_train/flac/' + train_df.filename + '.flac'
train_df['target'] = (train_df.class_name == 'spoof').astype('int32')  # set labels 1 for fake and 0 for real
if DEBUG:
    train_df = train_df.groupby(['target']).sample(500).reset_index(drop=True)
print(f'Train Samples: {len(train_df)}')
#
valid_df = pd.read_csv(f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                       sep=" ", header=None)
valid_df.columns = ['speaker_id', 'filename', 'system_id', 'null', 'class_name']
valid_df.drop(columns=['null'], inplace=True)
valid_df['filepath'] = f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_dev/flac/' + valid_df.filename + '.flac'
valid_df['target'] = (valid_df.class_name == 'spoof').astype('int32')
if DEBUG:
    valid_df = valid_df.groupby(['target']).sample(200).reset_index(drop=True)
print(f'Valid Samples: {len(valid_df)}')
valid_df.head(2)
#
test_df = pd.read_csv(f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                      sep=" ", header=None)
test_df.columns = ['speaker_id', 'filename', 'system_id', 'null', 'class_name']
test_df.drop(columns=['null'], inplace=True)
test_df['filepath'] = f'{BASE_PATH}asvspoof/LA/ASVspoof2019_LA_eval/flac/' + test_df.filename + '.flac'
test_df['target'] = (test_df.class_name == 'spoof').astype('int32')
if DEBUG:
    test_df = test_df.groupby(['target']).sample(200).reset_index(drop=True)
print(f'Test Samples: {len(test_df)}')
# Write TFRecord
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
SAMPLE_RATE = 16000
DURATION = 5.0  # duration in second
AUDIO_LEN = int(SAMPLE_RATE * DURATION)
# Spectrogram params
N_MELS = 128  # freq axis
N_FFT = 2048
SPEC_WIDTH = 256  # time axis
HOP_LEN = AUDIO_LEN // (SPEC_WIDTH - 1)  # non-overlap region
FMAX = SAMPLE_RATE // 2  # max frequency
row = train_df[train_df.target == 0].iloc[10]
def get_spec(audio):
    spec = librosa.feature.melspectrogram(audio, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)
    return spec
audio, sr = load_audio(row.filepath, sr=None)
audio = audio[:AUDIO_LEN]
spec = get_spec(audio)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# Split train data into folds
for fold, (_, val_idx) in enumerate(skf.split(train_df, y=train_df['target'])):
    train_df.loc[val_idx, 'fold'] = fold

# Split valid data into folds
for fold, (_, val_idx) in enumerate(skf.split(valid_df, y=valid_df['target'])):
    valid_df.loc[val_idx, 'fold'] = fold

# Split test data into folds
for fold, (_, val_idx) in enumerate(skf.split(test_df, y=test_df['target'])):
    test_df.loc[val_idx, 'fold'] = fold

train_df.fold.value_counts()
## TFRecord Data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def train_serialize_example(feature0, feature1, feature2,
                            feature3, feature4, feature5, feature6):
    feature = {
        'audio': _bytes_feature(feature0),
        'id': _bytes_feature(feature1),
        'speaker_id': _bytes_feature(feature2),
        'system_id': _bytes_feature(feature3),
        'class_name': _bytes_feature(feature4),
        'audio_len': _int64_feature(feature5),
        'target': _int64_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


os.makedirs(f'{TF_PATH}tmp/asvspoof', exist_ok=True)
def write_tfrecord(df, split='train', show=True):
    df = df.copy()
    folds = sorted(df.fold.unique().tolist())
    for fold in tqdm(folds):  # create tfrecord for each fold
        fold_df = df.query("fold==@fold").sample(frac=1.0)
        if show:
            print();
            print('Writing %s TFRecord of fold %i :' % (split, fold))
        with tf.io.TFRecordWriter(f'{TF_PATH}/tmp/asvspoof/%s%.2i-%i.tfrec' % (split, fold, fold_df.shape[0])) as writer:
            samples = fold_df.shape[0]  # samples = 200
            it = tqdm(range(samples)) if show else range(samples)
            for k in it:  # images in fold
                row = fold_df.iloc[k, :]
                audio, sr = load_audio(row['filepath'])
                audio_id = row['filename']
                speaker_id = row['speaker_id']
                system_id = row['system_id']
                class_name = row['class_name']
                target = row['target']
                example = train_serialize_example(
                    tf.audio.encode_wav(audio[..., None], sample_rate=sr),
                    str.encode(audio_id),
                    str.encode(speaker_id),
                    str.encode(system_id),
                    str.encode(class_name),
                    len(audio),
                    int(target),
                )
                writer.write(example)
            if show:
                filepath = f'{TF_PATH}/tmp/asvspoof/%s%.2i-%i.tfrec' % (split, fold, fold_df.shape[0])
                filename = filepath.split('/')[-1]
                filesize = os.path.getsize(filepath) / 10 ** 6
                print(filename, ':', np.around(filesize, 2), 'MB')


write_tfrecord(train_df, split='train', show=True)
write_tfrecord(valid_df, split='valid', show=True)
write_tfrecord(test_df, split='test', show=True)
def count_data_items(fileids):
    # the number of data items is written in the id of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(fileid).group(1)) for fileid in fileids]
    return np.sum(n)
BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE
TRAIN_FILENAMES = tf.io.gfile.glob(f'{TF_PATH}tmp/asvspoof/train*.tfrec')
VALID_FILENAMES = tf.io.gfile.glob(f'{TF_PATH}tmp/asvspoof/valid*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(f'{TF_PATH}tmp/asvspoof/test*.tfrec')
print('There are %i train, %i valid & %i test images' % (count_data_items(TRAIN_FILENAMES),
                                                         count_data_items(VALID_FILENAMES),
                                                         count_data_items(TEST_FILENAMES)))


