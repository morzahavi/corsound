# cs_functions.py
# Define all the function needed

# OS
import os
import shutil
# Show less log messages
import warnings
warnings.filterwarnings("ignore")
# Essentials
import pandas as pd
import numpy as np
import re, math
# Sound
import librosa
import librosa.display
import soundfile as sf
# Notebooks
import IPython.display as ipd
from tqdm.notebook import tqdm
# DL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
import tensorflow_io as tfio
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import keras
import tensorflow.keras.backend as K
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import wandb
# Visualisations
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'DejaVu Sans'
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
# Misc
import random
from glob import glob
import argparse
os.environ["WANDB_SILENT"] = "true"

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

def decode_audio(data, audio_len, target_len=1*16000):
    audio, sr = tf.audio.decode_wav(data)
    audio = tf.reshape(audio, [audio_len]) # explicit size needed for TPU
    audio = audio[:target_len]
    audio = tf.cast(audio,tf.float32)
    # compute min and max
    min_ = tf.reduce_min(audio)
    max_ = tf.reduce_max(audio)
    # normalization
    audio = (audio - min_) / (max_ - min_) # mean=0 & std=1
    return audio

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "audio" : tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "audio_len" : tf.io.FixedLenFeature([], tf.int64),
        "target" : tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    audio_len = example['audio_len']
    audio = decode_audio(example['audio'], audio_len)
    target = example['target']
    return audio, target  # returns a dataset of (image, label) pairs

def load_dataset(fileids, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(fileids, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_dataset(FILENAMES):
    dataset = load_dataset(FILENAMES, labeled=True)
    #     dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(100, seed=SEED)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(fileids):
    # the number of data items is written in the id of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(fileid).group(1)) for fileid in fileids]
    return np.sum(n)

def display_batch(batch, row=2, col=5):
    audios, targets = batch
    plt.figure(figsize=(col*5, 5*row))
    for idx in range(row*col):
        audio = audios[idx,]
        target = targets[idx,]
        plt.subplot(row, col, idx+1)
        plt.plot(audio, color='r' if target else 'b')
        plt.title('Fake' if target else 'Real', fontsize=15)
    #         plt.xticks([])
    #         plt.yticks([])
    plt.tight_layout()

##
def get_metrics():
    acc = tf.keras.metrics.BinaryAccuracy()
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='macro')
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    return [acc, precision, recall, f1_score]
##
def get_lr_callback(cfg):
    lr_start   = cfg['LR_START']
    lr_max     = cfg['LR_MAX'] * strategy.num_replicas_in_sync
    lr_min     = cfg['LR_MIN']
    lr_ramp_ep = cfg['LR_RAMPUP_EPOCHS']
    lr_sus_ep  = cfg['LR_SUSTAIN_EPOCHS']
    lr_decay   = cfg['LR_EXP_DECAY']

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
##

def Conformer(input_shape=(128, 80, 1),num_classes=1, final_activation='sigmoid', pretrain=True):
    """Souce Code: https://github.com/awsaf49/audio_classification_models"""
    inp = tf.keras.layers.Input(shape=input_shape)
    backbone = acm.ConformerEncoder()
    out = backbone(inp)
    if pretrain:
        acm.utils.weights.load_pretrain(backbone, url=URL)
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    #     out = tf.keras.layers.Dense(32, activation='selu')(out)
    out = tf.keras.layers.Dense(num_classes, activation=final_activation)(out)
    model = tf.keras.models.Model(inp, out)
    return model
##
def get_model(name=CFG.model_name, loss=CFG.loss,):
    model = Conformer(input_shape=[*CFG.spec_shape,1],pretrain=True)
    lr = CFG.lr
    if CFG.optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif CFG.optimizer == "AdamW":
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=lr)
    elif CFG.optimizer == "RectifiedAdam":
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    else:
        raise ValueError("Wrong Optimzer Name")
    model.compile(
        optimizer=opt,
        loss=loss,
        steps_per_execution=CFG.steps_per_execution, # to reduce idle time
        metrics=get_metrics()
    )
    return model

def wandb_init():
    "initialize project on wandb"
    id_ = wandb.util.generate_id() # generate random id
    config = {k: v for k, v in dict(vars(CFG)).items() if "__" not in k} # convert class to dict
    config["id"] = id_
    run = wandb.init(
        id=id_,
        project="fake-speech-detection",
        name=f"dim-{CFG.spec_shape[0]}x{CFG.spec_shape[1]}|model-{CFG.model_name}",
        config=config,
        anonymous=anonymous,
        group=CFG.comment,
        reinit=True,
        resume="allow",
    )
    return run

# Initialize wandb Run
if CFG.wandb:
    run = wandb_init()
    WandbCallback = wandb.keras.WandbCallback(save_model=False)


# Calculating EER
def calculate_eer(true_labels, predicted_probs):
    # Convert the probabilities to scores by taking the negative log-likelihood
    scores = -np.log(predicted_probs)

    # Calculate the False Accept Rate (FAR) and False Reject Rate (FRR) at various threshold points
    fars, frrs, thresholds = [], [], np.arange(0, 1, 0.001)
    for threshold in thresholds:
        predictions = (scores >= -np.log(threshold)).astype(int)
        cm = confusion_matrix(true_labels, predictions)
        fn = cm[1, 0]  # False Negative
        fp = cm[0, 1]  # False Positive
        tn = cm[0, 0]  # True Negative
        tp = cm[1, 1]  # True Positive

        far = fp / (fp + tn)  # False Accept Rate
        frr = fn / (fn + tp)  # False Reject Rate

        fars.append(far)
        frrs.append(frr)

    # Interpolate the values to find the threshold where FAR and FRR are equal (EER)
    fars = np.asarray(fars)
    frrs = np.asarray(frrs)
    eer_interpolator = interp1d(fars, thresholds)
    eer = brentq(lambda x: 1. - x - eer_interpolator(x), 0., 1.)

    return eer




































