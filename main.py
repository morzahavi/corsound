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
plt.savefig('wave_spect')
plt.close()
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

os.makedirs('/tmp/asvspoof', exist_ok=True)

write_tfrecord(train_df,split='train', show=True)
write_tfrecord(valid_df,split='valid', show=True)
write_tfrecord(test_df,split='test', show=True)

# Create the tf files
# if args.new_run.lower() == "yes":
#     write_tfrecord(train_df,split='train', show=True)
#     write_tfrecord(valid_df,split='valid', show=True)
#     write_tfrecord(test_df,split='test', show=True)
# else:
#     pass

BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE
TRAIN_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/train*.tfrec')
VALID_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/valid*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/test*.tfrec')

# Display Batch Images
ds = get_dataset(TRAIN_FILENAMES)
batch = next(iter(ds))
display_batch(batch, row=2, col=4)
plt.savefig('batch')
plt.close()










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

# Usage example:
eer = calculate_eer(test_labels, test_preds)
print("Equal Error Rate (EER):", eer)



