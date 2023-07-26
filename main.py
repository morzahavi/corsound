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


##
import tensorflow_io as tfio
ds = get_dataset(TRAIN_FILENAMES[:2], augment=False, cache=False, repeat=False).take(1)
batch = next(iter(ds.unbatch().batch(20)))
imgs, tars = batch
print(f'image_shape: {imgs.shape} target_shape:{tars.shape}')
print(f'image_dtype: {imgs.dtype} target_dtype:{tars.dtype}')
display_batch(batch, row=3, col=3)
plt.savefig('non_aug_batch')
plt.close()
##
ds = get_dataset(TRAIN_FILENAMES[:2], augment=True, cache=False, repeat=False).take(1)
batch = next(iter(ds.unbatch().batch(20)))
imgs, tars = batch
display_batch(batch, row=3, col=3)
plt.savefig('aug_batch')
plt.close()

import audio_classification_models as acm
model = acm.Conformer(input_shape=(128,80,1), pretrain=True)
##
import audio_classification_models as acm

URL = 'https://github.com/awsaf49/audio_classification_models/releases/download/v1.0.8/conformer-encoder.h5'

import tensorflow_addons as tfa
model = get_model()
model.summary()
##
strategy, CFG.device, tpu = configure_device()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
##
config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
##
get_metrics()
##
if CFG.wandb:
    "login in wandb otherwise run anonymously"
    try:
        # Addo-ons > Secrets > WANDB
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WANDB")
        wandb.login(key=api_key)
        anonymous = None
    except:
        anonymous = "must"


# Load gcs_path of train, valid & test
TRAIN_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/train*.tfrec')
VALID_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/valid*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob('/tmp/asvspoof/test*.tfrec')

TRAIN_FILENAMES = TRAIN_FILENAMES[:2]
VALID_FILENAMES = VALID_FILENAMES[:2]
TEST_FILENAMES = TEST_FILENAMES[:2]

# Take Only 10 Files if run in Debug Mode
if CFG.debug:
    TRAIN_FILENAMES = TRAIN_FILENAMES[:2]
    VALID_FILENAMES = VALID_FILENAMES[:2]
    TEST_FILENAMES = TEST_FILENAMES[:2]

# Shuffle train files
random.shuffle(TRAIN_FILENAMES)

# Count train and valid samples
NUM_TRAIN = count_data_items(TRAIN_FILENAMES)
NUM_VALID = count_data_items(VALID_FILENAMES)
NUM_TEST = count_data_items(TEST_FILENAMES)

# Compute batch size & steps_per_epoch
BATCH_SIZE = CFG.batch_size * REPLICAS
STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE

print("#" * 60)
print("#### IMAGE_SIZE: (%i, %i) | BATCH_SIZE: %i | EPOCHS: %i"% (CFG.spec_shape[0],
                                                                  CFG.spec_shape[1],
                                                                  BATCH_SIZE,
                                                                  CFG.epochs))
print("#### MODEL: %s | LOSS: %s"% (CFG.model_name, CFG.loss))
print("#### NUM_TRAIN: {:,} | NUM_VALID: {:,}".format(NUM_TRAIN, NUM_VALID))
print("#" * 60)

# Log in w&B before training
if CFG.wandb:
    wandb.log(
        {
            "num_train": NUM_TRAIN,
            "num_valid": NUM_VALID,
            "num_test": NUM_TEST,
        }
    )

# Build model in device
K.clear_session()
with strategy.scope():
    model = get_model(name=CFG.model_name,loss=CFG.loss)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "/models/ckpt.h5",
    verbose=CFG.verbose,
    monitor="val_f1_score",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
)
# callbacks = [checkpoint, get_lr_callback(mode=CFG.lr_schedule,epochs=CFG.epochs)]

# if CFG.wandb:
# Include w&b callback if WANDB is True
# callbacks.append(WandbCallback)

# Create train & valid dataset
train_ds = get_dataset(
    TRAIN_FILENAMES,
    augment=CFG.augment,
    batch_size=BATCH_SIZE,
    cache=False,
    drop_remainder=False,
)
valid_ds = get_dataset(
    VALID_FILENAMES,
    shuffle=False,
    augment=False,
    repeat=False,
    batch_size=BATCH_SIZE,
    cache=False,
    drop_remainder=False,
)

# Train model
history = model.fit(
    train_ds,
    epochs=CFG.epochs if not CFG.debug else 2,
    steps_per_epoch=STEPS_PER_EPOCH,
    # callbacks=callbacks,
    validation_data=valid_ds,
    #         validation_steps = NUM_VALID/BATCH_SIZE,
    verbose=CFG.verbose,
)

# Convert dict history to df history
history = pd.DataFrame(history.history)

# Load best weights
model.load_weights("/models/ckpt.h5")

# Plot Training History
if CFG.display_plot:
    plot_history(history)

#
eer = calculate_eer(test_labels, test_preds)
print("Equal Error Rate (EER):", eer)



