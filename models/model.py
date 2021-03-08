import os
import shutil
# import kaggle
import subprocess
import tarfile
import tensorflow as tf

import keras
import pandas as pd
import ast
import numpy as np
import random
import math
from tensorflow import FixedLenFeature
from tensorflow.python.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.python.keras.preprocessing.image import load_img as load_img



# Upload the API token.
# def get_kaggle_credentials():
#     token_dir = os.path.join(os.path.expanduser("~"),".kaggle")
#     token_file = os.path.join(token_dir, "kaggle.json")
#     if not os.path.isdir(token_dir):
#         os.mkdir(token_dir)
#     try:
#         with open(token_file,'r') as f:
#             pass
#     except IOError as no_file:
#         try:
#             from google.colab import files
#         except ImportError:
#             raise no_file
#
#         uploaded = files.upload()
#
#         if "kaggle.json" not in uploaded:
#             raise ValueError("You need an API key! see: "
#                            "https://github.com/Kaggle/kaggle-api#api-credentials")
#         with open(token_file, "wb") as f:
#             f.write(uploaded["kaggle.json"])
#         os.chmod(token_file, 600)
#
# get_kaggle_credentials()

# This will download the data (600 MB)
# competition_name = 'planet-understanding-the-amazon-from-space'
# os.system(f'kaggle competitions download -c {competition_name} -f train-jpg.tar.7z')
# # kaggle.api.competitions_data_download_file(competition_name, file_name='train-jpg.tar.7z')

# This will extract the data
subprocess.call('p7zip -d /app/models/train-jpg.tar.7z'.split(' '))
with tarfile.open('/app/train-jpg.tar', 'r:') as f:
    f.extractall()




df_train = pd.read_csv('https://raw.githubusercontent.com/sdcubber/keras-training-serving/master/KagglePlanetMCML.csv')
df_train.head()

IM_SIZE = 128 # image size

image_input = tf.keras.Input(shape=(IM_SIZE, IM_SIZE, 3), name='input_layer')

# Some convolutional layers
conv_1 = tf.keras.layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(image_input)
conv_1 = tf.keras.layers.MaxPooling2D(padding='same')(conv_1)
conv_2 = tf.keras.layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(conv_1)
conv_2 = tf.keras.layers.MaxPooling2D(padding='same')(conv_2)

# Flatten the output of the convolutional layers
conv_flat = tf.keras.layers.Flatten()(conv_2)

# Some dense layers with two separate outputs
fc_1 = tf.keras.layers.Dense(128,
                             activation='relu')(conv_flat)
fc_1 = tf.keras.layers.Dropout(0.2)(fc_1)
fc_2 = tf.keras.layers.Dense(128,
                             activation='relu')(fc_1)
fc_2 = tf.keras.layers.Dropout(0.2)(fc_2)

# Output layers: separate outputs for the weather and the ground labels
weather_output = tf.keras.layers.Dense(4,
                                       activation='softmax',
                                       name='weather')(fc_2)
ground_output = tf.keras.layers.Dense(13,
                                      activation='sigmoid',
                                      name='ground')(fc_2)

model = tf.keras.Model(inputs=image_input, outputs=[weather_output, ground_output])
print(model.summary())

model.compile(optimizer='adam',
              loss={'weather': 'categorical_crossentropy',
                    'ground': 'binary_crossentropy'})


def load_image(image_path, size):
    return img_to_array(load_img(image_path, target_size=(size, size))) / 255.


class KagglePlanetSequence(tf.keras.utils.Sequence):
    """
    Custom Sequence object to train a models on out-of-memory datasets.
    """

    def __init__(self, df, data_path, im_size, batch_size, mode='train'):
        """
        df: pandas dataframe that contains columns with image names and labels
        data_path: path that contains the training images
        im_size: image size
        mode: when in training mode, data will be shuffled between epochs
        """
        self.df = df
        self.batch_size = batch_size
        self.im_size = im_size
        self.mode = mode

        # Take labels and a list of image locations in memory
        self.wlabels = self.df['weather_labels'].apply(lambda x: ast.literal_eval(x)).tolist()
        self.glabels = self.df['ground_labels'].apply(lambda x: ast.literal_eval(x)).tolist()
        self.image_list = self.df['image_name'].apply(lambda x: os.path.join(data_path, x + '.jpg')).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch
        self.indexes = range(len(self.image_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return [self.wlabels[idx * self.batch_size: (idx + 1) * self.batch_size],
                self.glabels[idx * self.batch_size: (idx + 1) * self.batch_size]]

    def get_batch_features(self, idx):
        # Fetch a batch of images
        batch_images = self.image_list[idx * self.batch_size: (1 + idx) * self.batch_size]
        return np.array([load_image(im, self.im_size) for im in batch_images])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


seq = KagglePlanetSequence(df_train,
                           '/app/train-jpg/',
                           im_size=IM_SIZE,
                           batch_size=32)



callbacks = [
    tf.keras.callbacks.ModelCheckpoint('/app/models.h5', verbose=1)
]

model.fit_generator(generator=seq,
                    verbose=1,
                    epochs=1,
                    use_multiprocessing=False,
                    workers=1,
                    callbacks=callbacks)



another_model = tf.keras.models.load_model('/app/models.h5')
another_model.fit_generator(generator=seq, verbose=1, epochs=1)



test_seq = KagglePlanetSequence(df_train,
                       '/app/train-jpg/',
                       im_size=IM_SIZE,
                       batch_size=32,
                       mode='test') # test mode disables shuffling

predictions = model.predict_generator(generator=test_seq, verbose=1)
# We get a list of two prediction arrays, for weather and for label

len(predictions[1])  == len(df_train) # Total number of images in dataset


# Serialize images, together with labels, to TF records
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


tf_records_filename = '/app/KagglePlanetTFRecord_{}'.format(IM_SIZE)
writer = tf.python_io.TFRecordWriter(tf_records_filename)

# List of image paths, np array of labels
im_list = [os.path.join('/app/train-jpg/', v + '.jpg') for v in df_train['image_name'].tolist()]
w_labels_arr = np.array([ast.literal_eval(l) for l in df_train['weather_labels']])
g_labels_arr = np.array([ast.literal_eval(l) for l in df_train['ground_labels']])

for i in range(len(df_train)):
    w_labels = w_labels_arr[i].astype(np.float32)
    g_labels = g_labels_arr[i].astype(np.float32)
    im = np.array(img_to_array(load_img(im_list[i], target_size=(IM_SIZE, IM_SIZE))) / 255.)
    w_raw = w_labels.tostring()
    g_raw = g_labels.tostring()
    im_raw = im.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(im_raw),
                                                                   'weather_labels': _bytes_feature(w_raw),
                                                                   'ground_labels': _bytes_feature(g_raw)}))

    writer.write(example.SerializeToString())

writer.close()

featdef = {
           'image': FixedLenFeature(shape=[], dtype=tf.string),
           'weather_labels': FixedLenFeature(shape=[], dtype=tf.string),
           'ground_labels': FixedLenFeature(shape=[], dtype=tf.string)
          }

def _parse_record(example_proto, clip=False):
    ex = tf.parse_single_example(example_proto, featdef)

    im = tf.decode_raw(ex['image'], tf.float32)
    im = tf.reshape(im, (IM_SIZE, IM_SIZE, 3))

    weather = tf.decode_raw(ex['weather_labels'], tf.float32)
    ground = tf.decode_raw(ex['ground_labels'], tf.float32)

    return im, (weather, ground)


# Construct a dataset iterator
batch_size = 32
ds_train = tf.data.TFRecordDataset('/app/KagglePlanetTFRecord_{}'.format(IM_SIZE)).map(_parse_record)
ds_train = ds_train.repeat().shuffle(1000).batch(batch_size)



# models = tf.keras.Model(inputs=image_input, outputs=[weather_output, ground_output])
#
# models.compile(optimizer='adam',
#               loss={'weather': 'categorical_crossentropy',
#                     'ground': 'binary_crossentropy'})
#
# history = models.fit(ds_train,
#                     steps_per_epoch=100, # let's just take some steps
#                     epochs=1)

# The easy way, with simple_save(), where the signature def is defined implicitly and stored with the
# default graph way
# The export path contains the name and the version of the models

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('/app/models.h5')

if os.path.exists('/app/tfx/PlanetModel/1'):
    shutil.rmtree('/app/tfx/PlanetModel/1')

export_path = '/app/tfx/PlanetModel/1'

# Fetch the Keras session and save the models
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})