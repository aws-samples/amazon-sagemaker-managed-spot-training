# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.models import load_model

from keras.optimizers import Adam, SGD, RMSprop

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"


def keras_model_fn(learning_rate, weight_decay, optimizer, momentum):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model is transformed into a TensorFlow Estimator before training and saved in a
    TensorFlow Serving SavedModel at the end of training.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', name='inputs', input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    size = 1
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate * size, decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate * size, decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate * size, decay=weight_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')


def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')


def validation_input_fn():
    return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def _input(epochs, batch_size, channel, channel_name):
    """Uses the tf.data input pipeline for CIFAR-10 dataset."""
    mode = args.data_config[channel_name]['TrainingInputMode']
    logging.info("Running {} in {} mode".format(channel_name, mode))

    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        filenames = _get_filenames(channel_name, channel)
        dataset = tf.data.TFRecordDataset(filenames)

    # Repeat infinitely.
    dataset = dataset.repeat()
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    return {INPUT_TENSOR_NAME: image_batch}, label_batch


def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(value, featdef)
    image = tf.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32,
    )
    label = tf.cast(example['label'], tf.int32)
    image = _train_preprocess_fn(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def save_model(model, output):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input}, outputs={'scores': model.output}
    )

    builder = tf.saved_model.builder.SavedModelBuilder(output+'/1/')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        },
    )

    builder.save()
    logging.info("Model successfully saved at: {}".format(output))


def load_model_from_checkpoints(checkpoint_path):
    checkpoint_files = [file for file in os.listdir(checkpoint_path) if file.endswith('.' + 'h5')]
    logging.info('------------------------------------------------------')
    logging.info("Available checkpoint files: {}".format(checkpoint_files))
    epoch_numbers = [re.search('(\.*[0-9])(?=\.)',file).group() for file in checkpoint_files]
      
    max_epoch_number = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch_number)
    max_epoch_filename = checkpoint_files[max_epoch_index]

    logging.info('Latest epoch checkpoint file name: {}'.format(max_epoch_filename))
    logging.info('Resuming training from epoch: {}'.format(int(max_epoch_number)+1))
    logging.info('------------------------------------------------------')
    
    resumed_model_from_checkpoints = load_model(f'{checkpoint_path}/{max_epoch_filename}')
    return resumed_model_from_checkpoints, int(max_epoch_number)

    
def main(args):
    if os.path.isdir(args.checkpoint_path):
        logging.info("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        logging.info("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)

    logging.info("getting data")
    train_dataset = train_input_fn()
    eval_dataset = eval_input_fn()
    validation_dataset = validation_input_fn()

    logging.info("configuring model")
    
    # Load model
    if not os.listdir(args.checkpoint_path):
        model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum)
        initial_epoch_number = 0
    else:    
        model, initial_epoch_number = load_model_from_checkpoints(args.checkpoint_path)
         
    logging.info("Checkpointing to: {}".format(args.checkpoint_path))

    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    callbacks.append(ModelCheckpoint(args.checkpoint_path + '/checkpoint-{epoch}.h5'))

    logging.info("Starting training from epoch: {}".format(initial_epoch_number+1))
    
    size = 1
    model.fit(x=train_dataset[0],
              y=train_dataset[1],
              steps_per_epoch=(num_examples_per_epoch('train') // args.batch_size) // size,
              epochs=args.epochs,
              initial_epoch=initial_epoch_number,
              validation_data=validation_dataset,
              validation_steps=(num_examples_per_epoch('validation') // args.batch_size) // size,
              callbacks=callbacks)

    score = model.evaluate(eval_dataset[0],
                           eval_dataset[1],
                           steps=num_examples_per_epoch('eval') // args.batch_size,
                           verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    save_model(model, args.model_output_dir)
        


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 40000
    elif subset == 'validation':
        return 10000
    elif subset == 'eval':
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'),help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'),help='The directory where the CIFAR-10 validation data is stored.')
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'),help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints",help="Path where checkpoints will be saved.")
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help="This is the inital learning rate value. The learning rate will decrease during training. For more details check the model_fn implementation in this file.")
    parser.add_argument('--epochs',type=int,default=10,help='The number of steps to use for training.')
    parser.add_argument('--batch-size',type=int,default=128,help='Batch size for training.')
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')

    args = parser.parse_args()
    main(args)
