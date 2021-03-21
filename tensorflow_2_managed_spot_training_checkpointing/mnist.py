# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
import argparse
import os
import re
import numpy as np
import json


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_model_from_checkpoints(checkpoint_path):
    checkpoint_files = [file for file in os.listdir(checkpoint_path) if file.endswith('.' + 'h5')]
    print('------------------------------------------------------')
    print("Available checkpoint files: {}".format(checkpoint_files))
    epoch_numbers = [re.search('(\.*[0-9])(?=\.)',file).group() for file in checkpoint_files]
      
    max_epoch_number = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch_number)
    max_epoch_filename = checkpoint_files[max_epoch_index]

    print('Latest epoch checkpoint file name: {}'.format(max_epoch_filename))
    print('Resuming training from epoch: {}'.format(int(max_epoch_number)+1))
    print('------------------------------------------------------')
    
    resumed_model_from_checkpoints = tf.keras.models.load_model(f'{checkpoint_path}/{max_epoch_filename}')
    return resumed_model_from_checkpoints, int(max_epoch_number)


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs',type=int,default=10,help='The number of steps to use for training.')
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints",help="Path where checkpoints will be saved.")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    print("getting data")
    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)

    print("configuring model")
    
    # Load model
    if not os.listdir(args.checkpoint_path):
        model = model(train_data, train_labels, eval_data, eval_labels)
        initial_epoch_number = 0
    else:    
        model, initial_epoch_number = load_model_from_checkpoints(args.checkpoint_path)
         
    print("Checkpointing to: {}".format(args.checkpoint_path))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path + '/checkpoint-{epoch}.h5',
        save_freq='epoch',
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    print("Starting training from epoch: {}".format(initial_epoch_number+1))
    
    model.fit(x=train_data, 
              y=train_labels,
              epochs=args.epochs,
              initial_epoch=initial_epoch_number,
              callbacks=[model_checkpoint_callback])
    
    model.evaluate(train_data, train_labels)   

    model.save(os.path.join(args.sm_model_dir, '000000001'))
    
