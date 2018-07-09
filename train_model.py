# created by Sijmen van der Willik
# 05/07/2018 09:55

import os
import argparse

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle


def parse_data(data_dir='./data'):
    samples = []

    correction_factor = 0.2

    subdirs = [os.path.join(data_dir, o) for o in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, o))]

    for subdir in subdirs:
        print("Adding from: {}".format(subdir))
        with open(os.path.join(subdir, 'driving_log.csv'), 'r+') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split(",")

            base_angle = float(parts[3])

            # center
            im_path = parts[0]
            samples.append([im_path, base_angle])

            # left
            im_path = parts[1]
            samples.append([im_path, base_angle + correction_factor])

            # right
            im_path = parts[2]
            samples.append([im_path, base_angle - correction_factor])

    return samples


def train_model(train_gen, validation_gen):
    print("Training for {} epochs...".format(args.epochs))
    model = keras.models.Sequential()

    # crop and normalize
    model.add(keras.layers.Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(keras.layers.Lambda(lambda x: x/255.0 - 0.5))

    # add conv layers
    model.add(keras.layers.Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(keras.layers.Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(keras.layers.Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

    # add dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dense(50))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer='adam')

    result_hist = model.fit_generator(train_gen, steps_per_epoch=samples_per_epoch/batch_size,
                                      validation_steps=samples_per_valid_epoch/batch_size,
                                      validation_data=validation_gen, epochs=int(args.epochs))

    model.save('model.h5')

    return result_hist


def generator(samples):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                tmp_img = cv2.imread(batch_sample[0])
                tmp_angle = batch_sample[1]
                images.append(tmp_img)
                angles.append(tmp_angle)

                # add augmented version
                image_flipped = np.fliplr(tmp_img)
                measurement_flipped = -tmp_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs")
    args = parser.parse_args()

    batch_size = 32
    batch_size //= 2

    print("Parsing data...")
    data_samples = parse_data()

    train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    print("No. of examples: {}".format(len(data_samples)))
    samples_per_epoch = len(train_samples)
    samples_per_valid_epoch = len(validation_samples)

    print("Training model...")
    train_model(train_generator, validation_generator)
