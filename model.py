import csv
import cv2
import numpy as np
import sklearn
import math


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #print(batch_sample)
                for i in range(3):                    
                    directory = batch_sample[0].split('/')[3]
                    name = './'+directory+'/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    if "center_" in name: ## Only do the center images for now
                        images.append(image)
                        angles.append(center_angle)

                        #FLIPPED
                        image_flipped = np.fliplr(image)
                        measurement_flipped = -center_angle
                        images.append(image_flipped)
                        angles.append(measurement_flipped)

            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#Uncomment/comment sections of code below to add/remove them to/from the dataset
samples = []


with open('./data_forward_lap_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_forward_lap_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_forward_lap_3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_recovery_back_stretch_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_recovery_turn_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('./data_smooth_turn_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_smooth_turn_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
#with open('./data_problem_spots_back_stretch_1/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)
        
with open('./data_problem_spots_back_stretch_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_reverse_lap_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_turn_1_white_lines_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
  
with open('./data_recovery_turn_1_white_lines_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_smooth_turn_1_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_smooth_turn_1_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)        
        

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36, (5,5),strides=(2,2), activation="relu"))
model.add(Convolution2D(48, (5,5),strides=(2,2), activation="relu"))
model.add(Convolution2D(64, (3,3), activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, (3,3), activation="relu"))
#model.add(MaxPooling2D())
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=5)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

model.save('./CarND-Behavioral-Cloning-P3/model.h5')
