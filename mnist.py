# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:14:17 2020

@author: ram
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfd


mn_df,mn_info = tfd.load(name='mnist',with_info=True,as_supervised=True)


mn_test,mn_train= mn_df['test'],mn_df['train']



num_validation_samples = 0.1*mn_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)


num_test_samples = mn_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)


def scale(image,label):
    image=tf.cast(image,tf.float32)
    image /=255.
    return image,label


scaled_train_and_validation_data = mn_train.map(scale)
scaled_test_data = mn_test.map(scale)


BUFFER_SIZE=10000 

shuffled_train_and_validation_data= scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data=shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)



BATCH_SIZE = 100

train_data=train_data.batch(BATCH_SIZE)
validation_data=validation_data.batch(num_validation_samples)
test_data=scaled_test_data.batch(num_test_samples)
validation_inputs,validation_targets=next(iter(validation_data))



#model

input_size=784
output_size=10
hidden_layer_size = 300

model=tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(output_size,activation='softmax')
                            
                            ])
#optimizer and loss function

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#training

NUM_EPOCHS=5

model.fit(train_data,epochs=NUM_EPOCHS,validation_data=(validation_inputs,validation_targets),validation_steps=10,verbose=2)



test_loss,test_accuracy=model.evaluate(test_data)

print('test_loss:{},test_accuracy:{}'.format(test_loss, test_accuracy*100))









