import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import sklearn
import pprint

import build_data, build_model, make_features_labels, plot_diagnostics, compute_metrics, save_load_model

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import importlib as imp
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import f1_score, recall_score
from keras import backend as K

### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)

### Here is where we actually build the model

def compile_model():
  # This isn't good practice, but model will be a global variable
    global model

  # First we start with an input layer
    input_layer = Input(shape=x_train[0,:].shape) # size per sample, equal to number of features

  # Let's apply dropout to the input layer. (If Dropout(0), Dropout isn't being used)
    layers = Dropout(0)(input_layer)

  # Now let's add a few layers. We are going to do two layers with 5 nodes each.
  # Each uses the 'relu' activation function and randomizes the initial weights/biases
  # Dropout may be useful for hidden layers if the number of hidden nodes is large
  # If Dropout(0), Dropout isn't being used.

    for hidden, activation in zip(HIDDENS, ACTIVATIONS):
        layers = Dense(hidden, activation=activation, bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))(layers)
    
        layers = Dropout(0)(layers)

  # Output layer has a softmax function to convert output to class likelihood
    output_layer = Dense(num_classes-1, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED), kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))(layers)

  # Constructing the model
    model = Model(input_layer, output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) # Using the Adam optimizer
    model.compile(optimizer=optimizer, loss=LOSS,metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Accuracy(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives(),
                                                          tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(), tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
                 )
  
    model.summary()


### This is the function that will iteratively train the model

def fit_model():
    global history
    
    history = model.fit(x_train, y_train, 
                        epochs=NUM_EPOCHS,verbose=VERBOSITY,batch_size = BATCH_SIZE, shuffle=True,
                        validation_data=[x_val, y_val],
                        class_weight = CLASS_WEIGHT, callbacks=[EARLY_STOPPING]
                       )
    
def train_model():
    compile_model()
    fit_model()