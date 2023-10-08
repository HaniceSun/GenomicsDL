import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Dropout, Activation, Add, Cropping1D
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
import tensorflow.keras.backend as K

import sys

class ResidualBlock(Layer):
    def __init__(self, N, W, D):
        super().__init__()
        self.N = N
        self.W = W
        self.D = D
        self.BatchNormalization1 = BatchNormalization()
        self.BatchNormalization2 = BatchNormalization()
        self.Activation1 = Activation('relu')
        self.Activation2 = Activation('relu')
        self.Conv1D1 = Conv1D(self.N, self.W, dilation_rate=self.D, padding='same')
        self.Conv1D2 = Conv1D(self.N, self.W, dilation_rate=self.D, padding='same')
        self.Add = Add()
    def call(self, inputs):
        x = self.BatchNormalization1(inputs)
        x = self.Activation1(x)
        x = self.Conv1D1(x)
        x = self.BatchNormalization2(x)
        x = self.Activation2(x)
        x = self.Conv1D2(x)
        x = self.Add([x, inputs])
        return(x)
    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
            "W": self.W,
            "D": self.D,
        })
        return config

def checkpoint_callback(inF):
    ouF = inF + '/checkpoint-{epoch:03d}.ckpt'
    return(tf.keras.callbacks.ModelCheckpoint(filepath=ouF, save_weights_only=True, verbose=1))
def loss_callback(inF):
    ouF = inF + '/losslog.txt'
    return(tf.keras.callbacks.CSVLogger(filename=ouF, separator='\t'))
def tensorboard_callback(inF):
    ouF = inF + '/tensorboard'
    return(tf.keras.callbacks.TensorBoard(log_dir=ouF))

def learning_rate_decay_callback(epoch, lr):
    if epoch > 5:
        lr = lr * 0.5
    return(lr)

def LossFunction(y_true, y_pred, epsilon=1e-10):
    return(-K.mean(y_true[:, :, 0]*K.log(y_pred[:, :, 0] + epsilon) 
        + y_true[:, :, 1]*K.log(y_pred[:, :, 1] + epsilon) 
        + y_true[:, :, 2]*K.log(y_pred[:, :, 2] + epsilon)))

def getModel(NWD=[[32, 11, 1]] * 4, nFilters=32, FlankSize=40, nResidualBlocks=4, input_shape=[None, 4], learning_rate=0.001):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(nFilters, 1, dilation_rate=1)(inputs)
    skip = Conv1D(nFilters, 1, dilation_rate=1)(x)
    for i in range(len(NWD)):
        n,w,d = NWD[i]
        x = ResidualBlock(n, w, d)(x)
        if (i+1)%nResidualBlocks == 0:
            cv = Conv1D(nFilters, 1, dilation_rate=1)(x)
            skip = Add()([cv, skip])
    skip = Cropping1D(FlankSize)(skip)
    ### extra BatchNorm for tensorflow 2
    skip = BatchNormalization()(skip)
    outputs = Conv1D(3, 1, dilation_rate=1, activation='softmax')(skip)
    model = Model(inputs, outputs)
    model.compile(loss=LossFunction, optimizer=tf.keras.optimizers.Adam(learning_rate))
    print(model.summary())
    return(model)

def fitModel(inF='canonical_dataset_seq_nt5000_flank40.tfds', ouF='MethyAI_nt5000_flank40_SavedModel', NWD = [[32, 11, 1]] * 4, nFilters=32, nResidualBlocks=4, FlankSize=None, input_shape=[None, 4], split_method='chr', learning_rate=0.001, nEpochs=10, batchSize=32):
    if not FlankSize:
        FlankSize = int(inF.split('.tfds')[0].split('flank')[-1])

    inF1 = inF.split('.tfds')[0] + '_SM%s-train.tfds'%(split_method)
    inF2 = inF.split('.tfds')[0] + '_SM%s-val.tfds'%(split_method)
    inF3 = inF.split('.tfds')[0] + '_SM%s-test.tfds'%(split_method)

    dataset_train = tf.data.Dataset.load(inF1, compression='GZIP')
    dataset_val = tf.data.Dataset.load(inF2, compression='GZIP')
    dataset_test = tf.data.Dataset.load(inF3, compression='GZIP')
    
    dataset_train = dataset_train.batch(batchSize, drop_remainder=False)
    dataset_train = dataset_train.prefetch(8)
    dataset_val = dataset_val.batch(batchSize, drop_remainder=False)
    dataset_val = dataset_val.prefetch(8)
    dataset_test = dataset_test.batch(batchSize, drop_remainder=False)
    dataset_test = dataset_test.prefetch(8)
    
    model = getModel(NWD, nFilters, FlankSize, nResidualBlocks, learning_rate=learning_rate)

    history = model.fit(dataset_train, epochs=nEpochs, validation_data=dataset_val, callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_decay_callback, verbose=1), checkpoint_callback(ouF), loss_callback(ouF), tensorboard_callback(ouF)])
    model.save(ouF)

    df = pd.DataFrame(history.history)
    df.to_csv(ouF + '_history.txt', header=True, index=False, sep='\t')



if __name__ == '__main__':

    inF = sys.argv[1]
    nResidualBlocks = int(sys.argv[3].split('RB')[-1])
    nFilters = int(sys.argv[4].split('NF')[-1])
    split_method = sys.argv[5]
    learning_rate = float(sys.argv[6].split('LR')[-1])
    loss_function = sys.argv[7]
    nEpochs = int(sys.argv[8])


    if sys.argv[2] == 'NWD1':
        NWD = [[nFilters, 11, 1]] * nResidualBlocks
    elif sys.argv[2] == 'NWD2':
        NWD = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks
    elif sys.argv[2] == 'NWD3':
        NWD = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks + [[nFilters, 21, 10]] * nResidualBlocks
    elif sys.argv[2] == 'NWD4':
        NWD = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks + [[nFilters, 21, 10]] * nResidualBlocks + [[nFilters, 41, 25]] * nResidualBlocks

    ouF = '_'.join([inF.split('.tfds')[0], sys.argv[2], 'NRB%s_NF%s_SM%s_LR%s_LF%s_NE%s_SavedModel'%(nResidualBlocks, nFilters, split_method, learning_rate, loss_function, nEpochs)])

    fitModel(inF, ouF, NWD, nFilters, nResidualBlocks, split_method=split_method, learning_rate=learning_rate, nEpochs=nEpochs)

