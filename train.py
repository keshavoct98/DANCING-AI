import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]) # limits gpu memory usage

def preprocess(X, Y):
    ''' Scaling and Train-test split of data.'''
    
    test_data = X.shape[0] - int(X.shape[0]/20) # 5% data is used for predictions
    X_train = X[:test_data, :]
    Y_train = Y[:test_data, :]
    X_test = X[test_data:, :]
    Y_test = Y[test_data:, :]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, Y_train, Y_test, scaler


def train(X, Y):
    ''' Splits data into train-test, trains lstm on training split
    and return predictions on test data.'''
    
    X_train, X_test, Y_train, Y_test, scaler = preprocess(X, Y)
    
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(units = 72, input_shape = (36,1), return_sequences = True))
    model.add(LSTM(54, activation = 'tanh'))
    model.add(Dense(28))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(X_train, Y_train, batch_size = 16, epochs = 400)
    
    return scaler.inverse_transform(model.predict(X_test))