"""

"""

from tcn_keras.weightnorm.weightnorm import AdamWithWeightnorm, data_based_init
from tcn_keras.tcn import build_tcn
from keras.layers import Dense, Reshape
from keras.models import Model
from scipy.io import loadmat
import numpy as np


def build_data():
    """

    """
    dataset = "JSB"

    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./data/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat('./data/MuseData.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./data/Nottingham.mat')
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat('./data/Piano_midi.mat')

    print("DIM", data['traindata'][0][100].shape)
    
    X_train = np.stack(data['traindata'][0], axis=0)
    X_valid = np.vstack(data['validdata'][0])
    X_test = np.vstack(data['testdata'][0])
    print("DIM", X_train.shape)
    exit(1)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    print("DIM", X_train.shape)

    Y_train = X_train[1:]
    X_train = X_train[:-1]
    Y_valid = X_valid[1:]
    X_valid = X_valid[:-1]
    Y_test = X_test[1:]
    X_test = X_test[:-1]


    # for data in [X_train, X_valid, X_test]:
    #     data = np.array(data)
    #     #for i in range(len(data)):
    #     #    data[i] = #torch.Tensor(data[i].astype(np.float64))

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test



def build_model(input_dim, kernel_size, levels, units, lr, dropout, input_steps=1, loss='mse', clipnorm=-1.):
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout, input_steps=input_steps)
    # Reduce the `units` values per time step to 1
    linear = Dense(input_dim)(outs)

    model = Model(inputs=ins, outputs=linear)
    # TODO figure out if this applies weight norm as intended
    aww = AdamWithWeightnorm(lr=lr, clipnorm=clipnorm)
    model.compile(aww, loss=loss)
    data_based_init(model, train_X[:100])
    return model


if __name__ == '__main__':

    #Fixed for this problem
    input_dim = 88


    #Hyperparameters
    batch_size = 32
    dropout = 0.5
    epochs = 10
    kernel_size = 3
    levels = 2
    lr = 4e-3
    loss = 'categorical_crossentropy'
    units = 150
    clipnorm = 0.4


    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = build_data()
    #test_X, test_Y = build_data(test_size, input_steps)

    model = build_model(input_dim, kernel_size, levels, units, lr, dropout, input_steps=1, loss=loss, clipnorm=clipnorm)

    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(validate_X, validate_Y))
    model.predict(x=test_X, batch_size=batch_size )
