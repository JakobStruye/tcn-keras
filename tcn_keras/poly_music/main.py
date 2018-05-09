"""

"""

from tcn_keras.weightnorm.weightnorm import AdamWithWeightnorm, data_based_init
from tcn_keras.tcn import build_tcn
from keras.layers import Dense, Reshape
from keras.models import Model
from scipy.io import loadmat
import numpy as np


def build_data(size, input_steps):
    """

    """
    X = np.random.rand(size,input_steps, 1)
    # Y's time steps are X's time steps right-shifted by one
    # The shift is cyclical, so the final value in a Y sample is the first in its corresponding X sample
    # and therefore predictable. As long as input_steps is high enough this shouldn't have a drastic effect
    # on overall accuracy
    Y = np.roll(X,-1, axis=1)

    return X, Y

    dataset = "JSB"

    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat('./mdata/MuseData.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./mdata/Nottingham.mat')
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat('./mdata/Piano_midi.mat')

    X_train = np.array(data['traindata'][0])
    X_valid = np.array(data['validdata'][0])
    X_test = np.array(data['testdata'][0])

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



def build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss='mse', clipnorm=-1.):
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout, input_steps=input_steps)
    # Reduce the `units` values per time step to 1
    linear = Dense(input_dim)(outs)

    model = Model(inputs=ins, outputs=linear)
    # TODO figure out if this applies weight norm as intended
    aww = AdamWithWeightnorm(lr=lr)
    model.compile(aww, loss=loss, clipnorm=clipnorm)
    data_based_init(model, train_X[:100])
    return model


if __name__ == '__main__':

    #Fixed for this problem
    input_dim = 88

    #Modifies the problem
    input_steps = 600
    train_size = 50000
    test_size = 1000

    #Hyperparameters
    batch_size = 32
    dropout = 0.5
    epochs = 10
    kernel_size = 3
    levels = 2
    lr = 4e-3
    loss = 'cross_entropy_loss'
    units = 150
    clip = 0.4


    train_X, train_Y, validate_X, validate_Y, test_X, test_Y = build_data(train_size,input_steps)
    #test_X, test_Y = build_data(test_size, input_steps)

    model = build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss, clipnorm)

    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(validate_X, validate_Y))
    model.predict(x=test_X, batch_size=batch_size )
