"""
Sanity check test

Input consists of n random values in [0,1]
Value to predict is the next value in the sequence

With no knowledge of future data points the optimal solution is to always predict 0.5,
leading to a MAE of 0.25 (1/4) and MSE of 0.0833 (1/12)

Any better score indicates future data points are leaking in the network.
Changing the Temporal Block's second Convolutional layer's padding to `same`
should indicate this.
"""

from tcn_keras.weightnorm.weightnorm import AdamWithWeightnorm, data_based_init
from tcn_keras.tcn import build_tcn
from keras.layers import Dense, Reshape
from keras.models import Model
import numpy as np


def build_data(size, input_steps):
    """
    Builds the data for a sanity check (predicting next value in a random sequence)
    :param size:        Total number of samples
    :param input_steps: Time steps per sample
    :return:
    """
    X = np.random.rand(size,input_steps, 1)
    # Y's time steps are X's time steps right-shifted by one
    # The shift is cyclical, so the final value in a Y sample is the first in its corresponding X sample
    # and therefore predictable. As long as input_steps is high enough this shouldn't have a drastic effect
    # on overall accuracy
    Y = np.roll(X,-1, axis=1)

    return X, Y

def build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss='mse'):
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout, input_steps=input_steps)
    # Reduce the `units` values per time step to 1
    linear = Dense(1)(outs)

    model = Model(inputs=ins, outputs=linear)
    # TODO figure out if this applies weight norm as intended
    aww = AdamWithWeightnorm(lr=lr)
    model.compile(aww, loss=loss)
    data_based_init(model, train_X[:100])
    return model


if __name__ == '__main__':

    #Fixed for this problem
    input_dim = 1

    #Modifies the problem
    input_steps = 600
    train_size = 50000
    test_size = 1000

    #Hyperparameters
    batch_size = 32
    dropout = 0.0
    epochs = 10
    kernel_size = 7
    levels = 8
    lr = 4e-3
    loss = 'mae'
    units = 30
    clip = -1.0 # Clipping not implemented (yet)


    train_X, train_Y = build_data(train_size,input_steps)
    test_X, test_Y = build_data(test_size, input_steps)

    model = build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss)

    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y))
