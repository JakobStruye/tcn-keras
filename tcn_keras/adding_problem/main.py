"""
Adding problem

Input consists of n time steps with 2 features each
First feature is random value in [0,1]
Second feature is 1 in two time steps and 0 in all others
Value to predict is the sum of the two random values for which the second feature is 1
"""


from tcn_keras.weightnorm.weightnorm import AdamWithWeightnorm, data_based_init
from tcn_keras.tcn import build_tcn
from keras.layers import Dense, Reshape
from keras.models import Model
import numpy as np

def build_data(size, input_steps):
    """
    Builds the data for the adding problem
    :param size:        Total number of samples
    :param input_steps: Time steps per sample
    :return:        X and Y data
    """
    mask = np.zeros((size, input_steps, 1))
    values = np.random.rand(size, input_steps, 1)
    Y = np.zeros((size, 1))
    therange = range(input_steps)
    for i in range(size):
        shuffled = np.random.choice(therange, (2), replace=False)
        mask[i, shuffled[0], 0] = 1
        mask[i, shuffled[1], 0] = 1
        Y[i, 0] = values[i, shuffled[0], 0] + values[i, shuffled[1], 0]

    X = np.stack((mask, values), axis=2)
    X = np.reshape(X, (size, input_steps, 2))
    return X, Y


def build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss='mse'):
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout, input_steps=input_steps)
    # reshape the (input_steps, 2) to (input_steps,) and reduce to 1 value
    reshape = Reshape((-1,))(outs)
    linear = Dense(1)(reshape)

    model = Model(inputs=ins, outputs=linear)
    # TODO figure out if this applies weight norm as intended
    aww = AdamWithWeightnorm(lr=lr)
    model.compile(aww, loss=loss)
    data_based_init(model, train_X[:100])
    return model


if __name__ == '__main__':

    #Fixed for this problem
    input_dim = 2

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
    loss = 'mse'
    units = 30
    clip = -1.0 # Clipping not implemented (yet)


    train_X, train_Y = build_data(train_size,input_steps)
    test_X, test_Y = build_data(test_size, input_steps)

    model = build_model(input_dim, input_steps, kernel_size, levels, units, lr, dropout, loss)

    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y))
