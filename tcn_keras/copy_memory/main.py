"""

"""


from tcn_keras.weightnorm.weightnorm import AdamWithWeightnorm, data_based_init
from tcn_keras.tcn import build_tcn
from keras.layers import Dense, Reshape
from keras.models import Model
import numpy as np

def build_data(T, mem_length, b_size):
    """
    Generate data for the copying memory task
    :param T: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: X and Y data
    """

    seq = np.random.randint(1, 9, size=(b_size, mem_length))
    zeros = np.zeros((b_size, T))
    marker = 9 * np.ones((b_size, mem_length + 1))
    placeholders = np.zeros((b_size, mem_length))

    X = np.concatenate((seq, zeros[:,:-1], marker), axis=1)
    Y = np.concatenate((placeholders, zeros, seq), axis=1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
    return X, Y



def build_model(input_dim, input_steps, kernel_size, levels, units, lr, clipnorm, dropout, loss='mse'):
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout, input_steps=input_steps)
    # reshape the (input_steps, 2) to (input_steps,) and reduce to 1 value
    #reshape = Reshape((-1,))(outs)
    linear = Dense(1)(outs)

    model = Model(inputs=ins, outputs=linear)
    # TODO figure out if this applies weight norm as intended
    aww = AdamWithWeightnorm(lr=lr, clipnorm=clipnorm)
    model.compile(aww, loss=loss)
    data_based_init(model, train_X[:100])
    return model


if __name__ == '__main__':


    #Modifies the problem
    train_size = 10000
    test_size = 1000
    mem_length = 20
    T = 1000
    input_dim = T + 2 * mem_length

    #Hyperparameters
    batch_size = 32
    dropout = 0.05
    epochs = 10
    kernel_size = 8
    levels = 8
    lr = 5e-4
    loss = 'mse'
    units = 10
    clip = 1.0 # Clipping not implemented (yet)


    train_X, train_Y = build_data(T,mem_length, train_size)
    test_X, test_Y = build_data(T ,mem_length , test_size)

    model = build_model(1, input_dim , kernel_size, levels, units, lr, clip, dropout, loss)

    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y))
