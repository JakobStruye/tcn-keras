from keras.layers import Conv1D
from keras.layers import Dropout, Lambda, Add, Input, Dense, Activation, Reshape
from keras.models import Model
from weightnorm import AdamWithWeightnorm, data_based_init
import numpy as np


def add_temporal_block(inputs, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    conv1 = Conv1D(n_outputs, kernel_size, strides=stride, dilation_rate=dilation, padding='causal' )(inputs)
    #weightnorm
    #chomp?
    relu1= Activation('relu')(conv1)
    dropout1 = Dropout(dropout)(relu1)
    conv2 = Conv1D(n_outputs, kernel_size, strides=stride, dilation_rate=dilation, padding='causal' )(dropout1)
    #chomp?
    relu2 = Activation('relu')(conv2)
    dropout2 = Dropout(dropout)(relu2)


    residual = Conv1D(n_outputs,1)(inputs) if n_inputs != n_outputs else (Lambda(lambda x: x))(inputs)
    add = Add()([dropout2, residual])
    final = Activation('relu')(add)
    return final


def build_tcn(num_inputs, num_channels, kernel_size=2, dropout=2, seq_len=400) :
    layers = []
    num_levels = len(num_channels)
    inputs = Input(shape=(seq_len,1))
    current_latest = inputs
    for i in range(num_levels):
        dilation_size = 2 ** i
        in_channels = num_inputs if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
        current_latest = add_temporal_block(current_latest, in_channels, out_channels, kernel_size, 1, dilation_size, (kernel_size-1)*dilation_size, dropout)

    return (inputs, current_latest)

def build_data(size, seq_len):
    mask = np.zeros((size, seq_len, 1))
    values = np.random.rand(size, seq_len, 1)
    Y = np.zeros((size, 1))
    therange = range(seq_len)
    for i in range(size):
        shuffled = np.random.choice(therange, (2), replace=False)
        # tf.scatter_update(mask, [i,1,shuffled[0]], tf.reshape([1.0], (1,1,1)))
        mask[i, shuffled[0], 0] = 1
        mask[i, shuffled[1], 0] = 1
        Y[i, 0] = values[i, shuffled[0], 0] + values[i, shuffled[1], 0]

    X = np.stack((mask, values), axis=2)
    X = np.reshape(X, (size, seq_len, 2))
    return X, Y

def build_sanity_check_data(size, seq_len):
    X = np.random.rand(size,seq_len, 1)
    Y = np.roll(X,-1, axis=1)
    #Y = np.reshape(Y, (size,seq_len))
    return X, Y


if __name__ == '__main__':
    batch_size = 32

    dropout = 0.0
    clip = -1.0 #noclip
    epochs = 10
    kernel_size = 7
    levels = 8
    seq_len = 600
    lr = 4e-3
    units = 30

    input_channels = 2
    n_classes = 1

    #seq_len = 400
    train_X, train_Y = build_sanity_check_data(50000,seq_len)
    test_X, test_Y = build_sanity_check_data(1000, seq_len)


    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [units] * levels
    (ins, outs) = build_tcn(input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout, seq_len=seq_len)
    #reshape = Reshape((-1,))(outs)
    linear = Dense(n_classes)(outs)
    #final = linear
    final = linear
    model = Model(inputs=ins, outputs=final)
    aww = AdamWithWeightnorm()
    model.compile(aww, loss='mae')
    data_based_init(model, train_X[:100])
    print(train_Y.shape)
    print(model.summary())
    model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y))



# def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#     super(TemporalBlock, self).__init__()
#     self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                        stride=stride, padding=padding, dilation=dilation))
#     self.chomp1 = Chomp1d(padding)
#     self.relu1 = nn.ReLU()
#     self.dropout1 = nn.Dropout(dropout)
#
#     self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                        stride=stride, padding=padding, dilation=dilation))
#     self.chomp2 = Chomp1d(padding)
#     self.relu2 = nn.ReLU()
#     self.dropout2 = nn.Dropout(dropout)
#
#     self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                              self.conv2, self.chomp2, self.relu2, self.dropout2)
#     self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#     self.relu = nn.ReLU()
#     self.init_weights()
#
# def init_weights(self):
#     self.conv1.weight.data.normal_(0, 0.01)
#     self.conv2.weight.data.normal_(0, 0.01)
#     if self.downsample is not None:
#         self.downsample.weight.data.normal_(0, 0.01)
#
# def forward(self, x):
#     out = self.net(x)
#     res = x if self.downsample is None else self.downsample(x)
#     return self.relu(out + res)
