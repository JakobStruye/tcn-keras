from keras.layers import Conv1D
from keras.layers import Dropout, Lambda, Add, Input, Activation

def add_temporal_block(prev_layer, input_dim, output_dim, kernel_size, stride, dilation, dropout):
    """
    Appends a Temporal Block to the given layer
    :param prev_layer:  Layer to append TB to
    :param input_dim:    Input dimension
    :param output_dim:   Output dimension
    :param kernel_size: Convolutional kernel size
    :param stride:      Convolutional stride
    :param dilation:    Convolutional dilation
    :param dropout:     Dropout rate applied after each convolution
    :return: Final layer of the TB
    """

    # Main path
    # Casual padding applies `(kernel_size - 1) * dilation` left padding. No right padding so no need to chomp
    x = Conv1D(output_dim, kernel_size, strides=stride, dilation_rate=dilation, padding='causal')(prev_layer)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Conv1D(output_dim, kernel_size, strides=stride, dilation_rate=dilation, padding='causal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)


    # Residual path
    # Lambda layer is basically a no-op layer here
    residual = Conv1D(output_dim,1)(prev_layer) if input_dim != output_dim else (Lambda(lambda x: x))(prev_layer)


    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x



def build_tcn(input_dim, num_channels, kernel_size=2, dropout=2, input_steps=400) :
    """
    Builds all layers of a Temporal Convolutional Network.
    :param input_dim:   Input dimension
    :param num_channels: Number of channels (filters) in each Convolutional layer
    :param kernel_size:  Convolutional kernel size
    :param dropout:      Dropout rate applied after each convolution
    :param input_steps:  Number of time steps per input. Ignored if 1, removing 1 dimension from the input
    :return: Input and output layers of the TCN
    """
    layers = []
    num_levels = len(num_channels)
    input_shape = (input_steps, input_dim) if input_steps is not None and input_steps > 1 else (1,input_dim)
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_levels):
        dilation_size = 2 ** i
        in_channels = input_dim if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
        x = add_temporal_block(x, in_channels, out_channels, kernel_size, 1, dilation_size, dropout)

    return (inputs, x)

