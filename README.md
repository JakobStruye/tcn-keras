# Temporal Convolutional Network implementation based on Keras

Closely follows the [reference Torch implementation](https://github.com/locuslab/TCN), accompanying the work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Currently only the adding problem from the original experiments is implemented

Furthermore there is a 'Sanity Check' experiment, which tries to predict the next value in a random sequence in [0,1)
This experiment confirms that the network does not leak future information and converges to always predicting 0.5.


To install, run
`
pip install .
`
To run an experiment, run
`
python main.py
`
in the experiment's subdirectory.
(There are probably better ways to accomplish this)


Currently only tested with Keras 2.0.5; >2.1.0 is known to be incompatible due to an API change not yet incorporated into the [weightnorm code by OpenAI](https://github.com/openai/weightnorm) we use.


