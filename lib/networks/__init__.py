from lib.networks.discriminators import ResFNNDiscriminator, LSTMDiscriminator
from lib.networks.generators import LSTMGenerator, LogSigRNNGenerator


GENERATORS = {'LSTM': LSTMGenerator, 'LogSigRNN': LogSigRNNGenerator}


def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)


DISCRIMINATORS = {'ResFNN': ResFNNDiscriminator, 'LSTM': LSTMDiscriminator}


def get_discriminator(discriminator_type, input_dim, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, **kwargs)
