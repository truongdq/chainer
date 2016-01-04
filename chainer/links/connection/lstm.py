from chainer.functions.activation import lstm
from chainer import link
from chainer.links.connection import linear
from chainer import variable
import os


class LSTM(link.Chain):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
        c (chainer.Variable): Cell states of LSTM units.
        h (chainer.Variable): Output at the previous timestep.

    """
    def __init__(self, in_size, out_size, init_upward=None, init_lateral=None):
        if init_upward and init_lateral:
            assert in_size == init_upward.W.data.shape[1]
            assert out_size == init_upward.W.data.shape[0] / 4
            super(LSTM, self).__init__(upward=init_upward, lateral=init_lateral,
            )
        else:
            super(LSTM, self).__init__(
                upward=linear.Linear(in_size, 4 * out_size),
                lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
            )

        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        """Resets the internal state.

        It sets None to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def save(self, fname):
        self.upward.save(fname + '.upward')
        self.lateral.save(fname + '.lateral')

    @classmethod
    def load(cls, fname):
        upward = linear.Linear.load(fname + '.upward')
        lateral = linear.Linear.load(fname + '.lateral')
        out_size, in_size = upward.W.data.shape
        out_size /= 4
        lstm_layer = cls(in_size, out_size, init_upward=upward, init_lateral=lateral)
        return lstm_layer

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h


class LSTM_Emphasis_Decoder(link.Chain):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
        c (chainer.Variable): Cell states of LSTM units.
        h (chainer.Variable): Output at the previous timestep.

    """
    def __init__(self, in_size, out_size, use_attention=True, init_upward=None, init_lateral=None, init_attention=True):
        self.use_attention = use_attention
        if init_upward and init_lateral:
            assert in_size == init_upward.W.data.shape[1]
            assert out_size == init_upward.W.data.shape[0] / 4

            if init_attention:
                self.use_attention = True
                super(LSTM_Emphasis_Decoder, self).__init__(upward=init_upward, lateral=init_lateral, attention=init_attention)
            else:
                super(LSTM_Emphasis_Decoder, self).__init__(upward=init_upward, lateral=init_lateral)
        else:
            if not use_attention:
                super(LSTM_Emphasis_Decoder, self).__init__(
                    upward=linear.Linear(in_size, 4 * out_size),
                    lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
                )
            else:
                super(LSTM_Emphasis_Decoder, self).__init__(
                    upward=linear.Linear(in_size, 4 * out_size),
                    lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
                    attention=linear.Linear(in_size, 4 * out_size),
                )

        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        """Resets the internal state.

        It sets None to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def save(self, fname):
        self.upward.save(fname + '.upward')
        self.lateral.save(fname + '.lateral')
        if self.use_attention:
            self.attention.save(fname + '.attention')

    @classmethod
    def load(cls, fname):
        upward = linear.Linear.load(fname + '.upward')
        lateral = linear.Linear.load(fname + '.lateral')
        attention = None
        if os.path.exists(fname + '.attention'):
            attention = linear.Linear.load(fname + '.attention')

        out_size, in_size = upward.W.data.shape
        out_size /= 4
        lstm_layer = cls(in_size, out_size, init_upward=upward, init_lateral=lateral, init_attention=attention)
        return lstm_layer

    def __call__(self, x, src_hidden=None):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.
            emphasis (~chainer.Variable): A batch of previous estimated emphasis level.
            src_hidden (~chainer.Variable): A batch of corresponding source language LSTM output.
                                            Which is taken using word alignments.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)

        # Attention_in is the input vector of corresponding words from source language
        # which is derived by word alignments
        attention_in = None
        if self.use_attention:
            attention_in = self.attention(src_hidden)

        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        if attention_in is not None:
            self.c, self.h = lstm.lstm(self.c, lstm_in + attention_in)
        else:
            self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h
