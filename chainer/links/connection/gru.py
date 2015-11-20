from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear


class GRU(link.Chain):

    """Gated Recurrent Unit function (GRU).

    GRU function has six parameters :math:`W_r`, :math:`W_z`, :math:`W`,
    :math:`U_r`, :math:`U_z`, and :math:`U`. All these parameters are
    :math:`n \\times n` matricies, where :math:`n` is the dimension of
    hidden vectors.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, GRU returns the next hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the
    element-wise product.

    Args:
        n_units(int): Dimension of input vector :math:`x`, and hidden vector
            :math:`h`.

    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <http://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <http://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].

    """

    def __init__(self, n_units, init_param=None):
        if init_param is None:
            super(GRU, self).__init__(
                W_r=linear.Linear(n_units, n_units),
                U_r=linear.Linear(n_units, n_units),
                W_z=linear.Linear(n_units, n_units),
                U_z=linear.Linear(n_units, n_units),
                W=linear.Linear(n_units, n_units),
                U=linear.Linear(n_units, n_units),
            )
        else:
            super(GRU, self).__init__(
                W_r=init_param[0],
                U_r=init_param[1],
                W_z=init_param[2],
                U_z=init_param[3],
                W=init_param[4],
                U=init_param[5],
            )
            
    
    
    def save(self, fname):
        self.W_r.save(fname + '.W_r')
        self.U_r.save(fname + '.U_r')
        self.W_z.save(fname + '.W_z')
        self.U_z.save(fname + '.U_z')
        self.W.save(fname + '.W')
        self.U.save(fname + '.U')
    
    @classmethod
    def load(cls, fname):
        W_r = linear.Linear.load(fname + '.W_r')
        U_r = linear.Linear.load(fname + '.U_r')
        W_z = linear.Linear.load(fname + '.W_z')
        U_z = linear.Linear.load(fname + '.U_z')
        W = linear.Linear.load(fname + '.W')
        U = linear.Linear.load(fname + '.U')
        return cls(1, init_param=(W_r, U_r, W_z, U_z, W, U))

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new
