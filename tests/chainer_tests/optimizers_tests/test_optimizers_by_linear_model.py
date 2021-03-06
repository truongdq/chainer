import unittest

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class LinearModel(object):

    UNIT_NUM = 10
    BATCH_SIZE = 32
    EPOCH = 100

    def __init__(self, optimizer):
        self.model = L.Linear(self.UNIT_NUM, 2)
        self.optimizer = optimizer
        # true parameters
        self.w = np.random.uniform(-1, 1,
                                   (self.UNIT_NUM, 1)).astype(np.float32)
        self.b = np.random.uniform(-1, 1, (1, )).astype(np.float32)

    def _train_linear_classifier(self, model, optimizer, gpu):
        def _make_label(x):
            a = (np.dot(x, self.w) + self.b).reshape((self.BATCH_SIZE, ))
            t = np.empty_like(a).astype(np.int32)
            t[a >= 0] = 0
            t[a < 0] = 1
            return t

        def _make_dataset(batch_size, unit_num, gpu):
            x_data = np.random.uniform(
                -1, 1, (batch_size, unit_num)).astype(np.float32)
            t_data = _make_label(x_data)
            if gpu:
                x_data = cuda.to_gpu(x_data)
                t_data = cuda.to_gpu(t_data)
            x = chainer.Variable(x_data)
            t = chainer.Variable(t_data)
            return x, t

        for epoch in six.moves.range(self.EPOCH):
            x, t = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu)
            model.zerograds()
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        x_test, t_test = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu)
        y_test = model(x_test)
        return F.accuracy(y_test, t_test)

    def _accuracy_cpu(self):
        self.optimizer.setup(self.model)
        return self._train_linear_classifier(self.model, self.optimizer, False)

    def _accuracy_gpu(self):
        model = self.model
        optimizer = self.optimizer
        model.to_gpu()
        optimizer.setup(model)
        return self._train_linear_classifier(model, optimizer, True)

    def accuracy(self, gpu):
        if gpu:
            return cuda.to_cpu(self._accuracy_gpu().data)
        else:
            return self._accuracy_cpu().data


class OptimizerTestBase(object):

    def create(self):
        raise NotImplementedError()

    def setUp(self):
        self.model = LinearModel(self.create())

    @condition.retry(10)
    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy(False), 0.9)

    @attr.gpu
    @condition.retry(10)
    def test_linear_model_gpu(self):
        self.assertGreater(self.model.accuracy(True), 0.9)

    def test_initialize(self):
        model = self.model.model
        assert isinstance(model, chainer.Link)
        optimizer = self.create()
        optimizer.setup(model)

        msg = 'optimization target must be a link'
        with self.assertRaisesRegexp(TypeError, msg):
            optimizer.setup('xxx')


class TestAdaDelta(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.AdaDelta(eps=1e-5)


class TestAdaGrad(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.AdaGrad(0.1)


class TestAdam(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.Adam(0.1)


class TestMomentumSGD(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.MomentumSGD(0.1)


class NesterovAG(OptimizerTestBase, unittest.TestCase):
    def create(self):
        return optimizers.NesterovAG(0.1)


class TestRMSprop(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.RMSprop(0.1)


class TestRMSpropGraves(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.RMSpropGraves(0.1)


class TestSGD(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.SGD(0.1)


testing.run_module(__name__, __file__)
