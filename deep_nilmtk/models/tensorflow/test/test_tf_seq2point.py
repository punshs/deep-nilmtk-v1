import unittest
import  numpy as np

from deep_nilmtk.models.tensorflow.seq2point import Seq2Point
import tensorflow as tf

from deep_nilmtk.utils.test import assertNumpyArraysEqual

class TestSeq2Point(unittest.TestCase):

    def test_froward(self):
        N = 2
        input_batch = np.random.random(64*125*1).reshape(64,125,1)
        params = {
            'in_size': 125,
            'out_size': N,
            'model_name': 'seq2point',
            'version': 1,
            'feature_type': 'mains',
            'appliances': ['app1', 'app2'],
            'multi_appliance': True
        }
        model = Seq2Point(params)
        input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
        output = model(input_batch)
        self.assertEqual(output.shape, (input_batch.shape[0], N))

