import unittest
import numpy as np
import torch

from deep_nilmtk.models.pytorch.seq2point import Seq2Point
import tensorflow as tf

from deep_nilmtk.utils.test import assertNumpyArraysEqual

class TestSeq2Point(unittest.TestCase):

    def test_froward(self):
        N = 2
        input_batch = np.random.random(64*125*1).reshape(64,125,1)
        params = {
            'in_size': 125,
            'out_size': N,
            'feature_type': 'mains',
            'appliances': ['app1', 'app2']
        }
        model = Seq2Point(params)
        output = model(torch.tensor(input_batch, dtype=torch.float32))
        self.assertEqual(output.shape, (input_batch.shape[0], N))

