import unittest
from deep_nilmtk.data.loader.pytorch import GeneralDataLoader

__unittest = True

from deep_nilmtk.utils import assertNumpyArraysEqual
import numpy as np
from deep_nilmtk.data.pre_process import normalize

class TestGenralDataLoader(unittest.TestCase):

    def test_get_sample(self):
        inputs = np.random.randint(0, 1500, 3600).reshape(-1,1)
        targets = np.random.randint(0, 1500, 3600*2).reshape(-1,2)

        dataloader = GeneralDataLoader(inputs, targets, seq_type="seq2seq", in_size=56, out_size=56)
        in_, out_ = dataloader.get_sample(29)
        self.assertEqual(in_.shape[0], out_.shape[0])
        self.assertEqual(out_.shape, (56, 2))
        padded_targets = dataloader.targets.numpy()
        assertNumpyArraysEqual(out_.numpy(), padded_targets[29:85])

        dataloader = GeneralDataLoader(
            inputs,
            targets,
            seq_type="seq2point",
            point_position="mid_position",
            in_size=56,
            out_size=1,
        )
        in_, out_ = dataloader.get_sample(29)
        padded_targets = dataloader.targets.numpy()
        target_value = np.ascontiguousarray(np.expand_dims(padded_targets[29 + (56//2)], axis=0), dtype=np.float32)
        output_value = np.ascontiguousarray(out_.detach().numpy(), dtype=np.float32)
        assertNumpyArraysEqual(output_value, target_value)
        # Get the target value and ensure it has the same shape as the output
        target_value = padded_targets[29 + (56//2)]
        # Add batch dimension to match output shape and ensure it's contiguous
        target_value = np.ascontiguousarray(np.expand_dims(target_value, axis=0), dtype=np.float32)
        # Convert output to numpy array and ensure shapes match
        output_value = np.ascontiguousarray(out_.detach().numpy(), dtype=np.float32)
        assertNumpyArraysEqual(output_value, target_value)



