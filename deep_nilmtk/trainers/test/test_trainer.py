import unittest

from deep_nilmtk.trainers import Trainer, TorchTrainer

class TestTrainer(unittest.TestCase):

    def test_init(self):
        trainerImp=TorchTrainer({
            'test': 'Hello World'
        })
        hparams = {
            'model_class': None,
            'backend': 'pytorch',
            'model_name': 'Seq2Pointbaseline',
            'loader_class': None,
            'test': 'Hello World'
        }
        trainer = Trainer(trainerImp, hparams)
        self.assertEqual(trainer.test(),  'Hello World')


