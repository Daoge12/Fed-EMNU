"""Ensure request-weight ablation cannot silently change checkpoint training policy."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from revision_runner import unlearn

class EventPolicyTests(unittest.TestCase):
    def test_override_is_event_only_and_default_preserves_sample_weighting(self):
        train=TensorDataset(torch.zeros(6,1),torch.tensor([0,1,0,1,1,1]))
        partitions=[[0,1],[2,3,4,5]]
        retained=[[1],[3,4,5]]
        def local_update(model,data,indices,*unused,**kwargs):
            with torch.no_grad():model.weight.fill_(0 if len(indices)==1 else 4)
            return 1
        for override,expected in [(None,3.),('samples',3.),('equal',2.)]:
            args=argparse.Namespace(request_scope='global_class',num_clients=2,
                client_id=0,unlearn_class=[0],mode='ft',ablation='full',
                unlearn_seed=42,ft_epochs=1,ft_lr=.01,aggregation='samples',
                event_aggregation=override)
            model=torch.nn.Linear(1,1,bias=False)
            with tempfile.TemporaryDirectory() as folder:
                with patch('revision_runner.train_epochs',side_effect=local_update):
                    _,steps=unlearn(model,train,partitions,retained,args,Path(folder))
            self.assertEqual(steps,2)
            self.assertAlmostEqual(model.weight.item(),expected)
            self.assertEqual(args.aggregation,'samples')

if __name__=='__main__':unittest.main(verbosity=2)
