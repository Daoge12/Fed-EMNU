import argparse
import unittest
import numpy as np
import torch
from revision_runner import partition_labels, retained_partitions, aggregate_states, optimized_noise
from revision_evaluation import auc, calibrate_threshold, make_attack_splits, prediction_distance


class ProtocolTests(unittest.TestCase):
    def args(self, **kwargs):
        defaults = dict(num_clients=10, split_seed=42, iid=True, dirichlet_alpha=0.5,
                        request_scope="global_class", client_id=3, unlearn_class=[0, 2])
        return argparse.Namespace(**(defaults | kwargs))

    def test_every_class_allocated_once_including_cifar100(self):
        labels = np.repeat(np.arange(100), 30)
        for iid in (True, False):
            args = self.args(iid=iid)
            partitions = partition_labels(labels, args)
            self.assertEqual(sorted(i for part in partitions for i in part), list(range(len(labels))))
            self.assertEqual(partitions, partition_labels(labels, args))

    def test_request_scope_changes_exactly_the_requested_records(self):
        labels = np.tile(np.arange(10), 100)
        partitions = partition_labels(labels, self.args())
        for scope in ("requesting_client", "global_class"):
            args = self.args(request_scope=scope)
            retained = retained_partitions(partitions, labels, args)
            for cid, (old, new) in enumerate(zip(partitions, retained)):
                if scope == "global_class" or cid == args.client_id:
                    self.assertEqual(set(old) - set(new), {i for i in old if labels[i] in (0, 2)})
                    self.assertFalse(any(labels[i] in (0, 2) for i in new))
                else:
                    self.assertEqual(old, new)

    def test_weighted_aggregation_and_integer_buffers(self):
        states = [{"w": torch.tensor([0.]), "counter": torch.tensor(3)},
                  {"w": torch.tensor([4.]), "counter": torch.tensor(7)}]
        value = aggregate_states(states, [1, 3])
        self.assertAlmostEqual(value["w"].item(), 3.0)
        self.assertEqual(value["counter"].item(), 7)
        self.assertAlmostEqual(aggregate_states(states, [1, 3], "equal")["w"].item(), 2.0)

    def test_auc_ties_and_calibration_direction(self):
        self.assertEqual(auc(np.array([1, 1]), np.array([1, 1])), 0.5)
        self.assertEqual(auc(np.array([3, 4]), np.array([1, 2])), 1.0)
        sign, threshold = calibrate_threshold(np.array([-4, -3]), np.array([-2, -1]))
        self.assertEqual(sign, -1)
        self.assertTrue(np.all(sign * np.array([-4, -3]) >= threshold))

    def test_attack_calibration_test_disjoint_and_class_matched(self):
        train_y = np.repeat(np.arange(3), 100)
        test_y = np.repeat(np.arange(3), 20)
        splits = make_attack_splits(train_y, test_y, list(range(len(train_y))))
        for cls, split in splits.items():
            self.assertFalse(set(split["member_calibration"]) & set(split["member_test"]))
            self.assertFalse(set(split["nonmember_calibration"]) & set(split["nonmember_test"]))
            self.assertEqual(len(split["member_test"]), len(split["nonmember_test"]))
            for key, indices in split.items():
                labels = train_y if key.startswith("member_") else test_y
                self.assertTrue(np.all(labels[indices] == int(cls)))

    def test_identical_predictions_have_zero_distance(self):
        p = np.array([[.1, .9], [.6, .4]])
        self.assertEqual(prediction_distance(p, p), {"prediction_disagreement": 0.0, "total_variation": 0.0})

    def test_noise_optimization_preserves_model_parameters_and_bn(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 10))
        model.train()
        model[0].weight.requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        args = argparse.Namespace(batch_size=2, resolved_device="cpu", noise_lr=0.1,
                                  noise_epochs=1, noise_steps=2, noise_l2=0.1, mode="noise")
        noise = optimized_noise(model, 0, args)
        self.assertTrue(model.training)
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(tuple(noise.shape), (2, 3, 32, 32))
        self.assertFalse(noise.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
