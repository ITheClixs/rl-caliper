# caliper

Measurement tools for the batch-size and step-size scaling of reinforcement learning
post-training of language models.

The premise: the quantities that determine how large a batch and how large a step an RLVR
run should take are estimable online, from statistics the trainer already computes. This
repository contains the estimators, an exactly-solvable testbed used to validate them, and
the experiment harness used to test their predictions.
