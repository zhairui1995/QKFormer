import torch

from lut_if.neuron import DenseLUTIFNeuron, QuantizedArithmeticLIF


def test_dense_lutif_shape_and_gradient():
    module = DenseLUTIFNeuron(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        x_range=(-2.0, 2.0),
        v_range=(-1.0, 1.0),
        state_bits=3,
        input_bits=3,
        hard_eval=False,
    )
    module.train()
    x = torch.randn(4, 2, 3, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    loss = y.mean() + module.regularization()["delta_l2"]
    loss.backward()
    assert module.transition_table.grad is not None
    assert torch.isfinite(module.transition_table.grad).all()
    module.reset()
    assert module.v == 0.0


def test_dense_lutif_zero_input_stays_quiet():
    module = DenseLUTIFNeuron(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        x_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
        state_bits=4,
        input_bits=4,
        hard_eval=False,
        learn_threshold=False,
    )
    module.eval()
    output = module(torch.zeros(4, 2, 3))
    assert torch.count_nonzero(output) == 0
    assert module.v_seq is not None
    assert module.v_seq.shape == output.shape


def test_quantized_arithmetic_shape_and_metrics():
    module = QuantizedArithmeticLIF(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        x_range=(-2.0, 2.0),
        v_range=(-1.0, 1.0),
        state_bits=4,
        input_bits=4,
    )
    module.eval()
    x = torch.tensor([[[0.0, 3.0]], [[0.0, 3.0]]])
    y = module(x)
    assert y.shape == x.shape
    summary = module.summary()
    assert summary["executions"] == 1.0
    assert summary["clip_rate"] > 0.0
