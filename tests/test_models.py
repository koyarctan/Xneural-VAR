import pytest

torch = pytest.importorskip("torch")

from xneural_var.models import GVARWithNGCGates


def test_vectorized_gvar_forward_shapes_and_gradients():
    model = GVARWithNGCGates(
        num_vars=3,
        order=2,
        hidden_layer_size=5,
        num_hidden_layers=2,
    )
    inputs = torch.randn(4, 2, 3)

    preds, coeffs = model(inputs)
    loss = preds.pow(2).mean() + coeffs.pow(2).mean()
    loss.backward()

    assert preds.shape == (4, 3)
    assert coeffs.shape == (4, 2, 3, 3)
    assert model.causal_gate.grad is not None
