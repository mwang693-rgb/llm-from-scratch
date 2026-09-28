import io
import torch
import pytest
from cs336_basics.Transformer_lm import Transformer_lm
from cs336_basics.MoE import SparseMoE
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.Tokenizer import Tokenizer


def model(**kwargs):
    return Transformer_lm(16, 2, 32, 32, 16, 2, 10000, **kwargs)


def test_weight_tying_identity_gradients_and_roundtrip():
    tied, untied = model(tie_embeddings=True), model()
    assert tied.Embedding.weight is tied.lm_head.weights
    assert sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters()) == 32 * 16
    x = torch.randint(0, 32, (2, 8))
    cross_entropy(tied(x), x).backward()
    assert torch.isfinite(tied.Embedding.weight.grad).all()
    buf = io.BytesIO()
    torch.save(tied.state_dict(), buf)
    buf.seek(0)
    restored = model(tie_embeddings=True)
    restored.load_state_dict(torch.load(buf, weights_only=True))
    assert restored.Embedding.weight is restored.lm_head.weights
    torch.testing.assert_close(restored(x), tied(x))


@pytest.mark.parametrize('experts', [0, 4])
def test_causality_and_training_gradients(experts):
    torch.manual_seed(21)
    m = model(num_experts=experts, tie_embeddings=True)
    x = torch.randint(0, 32, (2, 10))
    edited = x.clone()
    edited[:, 6:] = (edited[:, 6:] + 1) % 32
    y, aux = m(x, return_aux_loss=True)
    torch.testing.assert_close(y[:, :6], m(edited)[:, :6])
    loss = cross_entropy(y, (x+1) % 32) + 0.01*aux
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if experts:
        assert m.layer[0].FFNlayer.router.weights.grad.abs().sum() > 0
    assert m.generate(x, 3, temperature=0).shape == (2, 13)


def test_moe_matches_dense_reference_and_backward():
    torch.manual_seed(123)
    moe = SparseMoE(8, 16, 4, 2)
    x = torch.randn(2, 5, 8, requires_grad=True)
    result, aux = moe(x)
    probs = moe.router(x.reshape(-1, 8)).softmax(-1)
    values, indices = probs.topk(2, dim=-1)
    weights = torch.zeros_like(probs).scatter(-1, indices, values / values.sum(-1, keepdim=True))
    all_outputs = torch.stack([e(x.reshape(-1, 8)) for e in moe.experts], dim=1)
    reference = (all_outputs * weights[..., None]).sum(1).reshape_as(x)
    torch.testing.assert_close(result, reference)
    grad = torch.autograd.grad(result.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference.square().sum(), x)[0]
    torch.testing.assert_close(grad, expected_grad)
    assert torch.isfinite(aux)


def test_tokenizer_save_load_and_unknown_unicode(tmp_path):
    tok = Tokenizer({i: bytes([i]) for i in range(256)}, [], ['<eos>'])
    text = '模型🙂 café <eos>'
    path = tmp_path / 'tokenizer.json'
    tok.save(path)
    restored = Tokenizer.load(path)
    assert restored.decode(restored.encode(text)) == text


def test_invalid_moe_configuration():
    with pytest.raises(ValueError):
        SparseMoE(8, 16, 2, 3)
