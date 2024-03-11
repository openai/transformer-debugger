from copy import deepcopy
from functools import cache

import pytest
import torch

from neuron_explainer.models import Autoencoder, Transformer
from neuron_explainer.models.hooks import AtLayers, AutoencoderHooks, Hooks, TransformerHooks


def unflatten(f):
    def _f(x):
        return f(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], -1)

    return _f


@cache
def get_test_model():
    return Transformer.load("gpt2/small")


def test_forward_backward_hooks():
    """Test hooks with a simple cache saving function"""

    # create transformer
    model = get_test_model()
    cfg = model.cfg

    # create a save function to hook to the autoencoder
    forward_cache, backward_cache = {}, {}

    def store_forward(xx, layer, **kwargs):
        forward_cache[layer] = xx.detach().clone()
        return xx

    def store_backward(xx, layer, **kwargs):
        backward_cache[layer] = xx.detach().clone()
        return xx

    # create hooks
    hooks = TransformerHooks()
    hooks.mlp.post_act.append_fwd(store_forward)
    hooks.mlp.post_act.append_bwd(store_backward)

    # run the model
    input_tokens = torch.arange(5, device=model.device)[None, :]
    model.zero_grad()
    X, _ = model.forward(input_tokens, hooks=hooks)
    X.sum().backward()
    n_neurons = model.xf_layers[0].mlp.out_layer.weight.shape[1]

    # check that the forward were stored for all layers
    assert list(forward_cache.keys()) == list(range(cfg.n_layers))
    assert forward_cache[0].shape == (*input_tokens.shape, n_neurons)

    # check that the gradient were stored for all layers
    assert list(backward_cache.keys()) == list(range(cfg.n_layers))[::-1]
    assert backward_cache[0].shape == (*input_tokens.shape, n_neurons)


def test_autoencoder_hooks():
    """Test autoencoder hook with a simple cache saving function"""
    # create transformer and autoencoder
    n_latents = 10
    model = get_test_model()
    cfg = model.cfg
    autoencoder = Autoencoder(n_latents=n_latents, n_inputs=cfg.d_ff).to(model.device)

    # create a save function to hook to the autoencoder
    cache = {}

    def store_latents(latents, layer, **kwargs):
        cache[layer] = latents.detach().clone()
        return latents

    # create hooks
    ae_hooks = AutoencoderHooks(
        encode=unflatten(autoencoder.encode), decode=unflatten(autoencoder.decode)
    )
    ae_hooks.latents.append_fwd(store_latents)
    hooks = TransformerHooks()
    hooks.mlp.post_act.append_fwd(ae_hooks)

    # run the model
    input_tokens = torch.arange(5, device=model.device)[None, :]
    model.forward(input_tokens, hooks=hooks)

    # check that the latents were stored for all layers
    assert list(cache.keys()) == list(range(cfg.n_layers))
    assert cache[0].shape == (*input_tokens.shape, n_latents)

    # do the same, but only on a single layer
    layer_idx = 4
    cache = {}
    ae_hooks = AutoencoderHooks(
        encode=unflatten(autoencoder.encode), decode=unflatten(autoencoder.decode)
    )
    ae_hooks.latents.append_fwd(store_latents)
    hooks = TransformerHooks()
    hooks.mlp.post_act.append_fwd(AtLayers(layer_idx).append(ae_hooks))
    model.forward(input_tokens, hooks=hooks)
    assert list(cache.keys()) == [layer_idx]
    assert cache[layer_idx].shape == (*input_tokens.shape, n_latents)

    # do the same, but on a backward pass
    ae_hooks = AutoencoderHooks(
        encode=unflatten(autoencoder.encode), decode=unflatten(autoencoder.decode)
    )
    ae_hooks.latents.append_fwd(store_latents)
    hooks = TransformerHooks()
    hooks.mlp.post_act.append_bwd(AtLayers(layer_idx).append(ae_hooks))
    model.zero_grad()
    X, _ = model.forward(input_tokens, hooks=hooks)
    X.sum().backward()
    assert list(cache.keys()) == [layer_idx]
    assert cache[layer_idx].shape == (*input_tokens.shape, n_latents)


@pytest.mark.parametrize("add_error", [True, False])
def test_autoencoder_backward_hooks(add_error):
    """Test autoencoder hook, running a backward pass *from* a latent"""
    layer_idx = 4
    n_latents = 10
    model = get_test_model()
    cfg = model.cfg
    autoencoder = Autoencoder(n_latents=n_latents, n_inputs=cfg.d_ff).to(model.device)
    input_tokens = torch.arange(5, device=model.device)[None, :]

    cache = {}

    def store_latents_no_detach(latents, layer, **kwargs):
        cache[layer] = latents
        return latents

    ae_hooks = AutoencoderHooks(
        encode=unflatten(autoencoder.encode),
        decode=unflatten(autoencoder.decode),
        add_error=add_error,
    )
    ae_hooks.latents.append_fwd(store_latents_no_detach)
    hooks = TransformerHooks()
    hooks.mlp.post_act.append_fwd(AtLayers(layer_idx).append(ae_hooks))
    model.zero_grad()
    X, _ = model.forward(input_tokens, hooks=hooks)

    assert model.xf_layers[0].mlp.in_layer.weight.grad is None
    latents = cache[layer_idx]
    latents[:, 0].sum().backward()  # backward from the first latent
    assert model.xf_layers[0].mlp.in_layer.weight.grad is not None


@pytest.mark.parametrize("add_error", [True, False])
def test_autoencoder_hooks_ablation(add_error):
    """To test ablation, create an autoencoder with large weights at one latent, and ablate it."""
    # create transformer and autoencoder
    layer_idx = 4
    latent_idx = 1
    n_latents = 10
    model = get_test_model()
    cfg = model.cfg
    autoencoder = Autoencoder(n_latents=n_latents, n_inputs=cfg.d_ff).to(model.device)
    autoencoder.latent_bias.data[latent_idx] = +1000  # make sure the latent activates strongly
    input_tokens = torch.arange(5, device=model.device)[None, :]

    # create an ablation hook
    def ablate_latent(latents, layer, **kwargs):
        latents[:, :, latent_idx] = 0
        return latents

    # create a cache hook, to check downstream of the autoencoder hook
    def store_mlp_post_act(x, layer, **kwargs):
        cache[layer] = x.detach().clone().max().item()
        return x

    for ablate in [False, True]:
        cache = {}
        ae_hooks = AutoencoderHooks(
            encode=unflatten(autoencoder.encode),
            decode=unflatten(autoencoder.decode),
            add_error=add_error,
        )
        if ablate:
            ae_hooks.latents.append_fwd(ablate_latent)
        hooks = TransformerHooks()
        hooks.mlp.post_act.append_all(
            AtLayers(layer_idx).append(ae_hooks).append(store_mlp_post_act)
        )

        output, _ = model.forward(input_tokens, hooks=hooks)

        # if not add_error, the MLP post_act is large with no ablation
        # (because the latent activates strongly), and small with ablation
        # if add_error, the MLP post_act is small with no ablation
        # (because the autoencoder returns the identity), and large with ablation
        if ablate != add_error:
            assert cache[layer_idx] < 100
        else:
            assert cache[layer_idx] > 100


def test_autoencoder_hooks_add_error():
    """Test that the autoencoder hook modifies (or not) the graph correctly"""
    # create transformer and autoencoder
    layer_idx = 4
    n_latents = 10
    model = get_test_model()
    cfg = model.cfg
    autoencoder = Autoencoder(n_latents=n_latents, n_inputs=cfg.d_ff).to(model.device)
    input_tokens = torch.arange(5, device=model.device)[None, :]

    for add_error in [True, False]:
        # create hooks
        hooks = TransformerHooks()
        ae_hooks = AutoencoderHooks(
            encode=unflatten(autoencoder.encode),
            decode=unflatten(autoencoder.decode),
            add_error=add_error,
        )
        hooks.mlp.post_act.append_fwd(AtLayers(layer_idx).append(ae_hooks))

        # make sure to reset autoencoder gradients
        autoencoder.zero_grad()
        assert autoencoder.encoder.weight.grad is None

        # run forward and backward pass, with hooks
        model.zero_grad()
        output, _ = model.forward(input_tokens, hooks=hooks)
        output.sum().backward()
        grads = model.xf_layers[0].mlp.in_layer.weight.grad.clone()

        # run forward and backward pass, without hook
        model.zero_grad()
        output_nohook, _ = model.forward(input_tokens)
        output_nohook.sum().backward()
        grads_nohook = model.xf_layers[0].mlp.in_layer.weight.grad.clone()

        # check forward pass
        # if add_error:
        #     torch.testing.assert_allclose(output, output_nohook, atol=1e-3, rtol=1e-5)
        #     torch.testing.assert_allclose(grads, grads_nohook, atol=1., rtol=1e-1)
        assert add_error == torch.allclose(output, output_nohook, atol=1e-3, rtol=1e-5)
        # check backward pass
        assert add_error == torch.allclose(grads, grads_nohook, atol=1.0, rtol=1e-1)

        # check that we can get gradient on autoencoder latents in all cases
        assert autoencoder.encoder.weight.grad is not None


def test_hook_copy():
    hooks = TransformerHooks()
    hooks_copy = deepcopy(hooks)
    hooks.bind(layer=0)
    hooks_copy.bind(layer=1)  # does not raise

    hooks = Hooks()
    hooks_copy = deepcopy(hooks)
    hooks.bind(layer=0)
    hooks_copy.bind(layer=1)  # does not raise

    hooks = AtLayers(0)
    hooks_copy = deepcopy(hooks)
    assert hooks.condition == hooks_copy.condition

    autoencoder = Autoencoder(n_latents=10, n_inputs=5)
    hooks = AutoencoderHooks(
        encode=unflatten(autoencoder.encode), decode=unflatten(autoencoder.decode)
    )
    hooks_copy = deepcopy(hooks)
    assert hooks.encode == hooks_copy.encode
