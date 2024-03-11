from collections import OrderedDict
from copy import deepcopy

import torch


class Hooks:
    """A callable that is a sequence of callables"""

    def __init__(self):
        self._hooks = []
        self.bound_kwargs = {}

    def __call__(self, x, *args, **kwargs):
        for hook in self._hooks:
            x = hook(x, *args, **kwargs, **self.bound_kwargs)
        return x

    def append(self, hook):
        self._hooks.append(hook)
        return self

    def bind(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.bound_kwargs:
                raise ValueError(f"Key {key} already bound")
            self.bound_kwargs[key] = value
        return self

    def unbind(self, keys: list):
        for key in keys:
            del self.bound_kwargs[key]
        return self

    def __repr__(self, indent=0, name=None):
        import inspect

        indent_str = " " * indent
        full_name = f"{name}" if name is not None else self.name
        if self.bound_kwargs:
            full_name += f" {self.bound_kwargs}"
        if self.is_empty():
            return f"{indent_str}{full_name}"

        def hook_repr(hook):
            if "indent" in inspect.signature(hook.__class__.__repr__).parameters:
                return hook.__repr__(indent=indent + 4)
            else:
                return indent_str + " " * 4 + repr(hook)

        hooks_repr = "\n".join(f"{hook_repr(hook)}" for hook in self._hooks)
        return f"{indent_str}{full_name}\n{hooks_repr}"

    @property
    def name(self):
        return self.__class__.__name__

    def is_empty(self):
        return len(self._hooks) == 0


# takes a gradient hook and makes into a forward pass hook
def grad_hook_wrapper(grad_hook):
    def fwd_hook(act, *args, **kwargs):
        class _IdentityWithGradHook(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return tensor

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = grad_hook(grad_output, *args, **kwargs)
                return grad_output

        return _IdentityWithGradHook.apply(act)

    return fwd_hook


class HookCollection:
    def __init__(self):
        self.all_hooks = OrderedDict()

    def bind(self, **kwargs):
        for hook in self.all_hooks.values():
            hook.bind(**kwargs)
        return self

    def unbind(self, keys):
        for hook in self.all_hooks.values():
            hook.unbind(keys)
        return self

    def append_all(self, hook):
        for hooks in self.all_hooks.values():
            try:
                hooks.append_all(hook)
            except AttributeError:
                hooks.append(hook)
        return self

    def append_to_path(self, hook_location_name, hook):
        """
        Adds a hook to a location in a nested hook collection with a dot-separated name.
        e.g. `self.append_to_path("resid.torso.post_mlp.fwd", hook)` adds `hook` to
        `self.all_hooks["resid"].all_hooks["torso"].all_hooks["post_mlp"].all_hooks["fwd"]`
        """
        hook_location_parts = hook_location_name.split(".", 1)  # split at first dot, if present
        top_level_location = hook_location_parts[0]
        assert top_level_location in self.all_hooks
        if len(hook_location_parts) == 1:  # no dot in path
            self.all_hooks[top_level_location].append(hook)
        else:  # at least one dot in path -> split outputs two parts
            sub_location = hook_location_parts[1]
            self.all_hooks[top_level_location].append_to_path(sub_location, hook)
        return self

    def __deepcopy__(self, memo):
        # can't use deepcopy because of __getattr__
        new = self.__class__()
        new.all_hooks = deepcopy(self.all_hooks)
        return new

    def add_subhooks(self, name, subhooks):
        self.all_hooks[name] = subhooks
        return self

    def __getattr__(self, name):
        if name in self.all_hooks:
            return self.all_hooks[name]
        else:
            raise AttributeError(f"HookCollection has no attribute {name}")

    def __repr__(self, indent=0, name=None):
        indent_str = " " * indent
        full_name = f"{name}" if name is not None else self.__class__.__name__
        prefix = f"{name}." if name is not None else ""
        hooks_repr = "\n".join(
            hook.__repr__(indent + 4, f"{prefix}{hook_name}")
            for hook_name, hook in self.all_hooks.items()
        )
        return f"{indent_str}{full_name}\n{hooks_repr}"

    def is_empty(self):
        return all(hook.is_empty() for hook in self.all_hooks.values())


class FwdBwdHooks(HookCollection):
    def __init__(self):
        super().__init__()
        # By default, all grad hooks are applied after all forward hooks.  This way,
        # the gradients are given for the final "hooked" output of the forward pass.
        # If you want gradients for an intermediate output, you should simply
        # append_fwd(from_grad_hook(hook)) at the appropriate time.
        self.add_subhooks("fwd", Hooks())
        self.add_subhooks("bwd", WrapperHooks(wrapper=grad_hook_wrapper))
        self.add_subhooks("fwd2", Hooks())

    def append_fwd(self, fwd_hook):
        self.fwd.append(fwd_hook)
        return self

    def append_bwd(self, bwd_hook):
        self.bwd.append(bwd_hook)
        return self

    def append_fwd2(self, fwd2_hook):
        self.fwd2.append(fwd2_hook)
        return self

    def __call__(self, x, *args, **kwargs):
        # hooks into fwd, then bwd, then fwd2
        x = self.fwd(x, *args, **kwargs)
        x = self.bwd(x, *args, **kwargs)
        x = self.fwd2(x, *args, **kwargs)
        return x


class MLPHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("pre_act", FwdBwdHooks())
        self.add_subhooks("post_act", FwdBwdHooks())


class NormalizationHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("post_mean_subtraction", FwdBwdHooks())
        self.add_subhooks("scale", FwdBwdHooks())
        self.add_subhooks("post_scale", FwdBwdHooks())


class AttentionHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("q", FwdBwdHooks())
        self.add_subhooks("k", FwdBwdHooks())
        self.add_subhooks("v", FwdBwdHooks())
        self.add_subhooks("qk_logits", FwdBwdHooks())
        self.add_subhooks("qk_softmax_denominator", FwdBwdHooks())
        self.add_subhooks("qk_probs", FwdBwdHooks())
        self.add_subhooks("v_out", FwdBwdHooks())  # pre-final projection


class ResidualStreamTorsoHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("post_ln_attn", FwdBwdHooks())
        self.add_subhooks("ln_attn", NormalizationHooks())
        self.add_subhooks("delta_attn", FwdBwdHooks())
        self.add_subhooks("post_attn", FwdBwdHooks())
        self.add_subhooks("ln_mlp", NormalizationHooks())
        self.add_subhooks("post_ln_mlp", FwdBwdHooks())
        self.add_subhooks("delta_mlp", FwdBwdHooks())
        self.add_subhooks("post_mlp", FwdBwdHooks())


class ResidualStreamHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("post_emb", FwdBwdHooks())
        self.add_subhooks("torso", ResidualStreamTorsoHooks())
        self.add_subhooks("ln_f", NormalizationHooks())
        self.add_subhooks("post_ln_f", FwdBwdHooks())


class TransformerHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("mlp", MLPHooks())
        self.add_subhooks("attn", AttentionHooks())
        self.add_subhooks("resid", ResidualStreamHooks())
        self.add_subhooks("logits", FwdBwdHooks())


class WrapperHooks(Hooks):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        super().__init__()

    def append(self, fn):
        self._hooks.append(self.wrapper(fn))


class ConditionalHooks(Hooks):
    def __init__(self, condition):
        self.condition = condition
        super().__init__()

    def __call__(self, x, *args, **kwargs):
        cond = self.condition(x, *args, **kwargs)
        if cond:
            x = super().__call__(x, *args, **kwargs)
        return x


class AtLayers(ConditionalHooks):
    def __init__(self, at_layers: int | list[int]):
        if isinstance(at_layers, int):
            at_layers = [at_layers]
        self.at_layers = at_layers

        def at_layers_condition(x, *, layer, **kwargs):
            return layer in at_layers

        super().__init__(condition=at_layers_condition)

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.at_layers})"


class AutoencoderHooks(HookCollection):
    """
    Hooks into the hidden dimension of an autoencoder.
    """

    def __init__(self, encode, decode, add_error=False):
        super().__init__()
        # hooks
        self.add_subhooks("latents", FwdBwdHooks())
        self.add_subhooks("reconstruction", FwdBwdHooks())
        self.add_subhooks("error", FwdBwdHooks())
        # autoencoder functions
        self.encode = encode
        self.decode = decode
        # if add_error is True, add the error to the reconstruction.
        self.add_error = add_error

    def __call__(self, x, *args, **kwargs):
        latents = self.encode(x)
        if self.add_error:
            # Here, the latents are cloned twice:
            # - the first clone is passed through `self.latents` and `self.reconstruction`
            # - the second clone is passed through `self.error`
            latents_to_hook = latents.clone()
            latents_to_error = latents.clone()
        else:
            latents_to_hook = latents

        latents_to_hook = self.latents(latents_to_hook, *args, **kwargs)  # call hooks
        reconstruction = self.decode(latents_to_hook)
        reconstruction = self.reconstruction(reconstruction, *args, **kwargs)  # call hooks

        if self.add_error:
            error = x - self.decode(latents_to_error)
            error = self.error(error, *args, **kwargs)  # call hooks
            return reconstruction + error
        else:
            error = x - reconstruction
            error = self.error(error, *args, **kwargs)  # call hooks
            return reconstruction

    def __deepcopy__(self, memo):
        # can't use deepcopy because of __getattr__
        new = self.__class__(self.encode, self.decode, self.add_error)
        new.all_hooks = deepcopy(self.all_hooks)
        return new
