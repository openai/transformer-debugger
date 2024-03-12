This directory contains:
- a simple transformer and autoencoder implementation, along with ways to hook their internals.
- the interfaces ModelContext and AutoencoderContext, which we use to wrap the simple implementations.

The rest of our codebase can work to some degree with other transformer implementations, so long as they are wrapped in ModelContext.

## Transformer

Example usage

```python
import torch
from neuron_explainer.models.hooks import TransformerHooks
from neuron_explainer.models.model_context import get_default_device
from neuron_explainer.models.transformer import Transformer

# load a saved model and cast all params to a dtype
device = get_default_device()
xf = Transformer.load("gpt2-small", dtype=torch.float32, device=device)

# forward pass
random_tokens = torch.randint(0, xf.n_vocab, (4, 128)).to(xf.device)
logits, kv_caches = xf(random_tokens)

# sample from the model
out = xf.sample("Hello, transformer!", num_tokens=10, top_p=0.5, temperature=1.0)
# out is a dict {"tokens": list[list[int]], "strings": list[str]}
print(out["strings"][0])

# create a hook to ablate the activations of the 300th neuron in the 3rd layer
def make_ablation_hook(at_layer, neuron):
    def ablate_neuron(xx, layer, **kwargs):
        if layer == at_layer:
            xx[..., neuron] = 0
        return xx

    return ablate_neuron

hooks = TransformerHooks()
hooks.mlp.post_act.append_fwd(make_ablation_hook(3, 300))

# sample from the model with the hooks
out = xf.sample("Hello, transformer!", hooks=hooks, num_tokens=10, top_p=0.5, temperature=1.0)
print(out["strings"][0])
```

The argument to `Transformer.load()` can be any folder.

## Sparse autoencoder

The `neuron_explainer.models.autoencoder` module implements a sparse autoencoder trained on the GPT-2 small model's activations.
The autoencoder's purpose is to expand the MLP layer activations into a larger number of dimensions,
providing an overcomplete basis of the MLP activation space. The learned dimensions have been
shown to be more interpretable than the original MLP dimensions.

The module is a slightly modified version of `https://github.com/openai/sparse_autoencoder`. It is included in this repository for convenience.

### Autoencoder settings

- Model used: "gpt2-small", 12 layers
- Autoencoder architecture: see [`autoencoder.py`](autoencoder.py)
- Autoencoder input: "mlp_post_act" (3072 dimensions), "resid_delta_mlp" (768 dimensions), or "resid_delta_attn" (768 dimensions)
- Number of autoencoder latents: 32768
- Number of training tokens: ~64M
- L1 regularization strength: 0.01 or 0.03 ("_v4")

### Trained autoencoder files

Trained autoencoder files (saved as torch state-dicts) are located at the following paths:
`az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}{version}/autoencoders/{layer_index}.pt`

with the following parameters:
- `autoencoder_input` is in ["mlp_post_act", "resid_delta_mlp", "resid_delta_attn"]
- `version` is in ["", "_v4"] ("resid_delta_attn" only available for "_v4")
- `layer_index` is in range(12) (GPT-2 small)

### Collated activation datasets

Note: collated activation datasets located at `az://openaipublic/sparse-autoencoder/gpt2-small` are not compatible with Transformer Debugger. Use the following datasets instead:
`az://openaipublic/neuron-explainer/gpt2-small/autoencoder_latent/{autoencoder_input}{version}/collated-activations/{layer_index}/{latent_index}.json`

See [datasets.md](../../datasets.md) for more details.

### Example usage - with Transformer Debugger

- see [Neuron Viewer](../../neuron_viewer/README.md) for instructions on how to start a Neuron Viewer or a Transformer Debugger with autoencoders.
- see [Activation Server](../../neuron_explainer/activation_server/README.md) for instructions on how to start an Activation Server with autoencoders.


### Example usage - with transformer hooks

Autoencoders can be used outside of Transformer Debugger.
Here, we provide a simple example showing how to extract neuron/attention activations, and how to encode them with autoencoders.


```py
import blobfile as bf
import torch

from neuron_explainer.models.autoencoder import Autoencoder
from neuron_explainer.models.hooks import TransformerHooks
from neuron_explainer.models.model_context import get_default_device
from neuron_explainer.models.transformer import Transformer

# Load the autoencoder
layer_index = 0  # in range(12)
autoencoder_input = ["mlp_post_act", "resid_delta_mlp", "resid_delta_attn"][1]
version = ["", "_v4"][1]
filename = f"az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}{version}/autoencoders/{layer_index}.pt"
with bf.BlobFile(filename, mode="rb") as f:
    print(f"Loading autoencoder..")
    state_dict = torch.load(f)
    autoencoder = Autoencoder.from_state_dict(state_dict, strict=False)

# Load the transformer
device = get_default_device()
print(f"Loading transformer..")
transformer = Transformer.load("gpt2/small", dtype=torch.float32, device=device)

# create hooks to store activations during the forward pass
def store_forward(xx, layer, **kwargs):
    activation_cache[layer] = xx.detach().clone()
    return xx

activation_cache = {}
hooks = TransformerHooks()
if autoencoder_input == "mlp_post_act":
    hooks.mlp.post_act.append_fwd(store_forward)
elif autoencoder_input == "resid_delta_mlp":
    hooks.resid.torso.delta_mlp.append_fwd(store_forward)
elif autoencoder_input == "resid_delta_attn":
    hooks.resid.torso.delta_attn.append_fwd(store_forward)

# Run the transformer and store activations
prompt = "What is essential is invisible to the"
tokens = transformer.enc.encode(prompt)  # (1, n_tokens)
print("tokenized prompt:", transformer.enc.decode_batch([[token] for token in tokens]))
with torch.no_grad():
    transformer(torch.tensor([tokens], device=device), hooks=hooks)
input_tensor = activation_cache[layer_index][0]  # (n_tokens, n_input_dimensions)

# Encode activations with the autoencoder
autoencoder.to(device)
with torch.no_grad():
    latent_activations = autoencoder.encode(input_tensor)  # (n_tokens, n_latents)
```
