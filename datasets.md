# Collated activation datasets

This document lists the collated activation datasets that are compatible with the Transformer Debugger. These datasets contain some top-activating examples for each MLP neuron, attention head, and autoencoder latent, as well as the corresponding activations for each token (or token pair) in the example. They provide a way to visualize what each neuron, attention head, or autoencoder latent is selective for (obviously in an incomplete way). These activation datasets are used by the [neuron viewer](neuron_viewer/README.md) to display the top-activating examples for each component, and are also typically used for [automated interpretability](https://openai.com/research/language-models-can-explain-neurons-in-language-models).

The activations datasets are located on Azure Blob Storage, for example accessible via the [`blobfile`](https://github.com/blobfile/blobfile) library. 

# GPT-2 small

Collated activation datasets are available for both the MLP neurons and the attention heads. MLP neuron activations are recorded for each token, while attention head activations are recorded for each token pair. 

The datasets are located at the following paths:
> - MLP neurons: `https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2_small_data/collated-activations/{layer_index}/{neuron_index}.json`
> - Attention heads: `https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2_small/attn_write_norm/collated-activations-by-token-pair/{layer_index}/{head_index}.json`

with the following parameters:
- `layer_index` is in range(12)
- `neuron_index` is in range(3084)
- `head_index` is in range(12)


## GPT-2 small - MLP autoencoders

MLP autoencoders were trained either on the MLP neurons (after the activation function), or on the MLP-layer output that is written to the residual stream. See [Autoencoders for GPT-2 small](neuron_explainer/autoencoder/README.md) for more details. 

The datasets are located at the following paths:

> - MLP latents: `https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2-small/autoencoder_latent/{autoencoder_input}{version}/collated-activations/{layer_index}/{latent_index}.pt`

with the following parameters:
- `autoencoder_input` is in ["mlp_post_act", "resid_delta_mlp"]
- `version` is in ["", "_v4"]. (The `_v4` versions use slightly different hyperparameters, and should be preferred.)
- `layer_index` is in range(12)
- `latent_index` is in range(32768)

## GPT-2 small - Attention autoencoders

Attention autoencoders were trained on the attention-layer output that is written to the residual stream. See [Autoencoders for GPT-2 small](neuron_explainer/autoencoder/README.md) for more details. The `collated-activations` dataset contains autoencoder latent activations for each token, while the `collated-activations-by-token-pair` dataset contains autoencoder latent *attribution* to each token pair. To compute the attribution given an autoencoder latent `L` and a token pair `(T1, T2)`, we multiply the attention pattern `A(T1, T2)` with the gradient of `L` with respect to the attention pattern: `attribution_L(T1, T2) = A(T1, T2) * ∂L/∂A(T1, T2)`. 

The datasets are located at the following paths:

> - Attention latents (by token): `https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2-small/autoencoder_latent/resid_delta_attn_v4/collated-activations/{layer_index}/{latent_index}.pt`
> - Attention latents (by token pair): `https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2-small/autoencoder_latent/resid_delta_attn_v4/collated-activations-by-token-pair/{layer_index}/{latent_index}.pt`

with the following parameters:
- `layer_index` is in range(12)
- `latent_index` is in range(10240)



# GPT-2 xl

For GPT-2 xl, only the MLP neurons activations are available. The datasets are located at the following paths:
> - MLP neurons: `https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations/{layer_index}/{neuron_index}.json`

with the following parameters:
- `layer_index` is in range(48)
- `neuron_index` is in range(6400)
