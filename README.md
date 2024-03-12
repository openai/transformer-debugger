# Transformer Debugger

Transformer Debugger (TDB) is a tool developed by OpenAI's [Superalignment
team](https://openai.com/blog/introducing-superalignment) with the goal of
supporting investigations into specific behaviors of small language models. The tool combines
[automated interpretability](https://openai.com/research/language-models-can-explain-neurons-in-language-models)
techniques with [sparse autoencoders](https://transformer-circuits.pub/2023/monosemantic-features).

TDB enables rapid exploration before needing to write code, with the ability to intervene in the
forward pass and see how it affects a particular behavior. It can be used to answer questions like,
"Why does the model output token A instead of token B for this prompt?" or "Why does attention head
H to attend to token T for this prompt?" It does so by identifying specific components (neurons,
attention heads, autoencoder latents) that contribute to the behavior, showing automatically
generated explanations of what causes those components to activate most strongly, and tracing
connections between components to help discover circuits.

These videos give an overview of TDB and show how it can be used to investigate [indirect object
identification in GPT-2 small](https://arxiv.org/abs/2211.00593):

- [Introduction](https://www.loom.com/share/721244075f12439496db5d53439d2f84?sid=8445200e-c49e-4028-8b8e-3ea8d361dec0)
- [Neuron viewer pages](https://www.loom.com/share/21b601b8494b40c49b8dc7bfd1dc6829?sid=ee23c00a-9ede-4249-b9d7-c2ba15993556)
- [Example: Investigating name mover heads](https://www.loom.com/share/3478057cec484a1b85471585fef10811?sid=b9c3be4b-7117-405a-8d31-0f9e541dcfb6)
- [Example: Beyond name mover heads](https://www.loom.com/share/6bd8c6bde84b42a98f9a26a969d4a3ad?sid=4a09ac29-58a2-433e-b55d-762414d9a7fa)

## What's in the release?

- [Neuron viewer](neuron_viewer/README.md): A React app that hosts TDB as well as pages with information about individual model components (MLP neurons, attention heads and autoencoder latents for both).
- [Activation server](neuron_explainer/activation_server/README.md): A backend server that performs inference on a subject model to provide data for TDB. It also reads and serves data from public Azure buckets.
- [Models](neuron_explainer/models/README.md): A simple inference library for GPT-2 models and their autoencoders, with hooks to grab activations.
- [Collated activation datasets](datasets.md): top-activating dataset examples for MLP neurons, attention heads and autoencoder latents.

## Setup

Follow these steps to install the repo.  You'll first need python/pip, as well as node/npm. Requires `Python >= 3.11`

Though optional, we recommend you use a virtual environment or equivalent:

```sh
# If you're already in a venv, deactivate it.
deactivate
# Create a new venv.
python -m venv ~/.virtualenvs/transformer-debugger
# Activate the new venv.
source ~/.virtualenvs/transformer-debugger/bin/activate
```

Once your environment is set up, follow the following steps:
```sh
git clone git@github.com:openai/transformer-debugger.git
cd transformer-debugger

# Install neuron_explainer
pip install -e .

# Set up the pre-commit hooks.
pre-commit install

# Install neuron_viewer.
cd neuron_viewer
npm install
cd ..
```

To run the TDB app, you'll then need to follow the instructions to set up the [activation server backend](neuron_explainer/activation_server/README.md) and [neuron viewer frontend](neuron_viewer/README.md).

## Making changes

To validate changes:

- Run `pytest`
- Run `mypy --config=mypy.ini .`
- Run activation server and neuron viewer and confirm that basic functionality like TDB and neuron
  viewer pages is still working


## Links

- [Terminology](terminology.md)

## How to cite

Please cite as:

```
Mossing, et al., “Transformer Debugger”, GitHub, 2024.
```

BibTex citation:

```
@misc{mossing2024tdb,
  title={Transformer Debugger},
  author={Mossing, Dan and Bills, Steven and Tillman, Henk and Dupré la Tour, Tom and Cammarata, Nick and Gao, Leo and Achiam, Joshua and Yeh, Catherine and Leike, Jan and Wu, Jeff and Saunders, William},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/openai/transformer-debugger}},
}
```
