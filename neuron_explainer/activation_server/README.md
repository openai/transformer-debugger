# Activation Server

Backend server for getting activation, neuron and explanation data via HTTP, either from Azure blob storage or from inference on a subject or assistant model.

- Gets activations, loss, or other inference data for individual neurons by performing inference on a subject model
- Reads existing neuron/explanation data from blob storage and returns it via HTTP
- Generates and scores explanations using the OpenAI API

## Running the server

To run activation server for GPT-2 small:

```sh
python neuron_explainer/activation_server/main.py --model_name gpt2-small --port 8000
```

To be able to replace MLP neurons with MLP autoencoder latents, and/or attention heads with attention autoencoder latents, 
the activation server needs to run the corresponding autoencoder:

```sh
python neuron_explainer/activation_server/main.py  --model_name gpt2-small --port 8000 --mlp_autoencoder_name ae-resid-delta-mlp-v4
python neuron_explainer/activation_server/main.py  --model_name gpt2-small --port 8000 --attn_autoencoder_name ae-resid-delta-attn-v4
python neuron_explainer/activation_server/main.py  --model_name gpt2-small --port 8000 --attn_autoencoder_name ae-resid-delta-attn-v4 --mlp_autoencoder_name ae-resid-delta-mlp-v4
```

Running the activation server with autoencoders will add one or two toggle buttons in the Transformer Debugger UI,
to switch between MLP neurons and MLP autoencoder latents ("Use MLP autoencoder"),
or between attention heads and attention autoencoder latents ("Use Attention autoencoder").

See all the available autoencoder names with
```py
from neuron_explainer.models.model_registry import list_autoencoder_names
print(list_autoencoder_names("gpt2-small"))
```

## Generating client libraries

Typescript client libraries for interacting with the activation server are auto-generated based on the server code. If you make any changes to this directory, be sure to regenerate the client libraries:

First, start up a local activation server without running a model:

```sh
python neuron_explainer/activation_server/main.py --run_model False --port 8000
```

Then in another terminal, run:

```sh
cd neuron_viewer
npm run generate-client
```

To run these steps without waiting on activation server to spin up, you can also run

```sh
python neuron_explainer/activation_server/main.py --run_model False --port 8000 &
while ! lsof -i :8000; do sleep 1; done; cd neuron_viewer; npm run generate-client; git commit -am "generate client"
kill -9 $(lsof -t -i:8000)
cd ../../..
```

in the second terminal.

## Code organization

- [main.py](main.py): Entry point for the server.
- \*\_routes.py: Route / endpoint definitions, organized by functionality. See individual files.
- [interactive_model.py](interactive_model.py): Code for performing inference on a model, used by [inference_routes.py](inference_routes.py).

## Debugging cuda memory issues

Pytorch has helpful utilities for debugging cuda memory issues, particularly OOMs. To enable cuda memory debugging, set `--cuda_memory_debugging True` when starting the server (presumably on a devbox). See [main.py](main.py) for the implementation. This will enable memory tracking when the server starts up and dump a snapshot every time the server receives a request to its `/dump_memory_snapshot` endpoint. Once you've generated a snapshot, run the following commands to generate and open an HTML page for viewing it:

```sh
# Download the memory viz script.
curl -o _memory_viz.py https://raw.githubusercontent.com/pytorch/pytorch/main/torch/cuda/_memory_viz.py

# Generate the HTML page.
python _memory_viz.py trace_plot torch_memory_snapshot_*.pkl -o snapshot.html

# Open the HTML page.
open snapshot.html
```

Click on individual bars to see a stack trace for the allocation.

See this blog post for more suggestions: https://pytorch.org/blog/understanding-gpu-memory-1/

## CORS

The activation server is configured to allow requests from any localhost origin. If you decide to
run one or both of the servers remotely, you will need to update the CORS configuration in
[main.py](main.py) to allow requests from the appropriate origins.

## Auto-explanation and scoring

[explainer_routes.py](explainer_routes.py) includes endpoints for explaining and scoring nodes (e.g. MLP neurons, autoencoder latents, attention heads, etc.). Scoring requires a simulator that estimates activations on a given token sequence using an explanation for a node. In the original "Language models can explain neurons in language models" paper, we use a simulator (`ExplanationNeuronSimulator`) that requires the model backing it to return logprobs for the input prompt. Thus, simulation of activations on a token sequence can be performed in a single forward pass. Unfortunately, this logprob-based methodology we used in our last release is no longer feasible since logprobs aren't available for prompt tokens for the relevant production models. This repo has as its default simulator a much slower design where the prompt completion includes the simulated activations (`LogprobFreeExplanationTokenSimulator`). This design is also less reliable, because the completion doesn't necessarily fit the expected format or accurately reproduce the token sequence. For many simulation requests, this simulator will log an error and return all zero estimated activations.
