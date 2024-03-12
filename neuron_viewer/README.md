# Neuron viewer

A React app that hosts TDB as well as pages with information about individual model components
(MLP neurons, attention heads and autoencoder latents for both).


## Running the server locally

First, install the app:

```sh
npm install
```

Then run the frontend:

```sh
npm start
```

- To open a Neuron Viewer page, navigate to `http://localhost:1234`.
- To open TDB, navigate to `http://localhost:1234/gpt2-small/tdb_alpha`.
- To open TDB with autoencoders, navigate to `http://localhost:1234/gpt2-small_ae-resid-delta-mlp-v4_ae-resid-delta-attn-v4/tdb_alpha`
(where `ae-resid-delta-mlp-v4` and `ae-resid-delta-attn-v4` must match the autoencoder names that are used in the [activation server](../neuron_explainer/activation_server/README.md)).

## Formatting code

To check whether the code is correctly formatted:

```sh
npm run check-code-format
```

To format the code:

```sh
npm run format-code
```

## Code organization

- [src/client](src/client/): Auto-generated code for interacting with the activation server (the neuron viewer's backend). Do not edit this code! Follow the instructions in [the activation server README](../neuron_explainer/activation_server/README.md) to regenerate this code if you make changes to the activation server. Use [src/requests](src/requests/) when calling the activation server.
- [src/panes](src/panes/): UI elements that can be used as panes on a page: tokens+activations, similar neurons, etc.
- [src/requests](src/requests/): Client libraries for making network requests to the activation server.
- [src/TransformerDebugger](src/TransformerDebugger/): Code related to the Transformer Debugger.
- [src](src/): Other code.

## Using a remote activation server

If you decide to run your activation server on a different host or port than the default, you can
point neuron viewer at it by setting the `NEURON_VIEWER_ACTIVATION_SERVER_URL` environment variable:
    
```sh
NEURON_VIEWER_ACTIVATION_SERVER_URL=https://some.url:port npm start
```

## Making changes

Be sure to run the following to validate any changes you make:

```sh
npm run check-type-warnings && npm run check-code-format && npm run build
```
