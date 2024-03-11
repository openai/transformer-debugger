# TDB Terminology

**Component**

- An attention head or neuron, or autoencoder latent
- Has some set of weights that define what the component does
- Analogy:
    - Component is like the code for a function
    - Node is like the a specific invocation of a function with specific input values and specific output values
- When invoked, each component produces nodes that read something from the unnormalized residual stream, then write some vector (the “write vector”) to the unnormalized residual stream
- Each component is independent from other components of the same type in the same layer

**Node**

- Specific invocation of a component which reads from the normalized residual stream at one token, maybe produces some intermediate values, and then writes to the normalized residual stream at one token
- Comes from talking about nodes in a computation graph/causal graph
- Neurons/latents produce one node per sequence token. they read from/write to the same token
    - Neuron pre/post activations are intermediate values
- Attention heads produce one node per pair of sequence tokens (reading from same/earlier token, writing to later token)
    - Attention QK products, value vectors are intermediate values
- Each node only exists in one forward/backward pass. If you modify the prompt and rerun, that would create different nodes

**Write vector**

- Vector written to the residual stream by a node

**Circuit**

- Set of nodes that work together to perform some behavior/reasoning

**Latent**

- Type of component corresponding to a direction in the activation space learned by a sparse autoencoder for a specific layer's MLP neurons or attention heads
- They correspond more often to semantically meaningful features than neurons or attention heads do

**Ablate**

- Turn off a node
- Right now we use zero-ablation, so the node won’t write anything to the residual stream. In principle we could implement other versions
- Lets you observe the downstream effects of the node
- Answers the question “What is the real effect of this node writing to the residual stream?”

**Trace upstream**

- Look at what upstream nodes caused this node to write to the residual stream
- Answer the question “Why did this node write to the residual stream in this way?”

**Direction of interest**

- TDB looks at the importance of nodes in terms of their effect on a specific direction in the transformer representations
    - In principle this could be a direction in the unnormalized residual stream, normalized residual stream, or some other vector space in transformer representations
    - For now, the only option is the direction in the final normalized residual stream corresponding to the unembedding of a target token minus the unembedding of a distractor token
- The activation of the direction of interest is the projection of the transformer’s representations onto the direction of interest at a specific token on a specific prompt, which yields a scalar value
    - For token differences, the activation = difference in logits = log ratio of probabilities between these two tokens: log(p(target_token)/p(distractor_token))

**Target token**

- The token that corresponds to positive activation along direction of interest, usually the token the model assigns highest probability to

**Distractor token**

- The token that corresponds to negative activation along direction of interest, usually a plausible but incorrect token

**Estimated total effect**

- Estimate of the total effect of the node on the activation of the direction of interest
- Positive values mean that the node increases the activation of the direction of interest; negative values mean that it decreases the activation
- Accounts for the node directly affecting the direction of interest and the indirect effect through intermediate nodes
- Implementation details:
    - Computed by taking the dot product of the activation and the gradient of the direction of interest
    - ACT_TIMES_GRAD in [derived scalars terminology](neuron_explainer/activations/derived_scalars/README.md)

**Direct effect**

- Projection of the node’s output onto the direction of interest
- Positive values mean that the node increases the activation of the direction of interest; negative values mean that it decreases the activation
- Only accounts for the node directly affecting the final residual stream, not impact through intermediate nodes
- Implementation details:
    - Computed by taking the dot product of the activation and the gradient of interest from the final residual stream
    - WRITE_TO_FINAL_RESIDUAL_GRAD in [derived scalars terminology](neuron_explainer/activations/derived_scalars/README.md)

**Write magnitude**

- Magnitude of the write vector produced by the node, including information not relevant to direction of interest
- Higher means that the node is more important to the forward pass
- Implementation details:
    - WRITE_NORM in [derived scalars terminology](neuron_explainer/activations/derived_scalars/README.md)

**Layer**

- Transformers consist of layers which each contain a block of attention heads, followed by a block of MLP neurons

**Upstream**

- Upstream in the causal graph. Modifying an upstream node can impact a downstream node
- Node must be earlier in the forward pass, and must be at the same token or previous token, not a future token

**Downstream**

- Downstream in the causal graph. Modifying a downstream node cannot impact an upstream node (but can impact our estimates, because they use gradients which are impacted by all nodes in the graph)
- Node must be later in the forward pass, and must be at the same token or a subsequent token, not a previous token
