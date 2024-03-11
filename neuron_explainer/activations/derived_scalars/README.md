Derived scalar types provide a shared interface for processing activations that are directly instantiated during the forward pass (corresponding to `ActivationLocationTypes`; for example, post-softmax attention), and functions of those activations which are useful to look at but not directly instantiated during the forward pass (for example, the norm of the attention write vector).

They are intended to be as flexible as possible, across models, architectures, and use cases (online processing or batch scripts); the cost of that is that there are many abstractions. The good news is that once you're familiar with the setup, it's pretty quick and easy to compute them and to define new ones.

Note that in many places in the `neuron_explainer` codebase, "dst" is used in place of "derived_scalar_type", or "ds" in place of "derived_scalar", for conciseness.

The key classes to understand are
 - `DerivedScalarType`: an enum containing names for all the derived scalars that have been defined
 - `PassType`: forward or backward pass; derived scalars can depend on either forward pass activations, or backward
 pass gradients
 - `ActivationsAndMetadata`: a container for a set of pytorch tensors, corresponding to a certain `DerivedScalarType`
 and `PassType` at every layer index in the model for which it is defined.
 - `ScalarDeriver`: an object specified by a certain `DerivedScalarType`, containing the necessary information to compute derived scalars for one or both `PassTypes`; specification often requires additional info in the `DstConfig`
 - `DstConfig`: a dataclass specifying any information required in constructing the `ScalarDeriver` beyond the `DerivedScalarType`. For example, DSTs that use weight tensors must know what model they are being computed for, so that the correct weights can be accessed. 
 - `ScalarSource`: an object specifying the inputs expected by a `ScalarDeriver`. This consists of either an `ActivationLocationType` and `PassType`, or a `ScalarDeriver` and `PassType` (for the case where a `ScalarDeriver`'s inputs themselves require a `ScalarDeriver` to compute) plus a `LayerIndexer`. These two types are referred to as `RawScalarSource` and `DerivedScalarSource` respectively. "Raw" in this name refers to an activation is literally instantiated during a transformer forward/backward pass, and can be extracted using a hook at a particular line of code. "Derived" refers to a quantity that can be computed from activations instantiated during a forward/backward pass (a superset of "Raw" activations).
 - `LayerIndexer`: defines a transformation to be applied to an `ActivationsAndMetadata` object such that each layer
 index of the output is the appropriate layer index for computing a given derived scalar. For example, sometimes a derived scalar D at layer L (for L in `0...num_layers-1`) is computed by operating on activation A from layer L together with activation B from layer L0 (constant). In this case, we would apply a `ConstantLayerIndexer` to B, such that the activations of the `ActivationsAndMetadata` passed to the scalar deriver are from layer L0 of B, no matter the value of L.
 - `RawActivationStore`: contains `ActivationsAndMetadata` stored from a forward and possibly a backward pass, separated by `ActivationLocationType` and `PassType`.
 - `DerivedScalarStore`: contains `ActivationsAndMetadata` computed from a `RawActivationStore`, separated by `DerivedScalarType` and `PassType`.

Derived scalars are computed like this:
 - User specifies `ScalarDeriver` objects using (`DerivedScalarType`, `DstConfig`) tuples, and constructs them.
 - For each `ScalarDeriver`, `scalar_deriver.sub_activation_location_type_and_pass_types` lets the user know the list of `ActivationLocationType` and `PassTypes` that it will require.
 - User populates a `RawActivationStore` with these `ActivationLocationTypes` and `PassType`s, whether by reading activations from disk, or computing fresh activations from a forward and backward pass on a running LM. 
 - User runs `derived_scalar_store = DerivedScalarStore.derive_from_raw(raw_activation_store, scalar_derivers)`
    - (under the hood) for each `ScalarDeriver`, run `derived_scalar_activations_and_metadata = scalar_deriver.derive_from_raw(raw_activation_store, pass_type)`
        - (under the hood) for each of the `ScalarSource` objects in `scalar_deriver.sub_scalar_sources`, run `scalar_source_activations_and_metadata = sub_scalar_source.derive_from_raw(raw_activation_store)`.
            - (under the hood) this either gets an `ActivationsAndMetadata` object by `ActivationLocationType` and `PassType` directly from `raw_activation_store`, or derives it using `sub_scalar_source.scalar_deriver.derive_from_raw(raw_activation_store, pass_type)`, then applies `sub_scalar_source.layer_indexer`.
        - (under the hood) run `derived_scalar_activations_and_metadata = scalar_deriver.activations_and_metadata_calculate_derived_scalar_fn(scalar_source_activations_and_metadata_tuple, pass_type)`.
            - (under the hood) run `derived_scalar_tensor = scalar_deriver.tensor_calculate_derived_scalar_fn(scalar_source_tensor_tuple, layer_index, pass_type)` for each `layer_index`, which together populate the `ActivationsAndMetadata` object.
    - (under the hood) the `ActivationsAndMetadata` objects together populate the `DerivedScalarStore`.

Optionally, the outermost loop may be skipped if only a single `derived_scalar_activations_and_metadata` object is required.

The most bare-bones function that shows the steps outlined above is in `activation_server/derived_scalar_computation.py:get_derived_scalars_for_prompt`.


When defining a new `ScalarDeriver` that does not correspond to an `ActivationLocationType`, the user provides (1) the information necessary to compute the derived scalar. This includes (a) the `ScalarSources` it expects, (b) `tensor_calculate_derived_scalar_fn`, which takes as arguments tensors corresponding to the `ScalarSources` as well as the `layer_index` and `pass_type`. This lives in a function called `make_..._scalar_deriver`. These can be specified implicitly, if the new scalar can be derived from a transformation on one or more pre-existing derived scalars. The `make_..._scalar_deriver` functions are associated to DSTs in `derived_scalars/make_scalar_derivers.py`. (2) a specification of the shape of the output, in terms of `Dimension` objects. This lives in `derived_scalars/derived_scalar_types.py`.

`ScalarDeriver` is meant to be the primary class used to refer to activations once they are READ from disk. When WRITTEN to disk, the primary class used is `ActivationLocationType`, since we always want to save the least processed form of the data possible.


TO DEFINE A NEW DERIVED SCALAR TYPE (DST):

1. Add a new `DerivedScalarType` to the Enum in `scalar_deriver.py`.

2. Add a specification of its intended shape per token sequence to `shape_spec_per_token_sequence_by_dst` (e.g. does it contain a separate dimension for attended-to sequence tokens? Is it per-attention head or per-MLP neuron?)

3. In a separate file (possibly an existing file, if related DSTs have been defined already) write a `make_..._scalar_deriver` function. This function takes a `DstConfig` and returns a `ScalarDeriver` object. The core of this object is `tensor_calculate_derived_scalar_fn`, which takes as input a tuple of tensors corresponding to an existing activation location type and pass type or DST and pass type, and returns a tensor corresponding to the new DST. To construct `tensor_calculate_derived_scalar_fn`, you might require some metadata (for example, a `ModelContext` object which gives the ability to load model weights from disk). If your DST requires a new piece of metadata, add it to `DstConfig` as an optional argument. In addition to `calculate_derived_scalar_fn`, you must also specify which ScalarSources are required to compute this `DerivedScalarType` (`sub_scalar_sources`)

4. Once the `make_..._scalar_deriver` function is done, add a row like this to the registry in `make_scalar_derivers.py`:
    `DerivedScalarType.NEW_DST: make_new_dst_scalar_deriver`,


FOR A SIMPLE EXAMPLE OF A `make_..._scalar_deriver` FUNCTION: 

See `make_mlp_write_norm_scalar_deriver` in `mlp.py`.


USAGE EXAMPLE:

```py
import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
    maybe_construct_loss_fn_for_backward_pass,
)
from neuron_explainer.activation_server.requests_and_responses import LossFnConfig, LossFnName
from neuron_explainer.activations.derived_scalars.config import DstConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.models.model_context import StandardModelContext

prompt = "This is a test"

# this object contains model metadata and has methods for loading weights. It also has a 
# method to spin up a transformer for running a forward and backward pass.
model_context = StandardModelContext(model_name="gpt2-small", device=torch.device("cuda:0"))

# these are the derived scalars of interest; these correspond to direct writes from
# MLP neurons and attention heads to the gradient at the final residual
# stream layer ("direct effects" on the loss).
dst_list = [
    DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
]

# the derived scalars require model weights and metadata (such as the number of layers)
# for their computation. Nothing else needs to be specified.
dst_config = DstConfig(
    model_context=model_context,
)
dst_and_config_list = [(dst, dst_config) for dst in dst_list]

# this specifies how the backward pass will be run (outside the DST framework)
loss_fn_for_backward_pass = maybe_construct_loss_fn_for_backward_pass(
    model_context=model_context,
    config=LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=["."],
        distractor_tokens=["!"],
    ),
)

# this returns a DerivedScalarStore containing the DSTs specified, as well as a RawActivationStore
# containing the activations that were required to compute those DSTs.
ds_store, _, raw_store = get_derived_scalars_for_prompt(
    model_context=model_context,
    prompt=prompt,
    loss_fn_for_backward_pass=loss_fn_for_backward_pass,
    dst_and_config_list=dst_and_config_list,
)

# this returns the top 10 largest derived scalar values, across all the types specified, as well 
# as identifiers for the location of each within the model (i.e. which neuron or attention head 
# they correspond to, and at which token or token pair)
values, indices = ds_store.topk(10)

```
