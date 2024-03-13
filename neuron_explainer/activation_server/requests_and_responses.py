"""
Request and response definitions. This file shouldn't contain any functions, other than those
defined on the dataclasses.

Requests to InteractiveModel have two parts: an InferenceRequestSpec, specifying how to
run inference to obtain activations, and a ProcessingRequestSpec, specifying how to process those
activations to obtain derived scalars. An InferenceSubRequest contains information for a single
inference step (forward and optionally also backward pass), and one or more ProcessingRequestSpecs
to process activations from the same inference step. A BatchedRequest contains one or more
InferenceSubRequests, whose inference steps are run in parallel, and whose processing steps are
performed sequentially.

InferenceRequests are analogous to single InferenceSubRequests, and are processed stand-alone rather
than in a batch.

TdbRequests compactly specify the information in InferenceRequestSpec and ProcessingRequestSpec,
with only the degrees of freedom permitted by the TDB UI. TdbRequests are converted to
InferenceSubRequests and ProcessingRequestSpecs in tdb_conversions.py. BatchedTdbRequests are
analogous to BatchedRequests.
"""

import math
from enum import Enum
from typing import Any, Literal, Union

import torch
from pydantic import root_validator

from neuron_explainer.activation_server.load_neurons import NodeIdAndDatasets
from neuron_explainer.activation_server.read_routes import TokenAndAttentionScalars
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    AblationSpec,
    MirroredActivationIndex,
    MirroredNodeIndex,
    MirroredTraceConfig,
    NodeAblation,
    NodeToTrace,
)
from neuron_explainer.activations.derived_scalars.tokens import TopTokens
from neuron_explainer.models.model_component_registry import Dimension, LayerIndex, PassType
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable

########## Types used by multiple requests and/or responses ##########


# NOTE: other than TDB_REQUEST_SPEC, these all contain params for processing activations
# only, and not for specifying how to run inference to obtain those activations
class SpecType(Enum):
    ACTIVATIONS_REQUEST_SPEC = "activations_request_spec"
    DERIVED_SCALARS_REQUEST_SPEC = "derived_scalars_request_spec"
    DERIVED_ATTENTION_SCALARS_REQUEST_SPEC = "derived_attention_scalars_request_spec"
    MULTIPLE_TOP_K_DERIVED_SCALARS_REQUEST_SPEC = "multiple_top_k_derived_scalars_request_spec"
    SCORED_TOKENS_REQUEST_SPEC = "scored_tokens_request_spec"
    TDB_REQUEST_SPEC = "tdb_request_spec"
    TOKEN_PAIR_ATTRIBUTION_REQUEST_SPEC = "token_pair_attribution_request_spec"


class ProcessingResponseDataType(Enum):
    DERIVED_SCALARS_RESPONSE_DATA = "derived_scalars_response_data"
    DERIVED_ATTENTION_SCALARS_RESPONSE_DATA = "derived_attention_scalars_response_data"
    MULTIPLE_TOP_K_DERIVED_SCALARS_RESPONSE_DATA = "multiple_top_k_derived_scalars_response_data"
    SCORED_TOKENS_RESPONSE_DATA = "scored_tokens_response_data"
    TOKEN_PAIR_ATTRIBUTION_RESPONSE_DATA = "token_pair_attribution_response_data"


class LossFnName(str, Enum):
    LOGIT_DIFF = "logit_diff"
    LOGIT_MINUS_MEAN = "logit_minus_mean"
    PROBS = "probs"
    ZERO = "zero"


class LossFnConfig(CamelCaseBaseModel):
    name: LossFnName
    target_tokens: list[str] | None = None
    distractor_tokens: list[str] | None = None


@immutable
class InferenceRequestSpec(CamelCaseBaseModel):
    """The minimum specification for performing a forward and/or backward pass on a model, with hooks at some set of layers."""

    prompt: str
    ablation_specs: list[AblationSpec] | None = None
    # note that loss_fn_config and trace_config are mutually exclusive
    loss_fn_config: LossFnConfig | None = None
    # used for performing a backward pass from an internal point within the network
    trace_config: MirroredTraceConfig | None = None
    # used for tracing latent activations back to the activations for the DSTs which they encode
    activation_index_for_within_layer_grad: MirroredActivationIndex | None = None


@immutable
class InferenceRequest(CamelCaseBaseModel):
    inference_request_spec: InferenceRequestSpec


class InferenceData(CamelCaseBaseModel):
    inference_time: float
    memory_used_before: float | None
    memory_used_after: float | None
    log: str | None = None
    loss: float | None = None
    activation_value_for_backward_pass: float | None = None


@immutable
class InferenceAndTokenData(InferenceData):
    tokens_as_ints: list[int]
    tokens_as_strings: list[str]


@immutable
class InferenceResponse(CamelCaseBaseModel):
    inference_and_token_data: InferenceAndTokenData


class GroupId(str, Enum):
    """Identifiers for groups in multi-top-k requests."""

    ACT_TIMES_GRAD = "act_times_grad"
    ACTIVATION = "activation"
    DIRECT_WRITE_TO_GRAD = "direct_write_to_grad"
    DIRECTION_WRITE = "direction_write"
    LOGITS = "logits"
    MLP_LAYER_WRITE = "mlp_layer_write"
    # Used in situations where there's only one group.
    SINGLETON = "singleton"
    # Used for projecting write vectors of nodes to token space.
    TOKEN_WRITE = "token_write"
    # Used for projecting read vectors of nodes to token space.
    TOKEN_READ = "token_read"
    WRITE_NORM = "write_norm"
    # Used for token pair attribution requests.
    TOKEN_PAIR_ATTRIBUTION = "token_pair_attribution"

    @property
    def exclude_bottom_k(self) -> bool:
        # if False, top k should return both the top k largest and smallest/(most negative) activations;
        # otherwise, should return the top k largest only. Generally, exclude_bottom_k = True is
        # appropriate for scalars that are non-negative (the values closest to 0 are not particularly interesting).
        # exclude_bottom_k = False is appropriate for scalars that can be positive or negative (the most negative values
        # may be interesting).
        return self in {
            GroupId.WRITE_NORM,
            GroupId.ACTIVATION,
            GroupId.LOGITS,  # logits can be positive or negative, but generally we are interested the most likely
            # tokens to be sampled, which are the most positive logits
        }


########## Tensors ##########


class TensorType(Enum):
    TENSOR_0D = "tensor_0d"
    TENSOR_1D = "tensor_1d"
    TENSOR_2D = "tensor_2d"
    TENSOR_3D = "tensor_3d"


class TorchableTensor(CamelCaseBaseModel):
    tensor_type: TensorType
    value: Any

    def torch(self) -> torch.Tensor:
        return torch.tensor(self.value)


@immutable
class Tensor0D(TorchableTensor):
    tensor_type: TensorType = TensorType.TENSOR_0D
    value: float


@immutable
class Tensor1D(TorchableTensor):
    tensor_type: TensorType = TensorType.TENSOR_1D
    value: list[float]


@immutable
class Tensor2D(TorchableTensor):
    tensor_type: TensorType = TensorType.TENSOR_2D
    value: list[list[float]]


@immutable
class Tensor3D(TorchableTensor):
    tensor_type: TensorType = TensorType.TENSOR_3D
    value: list[list[list[float]]]


TensorND = Union[Tensor0D, Tensor1D, Tensor2D, Tensor3D]


########## Model info ##########


@immutable
class ModelInfoResponse(CamelCaseBaseModel):
    model_name: str | None
    has_mlp_autoencoder: bool
    mlp_autoencoder_name: str | None
    has_attention_autoencoder: bool
    attention_autoencoder_name: str | None
    n_layers: int


########## Derived scalars ##########


@immutable
class DerivedScalarsRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[
        SpecType.DERIVED_SCALARS_REQUEST_SPEC
    ] = SpecType.DERIVED_SCALARS_REQUEST_SPEC
    dst: DerivedScalarType
    layer_index: LayerIndex
    activation_index: int
    normalize_activations_using_neuron_record: NodeIdAndDatasets | None = None
    """
    If non-None, the response will include normalized activations. The max scalar used for
    normalization will be the max scalar in the neuron record specified by the NodeIdAndDatasets.
    """

    pass_type: PassType = PassType.FORWARD
    num_top_tokens: int | None = None
    """
    If non-None, return the top and bottom tokens for the node, according to the scoring
    methodology associated with the derived scalar type.
    """


@immutable
class DerivedScalarsRequest(InferenceRequest):
    derived_scalars_request_spec: DerivedScalarsRequestSpec


@immutable
class DerivedScalarsResponseData(CamelCaseBaseModel):
    response_data_type: ProcessingResponseDataType = (
        ProcessingResponseDataType.DERIVED_SCALARS_RESPONSE_DATA
    )
    activations: list[float]
    normalized_activations: list[float] | None
    """
    The same activations, but normalized to [0, 1] using the max scalar in the specified neuron
    record. Only set if normalize_activations_using_neuron_record was specified in the request.
    """

    node_indices: list[MirroredNodeIndex]
    top_tokens: TopTokens | None
    """
    While this response covers multiple nodes, those nodes differ only in the sequence token index:
    they all correspond to a single component (per go/tdb-terminology). Top tokens are the same for
    all nodes associated with a single component, so we only need to return one set of top tokens
    for the entire component. This will be None if num_top_tokens is None or if the activation was
    zero, preventing the relevant write vector from being computed.
    """


@immutable
class DerivedScalarsResponse(InferenceResponse):
    derived_scalars_response_data: DerivedScalarsResponseData


########## Derived attention scalars ##########


@immutable
class DerivedAttentionScalarsRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[
        SpecType.DERIVED_ATTENTION_SCALARS_REQUEST_SPEC
    ] = SpecType.DERIVED_ATTENTION_SCALARS_REQUEST_SPEC
    dst: DerivedScalarType
    layer_index: LayerIndex
    activation_index: int
    normalize_activations_using_neuron_record: NodeIdAndDatasets | None = None
    """
    If non-None, the response will include normalized activations. The max scalars used for
    normalization will be the max scalars in the neuron record specified by the NodeIdAndDatasets.
    """


@immutable
class DerivedAttentionScalarsRequest(InferenceRequest):
    derived_attention_scalars_request_spec: DerivedAttentionScalarsRequestSpec


@immutable
class DerivedAttentionScalarsResponseData(CamelCaseBaseModel):
    response_data_type: ProcessingResponseDataType = (
        ProcessingResponseDataType.DERIVED_ATTENTION_SCALARS_RESPONSE_DATA
    )
    token_and_attention_scalars_list: list[TokenAndAttentionScalars]


@immutable
class DerivedAttentionScalarsResponse(InferenceResponse):
    derived_attention_scalars_response_data: DerivedAttentionScalarsResponseData


########## (Multi) top-k ##########


# This dataclass is not used in any requests or responses. It's used internally to represent a top-k
# operation performed as part of servicing a MultipleTopKDerivedScalarsRequest.
@immutable
class TopKParams(CamelCaseBaseModel):
    dst_list: list[DerivedScalarType]
    token_index: int | None
    top_and_bottom_k: int | None = None
    pass_type: PassType = PassType.FORWARD
    exclude_bottom_k: bool = False
    dimensions_to_keep_for_intermediate_sum: list[Dimension] = [
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
    ]


@immutable
class MultipleTopKDerivedScalarsRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[
        SpecType.MULTIPLE_TOP_K_DERIVED_SCALARS_REQUEST_SPEC
    ] = SpecType.MULTIPLE_TOP_K_DERIVED_SCALARS_REQUEST_SPEC
    dst_list_by_group_id: dict[GroupId, list[DerivedScalarType]]
    # dsts for each group ID are assumed to have defined node_type,
    # all node_types assumed to be distinct within a group_id, and all group_ids to
    # contain the same set of node_types.
    token_index: int | None
    top_and_bottom_k: int | None = None
    pass_type: PassType = PassType.FORWARD
    dimensions_to_keep_for_intermediate_sum: list[Dimension] = [
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
    ]

    def get_top_k_params_for_group_id(self, group_id: GroupId) -> TopKParams:
        """
        A MultipleTopKDerivedScalarsRequestSpec object contains the information necessary to
        generate multiple TopKParams objects, one for each group ID. This function returns the
        TopKParams for a specific group ID.
        """
        dst_list = self.dst_list_by_group_id[group_id]
        exclude_bottom_k = group_id.exclude_bottom_k

        # Convert the instance to a dictionary
        data = self.dict()

        # Remove the fields that are not needed in TopKDerivedScalarsRequestSpec
        data.pop("dst_list_by_group_id")
        data.pop("spec_type")

        # Add the fields specific to TopKDerivedScalarsRequestSpec
        data["dst_list"] = dst_list
        data["exclude_bottom_k"] = exclude_bottom_k

        return TopKParams(**data)


# All sub-requests within this request must have comparable prompts, since when top-k operations
# within the batch will union over node indices (within each spec name).
@immutable
class MultipleTopKDerivedScalarsRequest(InferenceRequest):
    multiple_top_k_derived_scalars_request_spec: MultipleTopKDerivedScalarsRequestSpec


@immutable
class MultipleTopKDerivedScalarsResponseData(CamelCaseBaseModel):
    response_data_type: ProcessingResponseDataType = (
        ProcessingResponseDataType.MULTIPLE_TOP_K_DERIVED_SCALARS_RESPONSE_DATA
    )
    # Activations associated with top-k nodes for this sub-request, as well as top-k nodes with the
    # same spec name in other (multi) top-k requests in this batched request.
    activations_by_group_id: dict[GroupId, list[float]]
    # Indices for top-k nodes associated with this request, as well as top-k nodes with the same
    # spec name in other (multi) top-k requests in this batched request.
    node_indices: list[MirroredNodeIndex]
    vocab_token_strings_for_indices: list[str | None] | None
    # sum_... entries indicate total of all activations in group, including non-top-k activations
    intermediate_sum_activations_by_dst_by_group_id: dict[
        GroupId, dict[DerivedScalarType, TensorND]
    ]

    @root_validator
    def check_consistency(cls, values: dict[str, Any]) -> dict[str, Any]:
        activations_by_group_id = values.get("activations_by_group_id")
        assert activations_by_group_id is not None
        node_indices = values.get("node_indices")
        assert node_indices is not None
        vocab_token_strings_for_indices = values.get("vocab_token_strings_for_indices")

        for group_id, activations in activations_by_group_id.items():
            assert len(node_indices) == len(activations), (
                f"Expected len(node_indices) == len(activations) for group_id {group_id},"
                f" but got len(node_indices)={len(node_indices)}, len(activations)={len(activations)}"
            )
            assert all(math.isfinite(activation) for activation in activations), (
                f"Expected all activations to be finite for group_id {group_id},"
                f" but got activations={activations}"
            )

        if vocab_token_strings_for_indices is not None:
            assert len(node_indices) == len(vocab_token_strings_for_indices), (
                f"Expected len(node_indices) == len(vocab_token_strings_for_indices),"
                f" but got len(node_indices)={len(node_indices)}, len(vocab_token_strings_for_indices)={len(vocab_token_strings_for_indices)}"
            )
        return values


@immutable
class MultipleTopKDerivedScalarsResponse(InferenceResponse):
    multiple_top_k_derived_scalars_response_data: MultipleTopKDerivedScalarsResponseData


########## Scored tokens ##########


class TokenScoringType(Enum):
    """Methods by which vocab tokens may be scored."""

    # Score tokens by the degree to which this node directly upvotes them. This is basically the
    # "logit lens".
    UPVOTED_OUTPUT_TOKENS = "upvoted_output_tokens"
    # Score tokens by the degree to which they directly upvote this node. Three flavors, each of
    # which applies to both "raw" components like neurons and attention heads, as well as
    # autoencoder latents:
    #  1) Upvoting MLP nodes
    #  2) Upvoting the Q part of attention nodes
    #  3) Upvoting the K part of attention nodes
    INPUT_TOKENS_THAT_UPVOTE_MLP = "input_tokens_that_upvote_mlp"
    INPUT_TOKENS_THAT_UPVOTE_ATTN_Q = "input_tokens_that_upvote_attn_q"
    INPUT_TOKENS_THAT_UPVOTE_ATTN_K = "input_tokens_that_upvote_attn_k"


@immutable
class ScoredTokensRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[SpecType.SCORED_TOKENS_REQUEST_SPEC] = SpecType.SCORED_TOKENS_REQUEST_SPEC

    # How tokens should be scored.
    token_scoring_type: TokenScoringType
    # A value of e.g. 10 means 10 top and 10 bottom tokens.
    num_tokens: int
    # Which nodes do we want to get scored tokens for, and which DSTs and DST configs should we use?
    # This request spec refers to another request spec and grabs those values from it.
    depends_on_spec_name: str


@immutable
class ScoredTokensRequest(InferenceRequest):
    scored_tokens_request_spec: ScoredTokensRequestSpec


@immutable
class ScoredTokensResponseData(CamelCaseBaseModel):
    response_data_type: ProcessingResponseDataType = (
        ProcessingResponseDataType.SCORED_TOKENS_RESPONSE_DATA
    )
    # These two lists are parallel and have the same length. "None" values in top_tokens_list
    # indicate that the specified TokenScoringType does not apply to the corresponding node.
    node_indices: list[MirroredNodeIndex]
    top_tokens_list: list[TopTokens | None]


@immutable
class ScoredTokensResponse(InferenceResponse):
    scored_tokens_response_data: ScoredTokensResponseData


########## TDB-specific ##########


class ComponentTypeForMlp(Enum):
    """The type of component / fundamental unit to use for MLP layers.

    This determines which types of node appear in the node table to represent the MLP layers.
    Neurons are the fundamental unit of MLP layers, but autoencoder latents are more interpretable.
    """

    NEURON = "neuron"
    AUTOENCODER_LATENT = "autoencoder_latent"


class ComponentTypeForAttention(Enum):
    """The type of component / fundamental unit to use for Attention layers.

    This determines which types of node appear in the node table to represent the Attention layers.
    Heads are the fundamental unit of Attention layers, but autoencoder latents are more interpretable.
    """

    ATTENTION_HEAD = "attention_head"
    AUTOENCODER_LATENT = "autoencoder_latent"


@immutable
class TdbRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[SpecType.TDB_REQUEST_SPEC] = SpecType.TDB_REQUEST_SPEC

    prompt: str
    target_tokens: list[str]
    distractor_tokens: list[str]
    component_type_for_mlp: ComponentTypeForMlp
    """Whether to use neurons or autoencoder latents as the basic unit for MLP layers."""

    component_type_for_attention: ComponentTypeForAttention
    """Whether to use heads or autoencoder latents as the basic unit for attention layers."""

    top_and_bottom_k_for_node_table: int
    """The number of top and bottom nodes to calculate for each column in the node table."""

    hide_early_layers_when_ablating: bool
    """Whether to exclude layers before the first ablated layer from the results."""

    node_ablations: list[NodeAblation] | None
    upstream_node_to_trace: NodeToTrace | None
    """The primary node at which a gradient is being computed"""

    downstream_node_to_trace: NodeToTrace | None
    """In the case where upstream_node_to_trace is an attention value subnode, you can also
    provide a downstream node to trace. A gradient is first computed with respect to this downstream
    node, and then the direct effect of the upstream node on this gradient direction is computed. A
    gradient is then computed with respect to that quantity, propagated back to upstream activations.
    In the case where no downstream node is provided, the loss is used as the "downstream node"."""


@immutable
class BatchedTdbRequest(CamelCaseBaseModel):
    sub_requests: list[TdbRequestSpec]


########## Attribution ##########


@immutable
class TopTokensAttendedTo(CamelCaseBaseModel):
    token_indices: list[int]  # in sequence
    attributions: list[float]


@immutable
class TokenPairAttributionRequestSpec(CamelCaseBaseModel):
    # note: the spec_type field is not to be populated by the user at __init__, but is
    # required for pydantic to distinguish between different XRequestSpec classes
    spec_type: Literal[
        SpecType.TOKEN_PAIR_ATTRIBUTION_REQUEST_SPEC
    ] = SpecType.TOKEN_PAIR_ATTRIBUTION_REQUEST_SPEC

    num_tokens_attended_to: int

    # Which nodes do we want to get scored tokens for, and which DSTs and DST configs should we use?
    # This request spec refers to another request spec and grabs those values from it.
    depends_on_spec_name: str


@immutable
class TokenPairAttributionRequest(InferenceRequest):
    token_pair_attribution_request_spec: TokenPairAttributionRequestSpec


@immutable
class TokenPairAttributionResponseData(CamelCaseBaseModel):
    response_data_type: ProcessingResponseDataType = (
        ProcessingResponseDataType.TOKEN_PAIR_ATTRIBUTION_RESPONSE_DATA
    )
    # These two lists are parallel and have the same length. "None" values in top_tokens_attended_to_list
    # indicate that token-pair attribution does not apply to the corresponding node.
    node_indices: list[MirroredNodeIndex]
    top_tokens_attended_to_list: list[TopTokensAttendedTo | None]


@immutable
class TokenPairAttributionResponse(InferenceResponse):
    token_pair_attribution_response_data: TokenPairAttributionResponseData


########## Batching ##########

# Order from most to least specific
# See https://docs.pydantic.dev/1.10/usage/types/#unions
ProcessingRequestSpec = Union[
    MultipleTopKDerivedScalarsRequestSpec,
    DerivedScalarsRequestSpec,
    DerivedAttentionScalarsRequestSpec,
    ScoredTokensRequestSpec,
    TokenPairAttributionRequestSpec,
]

# Order from most to least specific
# See https://docs.pydantic.dev/1.10/usage/types/#unions
ProcessingResponseData = Union[
    MultipleTopKDerivedScalarsResponseData,
    DerivedScalarsResponseData,
    DerivedAttentionScalarsResponseData,
    ScoredTokensResponseData,
    TokenPairAttributionResponseData,
]


@immutable
class InferenceSubRequest(CamelCaseBaseModel):
    inference_request_spec: InferenceRequestSpec
    processing_request_spec_by_name: dict[str, ProcessingRequestSpec] = {}


@immutable
class InferenceResponseAndResponseDict(CamelCaseBaseModel):
    inference_response: InferenceResponse
    processing_response_data_by_name: dict[str, ProcessingResponseData] = {}


@immutable
class BatchedRequest(CamelCaseBaseModel):
    inference_sub_requests: list[InferenceSubRequest]


@immutable
class BatchedResponse(CamelCaseBaseModel):
    inference_sub_responses: list[InferenceResponseAndResponseDict]
