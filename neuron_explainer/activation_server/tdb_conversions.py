# Code for converting between client-friendly TDB request/response dataclasses and internal
# representations used during request processing.

from typing import TypeVar

from neuron_explainer.activation_server.requests_and_responses import *
from neuron_explainer.activation_server.requests_and_responses import GroupId
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    DETACH_LAYER_NORM_SCALE,
    AblationSpec,
    AttentionTraceType,
    MirroredActivationIndex,
    MirroredNodeIndex,
    MirroredTraceConfig,
    NodeAblation,
    NodeToTrace,
    PreOrPostAct,
    TraceConfig,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    NodeType,
    PassType,
)

T = TypeVar("T")


def convert_tdb_request_spec_to_inference_sub_request(
    tdb_request_spec: TdbRequestSpec,
) -> InferenceSubRequest:
    """
    The client sends a TdbRequestSpec, but internally we do all processing in terms of
    InferenceSubRequests. This function converts from the client representation to the server
    representation.
    """
    loss_fn_config: LossFnConfig | None = LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=tdb_request_spec.target_tokens,
        distractor_tokens=tdb_request_spec.distractor_tokens,
    )
    ablation_specs = [
        node_ablation_to_ablation_spec(ablation)
        for ablation in (tdb_request_spec.node_ablations or [])
    ] + [
        AblationSpec(
            index=MirroredActivationIndex(
                activation_location_type=ActivationLocationType.RESID_FINAL_LAYER_NORM_SCALE,
                pass_type=PassType.BACKWARD,
                tensor_indices=("All", "All"),  # ablate at all positions in the sequence
                layer_index=None,
            ),
            value=0,
        )
    ]
    current_token_index = -1
    trace_config = None
    if tdb_request_spec.upstream_node_to_trace is None:
        assert tdb_request_spec.downstream_node_to_trace is None
    else:
        (
            trace_config,
            trace_token_index,
        ) = nodes_to_trace_to_trace_config(
            tdb_request_spec.upstream_node_to_trace, tdb_request_spec.downstream_node_to_trace
        )
        if trace_token_index is not None:
            current_token_index = trace_token_index

    if trace_config is None:  # not tracing -> DO compute loss
        pass
    elif trace_config.attention_trace_type == AttentionTraceType.V:  # tracing attention through V
        if trace_config.downstream_trace_config is None:
            pass  # tracing through V with no downstream trace -> DO compute loss
        else:
            loss_fn_config = (
                None  # tracing through V, but also with downstream trace -> DON'T compute loss
            )
    else:
        loss_fn_config = (
            None  # tracing something other than attention through V -> DON'T compute loss
        )

    inference_request_spec = InferenceRequestSpec(
        prompt=tdb_request_spec.prompt,
        loss_fn_config=loss_fn_config,
        ablation_specs=ablation_specs,
        trace_config=MirroredTraceConfig.from_trace_config(trace_config) if trace_config else None,
    )

    spec_by_component_for_top_k = MultipleTopKDerivedScalarsRequestSpec(
        token_index=None,
        dst_list_by_group_id=make_grouped_dsts_per_component(
            tdb_request_spec.component_type_for_mlp,
            tdb_request_spec.component_type_for_attention,
        ),
        top_and_bottom_k=tdb_request_spec.top_and_bottom_k_for_node_table,
    )

    spec_by_component_always_mlp_for_token_display = MultipleTopKDerivedScalarsRequestSpec(
        token_index=None,
        # the response to this request is to be used for summarizing the effects of entire
        # attention and MLP layers per token; thus, using a different basis for the activations
        # within a layer is not helpful, and we use MLP activations themselves.
        dst_list_by_group_id=make_grouped_dsts_per_component(
            ComponentTypeForMlp.NEURON, ComponentTypeForAttention.ATTENTION_HEAD
        ),
        top_and_bottom_k=1,
        dimensions_to_keep_for_intermediate_sum=[
            Dimension.SEQUENCE_TOKENS,
            Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        ],
    )

    def scored_tokens_request_spec(
        token_scoring_type: TokenScoringType,
    ) -> ScoredTokensRequestSpec:
        return ScoredTokensRequestSpec(
            token_scoring_type=token_scoring_type,
            num_tokens=10,
            # Our scored tokens requests are associated with the "topKComponents" request spec. This
            # means that they use the same node indices, DSTs and DST configs.
            depends_on_spec_name="topKComponents",
        )

    def token_pair_attribution_request_spec() -> TokenPairAttributionRequestSpec:
        return TokenPairAttributionRequestSpec(
            num_tokens_attended_to=3,
            # Our scored tokens requests are associated with the "topKComponents" request spec. This
            # means that they use the same node indices, DSTs and DST configs.
            depends_on_spec_name="topKComponents",
        )

    processing_request_spec_by_name: dict[str, ProcessingRequestSpec] = {
        "topKComponents": spec_by_component_for_top_k,
        "componentSumsForTokenDisplay": spec_by_component_always_mlp_for_token_display,
        # It's important for these request specs to come after the "topKComponents" request spec,
        # since they depend on data generated for that request spec.
        "upvotedOutputTokens": scored_tokens_request_spec(TokenScoringType.UPVOTED_OUTPUT_TOKENS),
        "inputTokensThatUpvoteMlp": scored_tokens_request_spec(
            TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP
        ),
        "inputTokensThatUpvoteAttnQ": scored_tokens_request_spec(
            TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_Q
        ),
        "inputTokensThatUpvoteAttnK": scored_tokens_request_spec(
            TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_K
        ),
        "tokenPairAttribution": token_pair_attribution_request_spec(),
    }

    spec_by_vocab_token = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id={GroupId.LOGITS: [DerivedScalarType.LOGITS]},
        top_and_bottom_k=100,
        token_index=current_token_index,
    )
    processing_request_spec_by_name["topOutputTokenLogits"] = spec_by_vocab_token

    return InferenceSubRequest(
        inference_request_spec=inference_request_spec,
        processing_request_spec_by_name=processing_request_spec_by_name,
    )


def node_ablation_to_ablation_spec(node_ablation: NodeAblation) -> AblationSpec:
    node_index = node_ablation.node_index
    value = node_ablation.value

    match node_index.node_type:
        case NodeType.ATTENTION_HEAD:
            activation_location_type = ActivationLocationType.ATTN_QK_PROBS
            indices = [
                get_sequence_token_index(node_index),
                "All",
                get_activation_index(node_index),
            ]
        case NodeType.MLP_NEURON:
            activation_location_type = ActivationLocationType.MLP_POST_ACT
            indices = [
                get_sequence_token_index(node_index),
                get_activation_index(node_index),
            ]
        case (
            NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            from neuron_explainer.activations.derived_scalars.autoencoder import (
                get_autoencoder_alt_from_node_type,
            )

            activation_location_type = get_autoencoder_alt_from_node_type(node_index.node_type)

            indices = [
                get_sequence_token_index(node_index),
                get_activation_index(node_index),
            ]
        case _:
            raise ValueError(f"Unknown node type {node_index.node_type}")

    return AblationSpec(
        index=MirroredActivationIndex(
            activation_location_type=activation_location_type,
            pass_type=PassType.FORWARD,
            # mypy has trouble understanding that all of the values that can be assigned to indices
            # match AllOrOneIndices.
            tensor_indices=indices,  # type: ignore
            layer_index=node_index.layer_index,
        ),
        value=value,
    )


def get_sequence_token_index(node_index: MirroredNodeIndex) -> int:
    return assert_non_none(node_index.tensor_indices[0])


def get_activation_index(node_index: MirroredNodeIndex) -> int:
    return assert_non_none(node_index.tensor_indices[-1])


def assert_non_none(value: T | None) -> T:
    assert value is not None
    return value


def make_grouped_dsts_per_component(
    component_type_for_mlp: ComponentTypeForMlp,
    component_type_for_attention: ComponentTypeForAttention,
) -> dict[GroupId, list[DerivedScalarType]]:
    # common dsts for all components
    dsts = {
        GroupId.WRITE_NORM: [
            DerivedScalarType.RESID_POST_EMBEDDING_NORM,
        ],
        GroupId.ACT_TIMES_GRAD: [
            DerivedScalarType.TOKEN_ATTRIBUTION,
        ],
        GroupId.DIRECTION_WRITE: [
            DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD,
        ],
        GroupId.ACTIVATION: [
            DerivedScalarType.ALWAYS_ONE,  # the resid post embedding is considered to have an
            # "activation" of 1.0 at every position, for display purposes
        ],
    }

    match component_type_for_mlp:
        case ComponentTypeForMlp.NEURON:
            dsts[GroupId.WRITE_NORM].append(DerivedScalarType.MLP_WRITE_NORM)
            dsts[GroupId.ACT_TIMES_GRAD].append(DerivedScalarType.MLP_ACT_TIMES_GRAD)
            dsts[GroupId.DIRECTION_WRITE].append(DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD)
            dsts[GroupId.ACTIVATION].append(DerivedScalarType.MLP_POST_ACT)
        case ComponentTypeForMlp.AUTOENCODER_LATENT:
            dsts[GroupId.WRITE_NORM].append(DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_NORM)
            dsts[GroupId.ACT_TIMES_GRAD].append(
                DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD
            )
            dsts[GroupId.DIRECTION_WRITE].append(
                DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD
            )
            dsts[GroupId.ACTIVATION].append(DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT)
        case _:
            raise ValueError(f"Unknown component type {component_type_for_mlp} in TdbRequestSpec")

    match component_type_for_attention:
        case ComponentTypeForAttention.ATTENTION_HEAD:
            dsts[GroupId.WRITE_NORM].append(DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM)
            dsts[GroupId.ACT_TIMES_GRAD].append(DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD)
            dsts[GroupId.DIRECTION_WRITE].append(
                DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD
            )
            dsts[GroupId.ACTIVATION].append(DerivedScalarType.ATTN_QK_PROBS)
        case ComponentTypeForAttention.AUTOENCODER_LATENT:
            dsts[GroupId.WRITE_NORM].append(
                DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM
            )
            dsts[GroupId.ACT_TIMES_GRAD].append(
                DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD
            )
            dsts[GroupId.DIRECTION_WRITE].append(
                DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD
            )
            dsts[GroupId.ACTIVATION].append(DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT)
        case _:
            raise ValueError(
                f"Unknown component type {component_type_for_attention} in TdbRequestSpec"
            )

    return dsts


def nodes_to_trace_to_trace_config(
    upstream_node_to_trace: NodeToTrace,
    downstream_node_to_trace: NodeToTrace | None,
) -> tuple[TraceConfig, int | None]:
    node_index = upstream_node_to_trace.node_index
    attention_trace_type = upstream_node_to_trace.attention_trace_type
    if downstream_node_to_trace is None:
        downstream_trace_config = None
    else:
        # only trace through V admits a downstream node to trace
        assert attention_trace_type == AttentionTraceType.V
        # don't assign downstream trace_token_index to a variable, as it's not used
        downstream_trace_config, _ = nodes_to_trace_to_trace_config(
            downstream_node_to_trace,
            None,  # treat the downstream node as the "upstream" node to trace
        )  # NOTE: downstream node must not trace through V
    trace_token_index = node_index.tensor_indices[0]
    return (
        TraceConfig(
            node_index=node_index.to_node_index(),
            pre_or_post_act=PreOrPostAct.PRE,
            attention_trace_type=attention_trace_type,
            downstream_trace_config=downstream_trace_config,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
        ),
        trace_token_index,
    )


def named_attention_head_indices(node_index: MirroredNodeIndex) -> tuple[int, int, int]:
    if node_index.node_type != NodeType.ATTENTION_HEAD:
        raise ValueError("Incorrect nodeType for namedAttentionHeadIndices function")
    (
        attended_from_token_index,
        attended_to_token_index,
        attention_head_index,
    ) = node_index.tensor_indices
    return (
        assert_non_none(attended_from_token_index),
        assert_non_none(attended_to_token_index),
        assert_non_none(attention_head_index),
    )
