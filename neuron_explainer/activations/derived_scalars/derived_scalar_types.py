from enum import Enum

from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    LocationWithinLayer,
    NodeType,
    PassType,
    node_type_by_dimension,
)

### ENUM; ADD NEW TYPES HERE, AND ALSO IN REGISTRIES shape_spec_per_token_sequence_by_dst
# in this file and _DERIVED_SCALAR_TYPE_REGISTRY in make_scalar_derivers.py ###


# Note: activation server client libraries should be regenerated after editing this enum, which is
# mirrored in typescript. See neuron_explainer/activation_server/README.md to learn how to
# regenerate them.
class DerivedScalarType(str, Enum):
    """
    List of implemented derived location types. When implementing a new one, add a make_<dst>
    function to _DERIVED_SCALAR_TYPE_REGISTRY in make_scalar_derivers.py and add the name of the
    derived location type to this enum.

    If implementing a new HookLocationType, also add its DerivedScalarType (trivially computed from
    the activations) to this enum, and add a row like this to the registry:
    DerivedScalarType.NEW_HOOK_LOCATION_TYPE: make_scalar_deriver_factory_for_hook_location_type(
        "new_hook_location_type"
    )

    Activations of DerivedScalarTypes for a given token sequence either have one or two dimensions
    indexed by tokens:
    1. sequence_tokens: all activations have as their first dimension the number of tokens in the
       sequence
    2. attended_to_sequence_tokens: pre- and post-softmax attention, and other activations derived
       from those, have an additional token dimension, corresponding to "attended to" tokens
       (sequence_tokens being the "attended from" tokens).

    The token dimensions can in general be represented as a (num_sequence_tokens,
    num_sequence_tokens) matrix, but in some settings this might be represented as a non-square
    matrix (num_sequence_tokens != num_attended_to_sequence_tokens), e.g. if there are irrelevant
    padding tokens we wish to leave out.

    For DerivedScalarTypes not using attended_to_sequence_tokens, a
    num_attended_to_sequence_tokens=None argument still gets passed around in computing the expected
    shape of the activations. This argument gets ignored.
    """

    # correspond 1:1 with HookLocationTypes
    LOGITS = "logits"
    RESID_POST_EMBEDDING = "resid_post_embedding"
    MLP_PRE_ACT = "mlp_pre_act"
    MLP_POST_ACT = "mlp_post_act"
    RESID_DELTA_MLP = "resid_delta_mlp"
    RESID_POST_MLP = "resid_post_mlp"
    ATTN_QUERY = "attn_query"
    ATTN_KEY = "attn_key"
    ATTN_VALUE = "attn_value"
    ATTN_QK_LOGITS = "attn_qk_logits"  # uses attended_to_sequence_tokens, with 2 token dimensions
    ATTN_QK_PROBS = "attn_qk_probs"  # uses attended_to_sequence_tokens, with 2 token dimensions
    ATTN_WEIGHTED_SUM_OF_VALUES = "attn_weighted_sum_of_values"
    RESID_DELTA_ATTN = "resid_delta_attn"
    RESID_POST_ATTN = "resid_post_attn"
    RESID_FINAL_LAYER_NORM_SCALE = "resid_final_layer_norm_scale"
    ATTN_INPUT_LAYER_NORM_SCALE = "attn_input_layer_norm_scale"
    MLP_INPUT_LAYER_NORM_SCALE = "mlp_input_layer_norm_scale"

    # additional hooks
    ONLINE_AUTOENCODER_LATENT = "online_autoencoder_latent"

    # derived from HookLocationTypes
    ATTN_WRITE_NORM = "attn_write_norm"  # uses attended_to_sequence_tokens, 1 token dimension, a flattened representation of lower triangle of 2D attention matrix
    FLATTENED_ATTN_POST_SOFTMAX = "flattened_attn_post_softmax"  # uses attended_to_sequence_tokens, 1 token dimension, a flattened representation of lower triangle of 2D attention matrix
    ATTN_ACT_TIMES_GRAD = "attn_act_times_grad"  # uses attended_to_sequence_tokens, 1 token dimension, a flattened representation of lower triangle of 2D attention matrix
    RESID_DELTA_MLP_FROM_MLP_POST_ACT = "resid_delta_mlp_from_mlp_post_act"  # aka "mlp_write"
    MLP_WRITE_NORM = "mlp_write_norm"
    MLP_ACT_TIMES_GRAD = "mlp_act_times_grad"
    AUTOENCODER_LATENT = "autoencoder_latent"
    AUTOENCODER_WRITE_NORM = "autoencoder_write_norm"
    MLP_WRITE_TO_FINAL_RESIDUAL_GRAD = "mlp_write_to_final_residual_grad"
    ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN = "attn_write_norm_per_sequence_token"
    ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN = (
        "attn_write_to_final_residual_grad_per_sequence_token"
    )
    ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN = "attn_act_times_grad_per_sequence_token"
    RESID_POST_EMBEDDING_NORM = "resid_post_embedding_norm"
    RESID_POST_MLP_NORM = "resid_post_mlp_norm"
    MLP_LAYER_WRITE_NORM = "mlp_layer_write_norm"  # could also be called RESID_DELTA_MLP_NORM
    RESID_POST_ATTN_NORM = "resid_post_attn_norm"
    ATTN_LAYER_WRITE_NORM = "attn_layer_write_norm"  # could also be called RESID_DELTA_ATTN_NORM
    RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD = (
        "resid_post_embedding_proj_to_final_residual_grad"
    )
    RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD = "resid_post_mlp_proj_to_final_residual_grad"
    MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD = "mlp_layer_write_to_final_residual_grad"  # could also be called RESID_DELTA_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD
    RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD = "resid_post_attn_proj_to_final_residual_grad"
    ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD = "attn_layer_write_to_final_residual_grad"  # could also be called RESID_DELTA_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD
    UNFLATTENED_ATTN_ACT_TIMES_GRAD = "unflattened_attn_act_times_grad"
    UNFLATTENED_ATTN_WRITE_NORM = "unflattened_attn_write_norm"
    UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD = "unflattened_attn_write_to_final_residual_grad"
    ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD = "attn_write_to_final_residual_grad"
    ONLINE_AUTOENCODER_ACT_TIMES_GRAD = "online_autoencoder_act_times_grad"
    ONLINE_AUTOENCODER_WRITE_NORM = "online_autoencoder_write_norm"
    ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = (
        "online_autoencoder_write_to_final_residual_grad"
    )
    ONLINE_MLP_AUTOENCODER_ERROR = "online_mlp_autoencoder_error"
    ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR = "online_residual_mlp_autoencoder_error"
    ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR = "online_residual_attention_autoencoder_error"
    ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD = "online_mlp_autoencoder_error_act_times_grad"
    ONLINE_MLP_AUTOENCODER_ERROR_WRITE_NORM = "online_mlp_autoencoder_error_write_norm"
    ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD = (
        "online_mlp_autoencoder_error_write_to_final_residual_grad"
    )
    ATTN_WRITE = "attn_write"
    ATTN_WRITE_SUM_HEADS = "attn_write_sum_heads"
    MLP_WRITE = "mlp_write"
    ONLINE_AUTOENCODER_WRITE = "online_autoencoder_write"
    ATTN_WEIGHTED_VALUE = "attn_weighted_value"
    PREVIOUS_LAYER_RESID_POST_MLP = "previous_layer_resid_post_mlp"
    MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "mlp_write_to_final_activation_residual_grad"
    UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = (
        "unflattened_attn_write_to_final_activation_residual_grad"
    )
    ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = (
        "online_autoencoder_write_to_final_activation_residual_grad"
    )
    RESID_POST_EMBEDDING_PROJ_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = (
        "resid_post_embedding_proj_to_final_activation_residual_grad"
    )
    # DSTs using gradient of autoencoder latent wrt input
    AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT = "autoencoder_latent_grad_wrt_residual_input"
    AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT = (
        "autoencoder_latent_grad_wrt_mlp_post_act_input"
    )
    ATTN_WRITE_TO_LATENT = "attn_write_to_latent"
    ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS = "attn_write_to_latent_summed_over_heads"
    FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS = (
        "flattened_attn_write_to_latent_summed_over_heads"
    )
    FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED = (
        "flattened_attn_write_to_latent_summed_over_heads_batched"
    )
    ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN = "attn_write_to_latent_per_sequence_token"
    ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED = (
        "attn_write_to_latent_per_sequence_token_batched"
    )
    TOKEN_ATTRIBUTION = "token_attribution"
    SINGLE_NODE_WRITE = "single_node_write"
    GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION = "grad_of_single_subnode_attribution"
    ATTN_OUT_EDGE_ATTRIBUTION = "attn_out_edge_attribution"
    MLP_OUT_EDGE_ATTRIBUTION = "mlp_out_edge_attribution"
    ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION = "online_autoencoder_out_edge_attribution"
    ATTN_QUERY_IN_EDGE_ATTRIBUTION = "attn_query_in_edge_attribution"
    ATTN_KEY_IN_EDGE_ATTRIBUTION = "attn_key_in_edge_attribution"
    ATTN_VALUE_IN_EDGE_ATTRIBUTION = "attn_value_in_edge_attribution"
    MLP_IN_EDGE_ATTRIBUTION = "mlp_in_edge_attribution"
    ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION = "online_autoencoder_in_edge_attribution"
    TOKEN_OUT_EDGE_ATTRIBUTION = "token_out_edge_attribution"
    SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD = "single_node_write_to_final_residual_grad"
    # compute the pre-activation of the node as if the token embedding vector were the input to the layer
    VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION = "vocab_token_write_to_input_direction"
    # this is a placeholder DST which has one value per layer and is always 1.0. It is used
    # for threading n/a values through activation server code, e.g. in the case of "activations"
    # of token/position embedding nodes.
    ALWAYS_ONE = "always_one"
    ATTN_QUERY_IN_EDGE_ACTIVATION = "attn_query_in_edge_activation"
    ATTN_KEY_IN_EDGE_ACTIVATION = "attn_key_in_edge_activation"
    MLP_IN_EDGE_ACTIVATION = "mlp_in_edge_activation"
    ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION = "online_autoencoder_in_edge_activation"

    MLP_AUTOENCODER_LATENT = "mlp_autoencoder_latent"
    MLP_AUTOENCODER_WRITE_NORM = "mlp_autoencoder_write_norm"
    ONLINE_MLP_AUTOENCODER_LATENT = "online_mlp_autoencoder_latent"
    ONLINE_MLP_AUTOENCODER_WRITE = "online_mlp_autoencoder_write"
    ONLINE_MLP_AUTOENCODER_WRITE_NORM = "online_mlp_autoencoder_write_norm"
    ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD = "online_mlp_autoencoder_act_times_grad"
    ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = (
        "online_mlp_autoencoder_write_to_final_residual_grad"
    )
    ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = (
        "online_mlp_autoencoder_write_to_final_activation_residual_grad"
    )
    ATTENTION_AUTOENCODER_LATENT = "attention_autoencoder_latent"
    ATTENTION_AUTOENCODER_WRITE_NORM = "attention_autoencoder_write_norm"
    ONLINE_ATTENTION_AUTOENCODER_LATENT = "online_attention_autoencoder_latent"
    ONLINE_ATTENTION_AUTOENCODER_WRITE = "online_attention_autoencoder_write"
    ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM = "online_attention_autoencoder_write_norm"
    ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD = "online_attention_autoencoder_act_times_grad"
    ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = (
        "online_attention_autoencoder_write_to_final_residual_grad"
    )
    ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = (
        "online_attention_autoencoder_write_to_final_activation_residual_grad"
    )

    @property
    def is_raw_activation_type(self) -> bool:
        return self in {
            DerivedScalarType.RESID_POST_EMBEDDING,
            DerivedScalarType.RESID_DELTA_MLP,
            DerivedScalarType.RESID_POST_MLP,
            DerivedScalarType.MLP_PRE_ACT,
            DerivedScalarType.MLP_POST_ACT,
            DerivedScalarType.ATTN_QUERY,
            DerivedScalarType.ATTN_KEY,
            DerivedScalarType.ATTN_VALUE,
            DerivedScalarType.ATTN_QK_LOGITS,
            DerivedScalarType.ATTN_QK_PROBS,
            DerivedScalarType.ATTN_WEIGHTED_SUM_OF_VALUES,
            DerivedScalarType.RESID_DELTA_ATTN,
            DerivedScalarType.RESID_POST_ATTN,
            DerivedScalarType.RESID_FINAL_LAYER_NORM_SCALE,
            DerivedScalarType.ATTN_INPUT_LAYER_NORM_SCALE,
            DerivedScalarType.MLP_INPUT_LAYER_NORM_SCALE,
            DerivedScalarType.LOGITS,
        }

    @property
    def is_autoencoder_latent(self) -> bool:
        return self in {
            DerivedScalarType.AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
            DerivedScalarType.MLP_AUTOENCODER_LATENT,
            DerivedScalarType.MLP_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
            DerivedScalarType.ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.ATTENTION_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
        }

    def update_from_autoencoder_node_type(self, node_type: NodeType | None) -> "DerivedScalarType":
        """
        When multiple autoencoders are used, the DST needs to be specific to the autoencoder type.
        This function updates the DST to be specific to the autoencoder type:
            - NodeType.AUTOENCODER_LATENT: default autoencoder, used when no specific autoencoder type is specified
            - NodeType.MLP_AUTOENCODER_LATENT: autoencoder trained on activations from an MLP layer
            - NodeType.ATTENTION_AUTOENCODER_LATENT: autoencoder trained on activations from an Attention layer
        """
        node_type = node_type or NodeType.AUTOENCODER_LATENT
        assert node_type.is_autoencoder_latent
        new_dst_by_node_type = {
            DerivedScalarType.AUTOENCODER_LATENT: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.AUTOENCODER_LATENT,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.MLP_AUTOENCODER_LATENT,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ATTENTION_AUTOENCODER_LATENT,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_LATENT: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_WRITE,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE,
            },
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_NORM,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM,
            },
            DerivedScalarType.AUTOENCODER_WRITE_NORM: {
                NodeType.AUTOENCODER_LATENT: DerivedScalarType.AUTOENCODER_WRITE_NORM,
                NodeType.MLP_AUTOENCODER_LATENT: DerivedScalarType.MLP_AUTOENCODER_WRITE_NORM,
                NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ATTENTION_AUTOENCODER_WRITE_NORM,
            },
        }[self]

        return new_dst_by_node_type[node_type]

    @property
    def node_type(self) -> NodeType:
        """
        The last index of a tensor of derived scalars can correspond to a type of object in the
        network called a 'node'. This can be an MLP neuron, an attention head, an autoencoder
        latent, etc. If we don't yet have a name for the last dimension of a derived scalar type,
        this throws an error.
        """
        if self.is_autoencoder_latent:
            if "mlp" in self.value:
                return NodeType.MLP_AUTOENCODER_LATENT
            elif "attention" in self.value:
                return NodeType.ATTENTION_AUTOENCODER_LATENT
            else:
                return NodeType.AUTOENCODER_LATENT
        last_dimension = self.shape_spec_per_token_sequence[-1]
        if last_dimension in node_type_by_dimension:
            return node_type_by_dimension[last_dimension]
        else:
            raise NotImplementedError(f"Unknown node type for {last_dimension=}")

    @classmethod
    def from_activation_location_type(
        cls, activation_location_type: ActivationLocationType
    ) -> "DerivedScalarType":
        if activation_location_type.name in direct_mapping_alt_and_dst:
            return getattr(DerivedScalarType, activation_location_type.name)
        else:
            raise ValueError(
                f"{activation_location_type} does not have a corresponding DerivedScalarType"
            )

    def to_activation_location_type(self) -> ActivationLocationType:
        if self.name in direct_mapping_alt_and_dst:
            return getattr(ActivationLocationType, self.name)
        elif self == DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT:
            return ActivationLocationType.RESID_DELTA_MLP
        else:
            raise ValueError(f"{self=} does not have a corresponding ActivationLocationType")

    @property
    def shape_spec_per_token_sequence(self) -> tuple[Dimension, ...]:
        shape_spec_per_token_sequence = shape_spec_per_token_sequence_by_dst[self]
        if self.is_raw_activation_type:
            activation_shape_spec_per_token_sequence = (
                self.to_activation_location_type().shape_spec_per_token_sequence
            )
            assert shape_spec_per_token_sequence == activation_shape_spec_per_token_sequence, (
                f"{shape_spec_per_token_sequence=} != "
                f"{activation_shape_spec_per_token_sequence=}"
            )
        return shape_spec_per_token_sequence

    @property
    def ndim_per_token_sequence(self) -> int:
        return len(self.shape_spec_per_token_sequence)

    @property
    def sequence_dim_is_sequence_token_pairs(self) -> bool:
        # If True, the sequence dimension is a flattened representation of lower triangle of 2D
        # attention matrix.
        return self in {
            DerivedScalarType.ATTN_WRITE_NORM,
            DerivedScalarType.FLATTENED_ATTN_POST_SOFTMAX,
            DerivedScalarType.ATTN_ACT_TIMES_GRAD,
        }

    @property
    def requires_grad_for_forward_pass(self) -> bool:
        return self in {
            DerivedScalarType.MLP_ACT_TIMES_GRAD,
            DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ATTN_ACT_TIMES_GRAD,
            DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN,
            DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN,
            DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.TOKEN_ATTRIBUTION,
            DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION,
            DerivedScalarType.ATTN_OUT_EDGE_ATTRIBUTION,
            DerivedScalarType.MLP_OUT_EDGE_ATTRIBUTION,
            DerivedScalarType.ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION,
            DerivedScalarType.ATTN_QUERY_IN_EDGE_ATTRIBUTION,
            DerivedScalarType.ATTN_KEY_IN_EDGE_ATTRIBUTION,
            DerivedScalarType.ATTN_VALUE_IN_EDGE_ATTRIBUTION,
            DerivedScalarType.MLP_IN_EDGE_ATTRIBUTION,
            DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION,
            DerivedScalarType.TOKEN_OUT_EDGE_ATTRIBUTION,
            DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD,
        }

    def requires_grad_for_pass_type(self, pass_type: PassType) -> bool:
        if pass_type == PassType.BACKWARD:
            return True
        else:
            return self.requires_grad_for_forward_pass

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        # returns the LocationWithinLayer if it's inferrable from the DST alone
        # None means ambiguous based on this information; must be specified by DstConfig
        if self.is_raw_activation_type:
            activation_location_type = self.to_activation_location_type()
            return activation_location_type.location_within_layer
        else:
            return self.node_type.location_within_layer

    @property
    def has_no_layers(self) -> bool:
        if self.is_raw_activation_type:
            activation_location_type = self.to_activation_location_type()
            return activation_location_type.has_no_layers
        else:
            raise NotImplementedError(
                "has_no_layers not implemented for nontrivial DSTs (i.e. with non-raw activations)"
            )

    def __repr__(self) -> str:
        return f"'{self.value}'"


# ActivationLocationType and DerivedScalarType with the same name and a direct one-to-one mapping.
# Note that the names are the same, but the values can be different, for example:
# DerivedScalarType.ATTN_QUERY = "attn_query" and ActivationLocationType.ATTN_QUERY = "attn.q"
direct_mapping_alt_and_dst = [
    "LOGITS",
    "RESID_POST_EMBEDDING",
    "MLP_PRE_ACT",
    "MLP_POST_ACT",
    "RESID_DELTA_MLP",
    "RESID_POST_MLP",
    "ATTN_QUERY",
    "ATTN_KEY",
    "ATTN_VALUE",
    "ATTN_QK_LOGITS",
    "ATTN_QK_PROBS",
    "ATTN_WEIGHTED_SUM_OF_VALUES",
    "RESID_DELTA_ATTN",
    "RESID_POST_ATTN",
    "ONLINE_AUTOENCODER_LATENT",
    "ONLINE_MLP_AUTOENCODER_LATENT",
    "ONLINE_ATTENTION_AUTOENCODER_LATENT",
    "ONLINE_MLP_AUTOENCODER_ERROR",
    "ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR",
    "ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR",
    "MLP_INPUT_LAYER_NORM_SCALE",
    "ATTN_INPUT_LAYER_NORM_SCALE",
    "RESID_FINAL_LAYER_NORM_SCALE",
]

shape_spec_per_token_sequence_by_dst: dict[DerivedScalarType, tuple[Dimension, ...]] = {
    DerivedScalarType.RESID_POST_EMBEDDING: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.RESID_POST_MLP: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.RESID_POST_ATTN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.MLP_INPUT_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.ATTN_INPUT_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.RESID_FINAL_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.LOGITS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.VOCAB_SIZE,
    ),
    DerivedScalarType.ATTN_QUERY: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    DerivedScalarType.ATTN_KEY: (
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    DerivedScalarType.ATTN_VALUE: (
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
    ),
    DerivedScalarType.ATTN_QK_LOGITS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_QK_PROBS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_WEIGHTED_SUM_OF_VALUES: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
    ),
    DerivedScalarType.RESID_DELTA_ATTN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.MLP_PRE_ACT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.MLP_POST_ACT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ATTN_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.FLATTENED_ATTN_POST_SOFTMAX: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.RESID_DELTA_MLP: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ATTN_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.MLP_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.MLP_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.MLP_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ATTENTION_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.MLP_AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ATTENTION_AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.RESID_POST_EMBEDDING_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,  # denotes dim=1 always (i.e. one value per layer)
    ),
    DerivedScalarType.RESID_POST_MLP_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.MLP_LAYER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.RESID_POST_ATTN_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.ATTN_LAYER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ATTN_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ATTN_WRITE_SUM_HEADS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.MLP_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ATTN_WEIGHTED_VALUE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
    ),
    DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.TOKEN_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.SINGLE_NODE_WRITE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    DerivedScalarType.ATTN_OUT_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.MLP_OUT_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    # Note that although the attn query is per sequence token, and not per attended to sequence
    # token, we can separately consider edges at each (sequence token, attended to sequence token)
    # pair since the edge attribution can depend on the attended to sequence token. Analogously for
    # key and value below.
    DerivedScalarType.ATTN_QUERY_IN_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_KEY_IN_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_VALUE_IN_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.MLP_IN_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    DerivedScalarType.TOKEN_OUT_EDGE_ATTRIBUTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.VOCAB_SIZE,
    ),
    DerivedScalarType.ALWAYS_ONE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    # Note that although the attn query is per sequence token, and not per attended to sequence
    # token, we can separately consider edges at each (sequence token, attended to sequence token
    # pair since the edge activation can depend on the attended to sequence token. Analogously for
    # key and value below.
    DerivedScalarType.ATTN_QUERY_IN_EDGE_ACTIVATION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.ATTN_KEY_IN_EDGE_ACTIVATION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    DerivedScalarType.MLP_IN_EDGE_ACTIVATION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
}
