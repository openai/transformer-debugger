"""
This file contains Enums of three kinds of objects that are common across
LM architectures: weight locations, activation locations, and model dimensions.
The weight locations and activation locations are associated
to tuples of dimensions, to define their standardized shapes. This file is intended to
serve as a source of truth for those standards.

Abbreviations appearing in this file:
attn = attention
emb = embedding
ln = layer norm
ln_f = final layer norm
mlp = multi-layer perceptron
act = activation
resid = residual
q = query
k = key
v = value
qk = query-key inner product (with 1/sqrt(query_and_key_channels) scaling already applied)
"""

from dataclasses import dataclass
from enum import Enum, EnumMeta, auto, unique
from typing import Optional


class WeightLocationType(str, Enum):
    """These are the names of tensors that are expected to be found in model weights."""

    MLP_TO_HIDDEN = "mlp.to_hidden"
    MLP_TO_RESIDUAL = "mlp.to_residual"
    EMBEDDING = "embedding"
    UNEMBEDDING = "unembedding"
    POSITION_EMBEDDING = "position_embedding"
    ATTN_TO_QUERY = "attn.to_query"
    ATTN_TO_KEY = "attn.to_key"
    ATTN_TO_VALUE = "attn.to_value"
    ATTN_TO_RESIDUAL = "attn.to_residual"
    LAYER_NORM_GAIN_FINAL = "layer_norm_gain.final"
    LAYER_NORM_BIAS_FINAL = "layer_norm_bias.final"
    LAYER_NORM_GAIN_PRE_ATTN = "layer_norm_gain.pre_attn"
    LAYER_NORM_GAIN_PRE_MLP = "layer_norm_gain.pre_mlp"

    @property
    def is_mlp_specific(self) -> bool:
        return self in {
            WeightLocationType.MLP_TO_HIDDEN,
            WeightLocationType.MLP_TO_RESIDUAL,
            WeightLocationType.LAYER_NORM_GAIN_PRE_MLP,
        }

    @property
    def has_no_layers(self) -> bool:
        # if True, there is one of this type of weight tensor per model, and the layer index is set
        # as None wherever used (these occur at the very beginning or end of the model)
        # if False, there is one of this type of weight tensor per layer, and the tensor additionally
        # requires a layer index to specify
        return self in {
            WeightLocationType.EMBEDDING,
            WeightLocationType.UNEMBEDDING,
            WeightLocationType.POSITION_EMBEDDING,
            WeightLocationType.LAYER_NORM_GAIN_FINAL,
            WeightLocationType.LAYER_NORM_BIAS_FINAL,
        }

    @property
    def is_absolute_position_embedding_specific(self) -> bool:
        return self in {WeightLocationType.POSITION_EMBEDDING}

    @property
    def shape_spec(self) -> tuple["Dimension", ...]:
        return weight_shape_by_location_type[self]


class EnumMetaContains(EnumMeta):
    def __contains__(cls, item: object) -> bool:
        # enables the syntax "if item in enum:"
        return isinstance(item, cls) or item in cls._value2member_map_


LayerIndex = int | None  # None refers to an activation with no layer index (e.g. embeddings)


class LocationWithinLayer(int, Enum):
    """
    Coarsely specifies the location of a tensor within a layer. Each of the following is mapped to an
    int, and the ordering is from the beginning of the layer to the end.
    This is to be inferred for a scalar deriver based on information available to it.

    The level of granularity is enough to determine whether one node is upstream of a different node,
    but no more. For example, it doesn't distinguish between attention pre- and post-softmax, but it does
    distinguish between an attention head and the residual stream locations immediately before and after
    the attention layer.

    THESE VALUES ARE NOT INTENDED TO BE SERIALIZED.

    The strategy for inferring the LocationWithinLayer is to check
    (1) the node type of the ScalarDeriver
    (2) the activation location type of the ScalarDeriver (if it corresponds to a raw activation)
    (3) the dst of the ScalarDeriver
    (4) any other information available to the ScalarDeriver.

    Based on any one
    piece of information alone, it may be ambiguous what the location within layer is. For example,
    DerivedScalarType.RESID_POST_ATTN and DerivedScalarType.RESID_POST_MLP both correspond to the same
    node type, but post-attn appears earlier in the layer than post-mlp. So the location within layer
    is left ambiguous after checking the node, but clarified after checking the activation location type.
    """

    END_OF_PREV_LAYER = (
        auto()
    )  # auto() assigns increasing integer values to the enum values, starting from 1
    ATTN = auto()
    RESID_POST_ATTN = auto()
    MLP = auto()
    RESID_POST_MLP = auto()


class NodeType(str, Enum):
    """
    A "node" is defined as a model component associated with a scalar activation per
    token or per token pair. The canonical example is an MLP neuron. An activation
    for which the NodeType is defined has the node as the last dimension of the
    activation tensor.
    """

    ATTENTION_HEAD = "attention_head"
    QK_CHANNEL = "qk_channel"
    V_CHANNEL = "v_channel"
    MLP_NEURON = "mlp_neuron"
    AUTOENCODER_LATENT = "autoencoder_latent"
    MLP_AUTOENCODER_LATENT = "mlp_autoencoder_latent"
    ATTENTION_AUTOENCODER_LATENT = "attention_autoencoder_latent"
    # TODO: remove this hack, and make NodeType depend on the token dimensions
    AUTOENCODER_LATENT_BY_TOKEN_PAIR = "autoencoder_latent_by_token_pair"
    LAYER = "layer"
    RESIDUAL_STREAM_CHANNEL = "residual_stream_channel"
    VOCAB_TOKEN = "vocab_token"

    @property
    def location_within_layer(self) -> Optional["LocationWithinLayer"]:
        # this uses the information available to infer the location within a layer of a specific node_type.
        # It returns None in cases where the location within layer is ambiguous based on the information
        # provided; e.g. for residual stream node types, it might be post-attn or post-mlp.
        # (with activation location type
        # for additional clarification). It returns None in cases where the location within layer is ambiguous based on the information
        # provided; one example is for autoencoder latents, which might be based on any dst. In this case, further information from the
        # DSTConfig is used.
        # It throws an error if the node_type is *never* associated with a location within layer (e.g. vocab tokens)
        match self:
            case NodeType.MLP_NEURON:
                return LocationWithinLayer.MLP
            case NodeType.ATTENTION_HEAD | NodeType.QK_CHANNEL | NodeType.V_CHANNEL:
                return LocationWithinLayer.ATTN
            case (
                NodeType.RESIDUAL_STREAM_CHANNEL
                | NodeType.LAYER
                | NodeType.AUTOENCODER_LATENT
                | NodeType.MLP_AUTOENCODER_LATENT
                | NodeType.ATTENTION_AUTOENCODER_LATENT
                | NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR
            ):
                # these node types are ambiguous based on the information provided
                return None
            case NodeType.VOCAB_TOKEN:
                # users should not be asking about the location within layer of vocab tokens; this indicates something's wrong
                raise ValueError("Vocab tokens don't have a location within layer")
            case _:
                raise NotImplementedError(f"Unknown node type {self=}")

    @property
    def is_autoencoder_latent(self) -> bool:
        return self in {
            NodeType.AUTOENCODER_LATENT,
            NodeType.MLP_AUTOENCODER_LATENT,
            NodeType.ATTENTION_AUTOENCODER_LATENT,
            NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR,
        }


class ActivationLocationType(str, Enum, metaclass=EnumMetaContains):
    """These are the names of activations expected to be instantiated during a forward pass. All activations are
    pre-layer norm unless otherwise specified (RESID_POST_XYZ_LAYER_NORM)."""

    RESID_POST_EMBEDDING = "resid.post_emb"
    RESID_DELTA_ATTN = "resid.delta_attn"
    RESID_POST_ATTN = "resid.post_attn"
    RESID_DELTA_MLP = "resid.delta_mlp"
    RESID_POST_MLP = "resid.post_mlp"
    RESID_POST_MLP_LAYER_NORM = "resid.post_mlp_ln"
    RESID_POST_ATTN_LAYER_NORM = "resid.post_attn_ln"
    RESID_POST_FINAL_LAYER_NORM = "resid.post_ln_f"
    MLP_INPUT_LAYER_NORM_SCALE = "mlp_ln.scale"
    ATTN_INPUT_LAYER_NORM_SCALE = "attn_ln.scale"
    RESID_FINAL_LAYER_NORM_SCALE = "resid.ln_f.scale"
    ATTN_QUERY = "attn.q"
    ATTN_KEY = "attn.k"
    ATTN_VALUE = "attn.v"
    ATTN_QK_LOGITS = "attn.qk_logits"
    ATTN_QK_PROBS = "attn.qk_probs"
    ATTN_WEIGHTED_SUM_OF_VALUES = "attn.v_out"
    MLP_PRE_ACT = "mlp.pre_act"
    MLP_POST_ACT = "mlp.post_act"
    LOGITS = "logits"

    ONLINE_AUTOENCODER_LATENT = "online_autoencoder_latent"
    ONLINE_MLP_AUTOENCODER_LATENT = "online_mlp_autoencoder_latent"
    ONLINE_ATTENTION_AUTOENCODER_LATENT = "online_attention_autoencoder_latent"
    ONLINE_MLP_AUTOENCODER_ERROR = "online_mlp_autoencoder_error"
    ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR = "online_residual_mlp_autoencoder_error"
    ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR = "online_residual_attention_autoencoder_error"

    @property
    def has_no_layers(self) -> bool:
        # if True, there is one of this type of activation tensor per model, and the layer index is set
        # as None wherever used (these occur at the very beginning or end of the model)
        # if False, there is one of this type of activation tensor per layer, and the tensor additionally
        # requires a layer index to specify
        return self in {
            ActivationLocationType.RESID_POST_EMBEDDING,
            ActivationLocationType.RESID_FINAL_LAYER_NORM_SCALE,
            ActivationLocationType.RESID_POST_FINAL_LAYER_NORM,
            ActivationLocationType.LOGITS,
        }

    @property
    def shape_spec_per_token_sequence(self) -> tuple["Dimension", ...]:
        return _activation_shape_per_token_sequence_by_location_type[self]

    @property
    def ndim_per_token_sequence(self) -> int:
        return len(self.shape_spec_per_token_sequence)

    @property
    def exists_by_default(self) -> bool:
        # this returns True if the activation is expected to exist by default in the model, and False if
        # it needs to be added using hooks
        return self not in {
            ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
        }

    @property
    def node_type(self) -> NodeType:
        """The last index of an activation tensor can correspond to a type of object
        in the network called a 'node'. This can be an MLP neuron, an attention head, an autoencoder
        latent, etc. If we don't yet have a name for the last dimension of an activation tensor,
        this throws an error."""
        last_dimension = self.shape_spec_per_token_sequence[-1]
        if last_dimension in node_type_by_dimension:
            return node_type_by_dimension[last_dimension]
        else:
            raise NotImplementedError(f"Unknown node type for {last_dimension=}")

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        # this uses the information available to infer the location within a layer of a specific node_type (with activation location type
        # for additional clarification). It returns None in cases where the location within layer is ambiguous based on the information
        # provided; one example is for autoencoder latents, which might be based on any dst. In this case, further information from the
        # DSTConfig is needed.
        # It throws an error if the node_type is not associated with a location within layer.
        if self.node_type.location_within_layer is None:
            if self.node_type == NodeType.RESIDUAL_STREAM_CHANNEL:
                if self == ActivationLocationType.RESID_POST_EMBEDDING:
                    return None
                elif self == ActivationLocationType.RESID_DELTA_ATTN:
                    return LocationWithinLayer.ATTN
                elif self == ActivationLocationType.RESID_POST_ATTN:
                    return LocationWithinLayer.RESID_POST_ATTN
                elif self == ActivationLocationType.RESID_DELTA_MLP:
                    return LocationWithinLayer.MLP
                elif self == ActivationLocationType.RESID_POST_MLP:
                    return LocationWithinLayer.RESID_POST_MLP
                else:
                    return None
            else:
                return None
        else:
            return self.node_type.location_within_layer


class Dimension(str, Enum):
    """Dimensions correspond to the names of dimensions of activation tensors, and can depend on the input,
    the model, or e.g. parameters of added subgraphs such as autoencoders.
    The dimensions below are taken to be 'per layer' wherever applicable.
    Dimensions associated with attention heads (e.g. value channels) are taken to be 'per attention head'.
    """

    SEQUENCE_TOKENS = "sequence_tokens"
    ATTENDED_TO_SEQUENCE_TOKENS = "attended_to_sequence_tokens"

    """These are the names of dimensions of activation tensors that are intrinsic to a model,
    and are not a consequence of a particular input sequence. The shape of activations will in general
    depend on these and on Dimension, above. The shape of weights will in general depend only on
    these."""
    # "context" refers to the number of tokens in the sequence being processed by the model
    # "max_context_length" refers to the maximum number of tokens that can be processed by the
    # model (relevant for models with absolute position embeddingsg)
    MAX_CONTEXT_LENGTH = "max_context_length"
    # "residual_stream_channels" means the same as "d_model"
    RESIDUAL_STREAM_CHANNELS = "residual_stream_channels"
    VOCAB_SIZE = "vocab_size"
    ATTN_HEADS = "attn_heads"
    QUERY_AND_KEY_CHANNELS = "query_and_key_channels"
    VALUE_CHANNELS = "value_channels"
    MLP_ACTS = "mlp_acts"
    LAYERS = "layers"
    SINGLETON = "singleton"  # always 1

    """These are the names of dimensions that are not intrinsic to a model's activations, but that in some
    way parameterize its activations (currently just including autoencoder latents, but in future possibly
    including other methods for finding useful directions within activations)."""
    AUTOENCODER_LATENTS = "autoencoder_latents"
    # TODO: remove this hack, and make NodeType depend on the token dimensions
    AUTOENCODER_LATENTS_BY_TOKEN_PAIR = "autoencoder_latents_by_token_pair"

    @property
    def is_sequence_token_dimension(self) -> bool:
        return self in {Dimension.SEQUENCE_TOKENS, Dimension.ATTENDED_TO_SEQUENCE_TOKENS}

    @property
    def is_parameterized_dimension(self) -> bool:
        return self in {Dimension.AUTOENCODER_LATENTS}

    @property
    def is_model_intrinsic(self) -> bool:
        """this is True for dimensions that depend only on the model. This is False for dimensions that depend on either (1) the input being processed, or (2) some parameterization
        of the model activations (e.g. autoencoder latents).
        """
        return not (self.is_parameterized_dimension or self.is_sequence_token_dimension)


node_type_by_dimension: dict[Dimension, NodeType] = {
    Dimension.MLP_ACTS: NodeType.MLP_NEURON,
    Dimension.ATTN_HEADS: NodeType.ATTENTION_HEAD,
    Dimension.QUERY_AND_KEY_CHANNELS: NodeType.QK_CHANNEL,
    Dimension.VALUE_CHANNELS: NodeType.V_CHANNEL,
    Dimension.SINGLETON: NodeType.LAYER,
    Dimension.RESIDUAL_STREAM_CHANNELS: NodeType.RESIDUAL_STREAM_CHANNEL,
    Dimension.VOCAB_SIZE: NodeType.VOCAB_TOKEN,
    Dimension.AUTOENCODER_LATENTS: NodeType.AUTOENCODER_LATENT,
    Dimension.AUTOENCODER_LATENTS_BY_TOKEN_PAIR: NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR,
}


_activation_shape_per_token_sequence_by_location_type: dict[
    ActivationLocationType, tuple[Dimension, ...]
] = {
    # this is a standard convention for the shape of activation tensors per token sequence. All activations
    # by convention have either Dimension.SEQUENCE_TOKENS, Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
    # or both, as their first dimension. The ordering of dimensions is generally:
    # 1. tokens
    # 2. dimensions with a privileged basis (e.g. attention heads, MLP neurons), descending in order of
    # hierarchy
    # 3. dimensions without a privileged basis (e.g. residual stream or attention head hidden dimension)
    ActivationLocationType.RESID_POST_EMBEDDING: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_DELTA_ATTN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_POST_ATTN: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_DELTA_MLP: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_POST_MLP: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_POST_MLP_LAYER_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_POST_ATTN_LAYER_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.RESID_POST_FINAL_LAYER_NORM: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.MLP_INPUT_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    ActivationLocationType.ATTN_INPUT_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    ActivationLocationType.RESID_FINAL_LAYER_NORM_SCALE: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.SINGLETON,
    ),
    ActivationLocationType.ATTN_QUERY: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    ActivationLocationType.ATTN_KEY: (
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    ActivationLocationType.ATTN_VALUE: (
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
    ),
    ActivationLocationType.ATTN_QK_LOGITS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    ActivationLocationType.ATTN_QK_PROBS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
    ),
    ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
    ),
    ActivationLocationType.MLP_PRE_ACT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    ActivationLocationType.MLP_POST_ACT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    ActivationLocationType.ONLINE_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.AUTOENCODER_LATENTS,
    ),
    ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.MLP_ACTS,
    ),
    ActivationLocationType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    ActivationLocationType.LOGITS: (
        Dimension.SEQUENCE_TOKENS,
        Dimension.VOCAB_SIZE,
    ),
}


weight_shape_by_location_type: dict[WeightLocationType, tuple[Dimension, ...]] = {
    # this is a standard convention for the shape of weight tensors. All weights by convention have
    # the privileged basis at the top of the hierarchy first, if applicable (e.g. attention heads), and the
    # remaining dimensions are ordered: input, then output. Some tensors (e.g. biases, layernorm parameters) have only
    # a single dimension.
    WeightLocationType.POSITION_EMBEDDING: (
        Dimension.MAX_CONTEXT_LENGTH,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    WeightLocationType.EMBEDDING: (
        Dimension.VOCAB_SIZE,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    WeightLocationType.UNEMBEDDING: (
        Dimension.RESIDUAL_STREAM_CHANNELS,
        Dimension.VOCAB_SIZE,
    ),
    WeightLocationType.ATTN_TO_QUERY: (
        Dimension.ATTN_HEADS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    WeightLocationType.ATTN_TO_KEY: (
        Dimension.ATTN_HEADS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
        Dimension.QUERY_AND_KEY_CHANNELS,
    ),
    WeightLocationType.ATTN_TO_VALUE: (
        Dimension.ATTN_HEADS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
        Dimension.VALUE_CHANNELS,
    ),
    WeightLocationType.ATTN_TO_RESIDUAL: (
        Dimension.ATTN_HEADS,
        Dimension.VALUE_CHANNELS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    WeightLocationType.MLP_TO_HIDDEN: (
        Dimension.RESIDUAL_STREAM_CHANNELS,
        Dimension.MLP_ACTS,
    ),
    WeightLocationType.MLP_TO_RESIDUAL: (
        Dimension.MLP_ACTS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ),
    WeightLocationType.LAYER_NORM_GAIN_PRE_MLP: (Dimension.RESIDUAL_STREAM_CHANNELS,),
    WeightLocationType.LAYER_NORM_GAIN_PRE_ATTN: (Dimension.RESIDUAL_STREAM_CHANNELS,),
    WeightLocationType.LAYER_NORM_GAIN_FINAL: (Dimension.RESIDUAL_STREAM_CHANNELS,),
    WeightLocationType.LAYER_NORM_BIAS_FINAL: (Dimension.RESIDUAL_STREAM_CHANNELS,),
}


def get_dimension_index_of_weight_location_type(
    weight_location_type: WeightLocationType, dimension: Dimension
) -> int:
    """Returns the index of a dimension within a weight tensor, and raises an error if the
    dimension is found 0 or >1 time (0 can happen, but >1 indicates a bug somewhere)."""
    assert weight_shape_by_location_type[weight_location_type].count(dimension) == 1
    return weight_shape_by_location_type[weight_location_type].index(dimension)


@unique
class PassType(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"

    def __repr__(self) -> str:
        return f"'{self.value}'"


@dataclass(frozen=True)
class ActivationLocationTypeAndPassType:
    activation_location_type: ActivationLocationType
    pass_type: PassType

    @property
    def exists_by_default(self) -> bool:
        return self.activation_location_type.exists_by_default
