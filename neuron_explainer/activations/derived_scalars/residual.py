"""This file contains code to compute derived scalars related to activations in the residual stream basis."""

from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.direct_effects import (
    convert_scalar_deriver_to_write_to_final_residual_grad,
)
from neuron_explainer.activations.derived_scalars.indexing import DETACH_LAYER_NORM_SCALE
from neuron_explainer.activations.derived_scalars.locations import (
    NoLayersLayerIndexer,
    OffsetLayerIndexer,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    get_scalar_sources_for_activation_location_types,
    make_scalar_deriver_factory_for_act_times_grad,
    make_scalar_deriver_factory_for_activation_location_type,
)
from neuron_explainer.activations.derived_scalars.reconstituted import (
    apply_autoencoder_pre_latent,
    apply_mlp_pre_act,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DstConfig,
    PassType,
    RawScalarSource,
    ScalarDeriver,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    LayerIndex,
    NodeType,
)
from neuron_explainer.models.model_context import StandardModelContext, get_embedding

_residual_norm_dst_by_activation_location_type: dict[ActivationLocationType, DerivedScalarType] = {
    ActivationLocationType.RESID_POST_EMBEDDING: DerivedScalarType.RESID_POST_EMBEDDING_NORM,
    ActivationLocationType.RESID_POST_MLP: DerivedScalarType.RESID_POST_MLP_NORM,
    ActivationLocationType.RESID_DELTA_MLP: DerivedScalarType.MLP_LAYER_WRITE_NORM,
    ActivationLocationType.RESID_POST_ATTN: DerivedScalarType.RESID_POST_ATTN_NORM,
    ActivationLocationType.RESID_DELTA_ATTN: DerivedScalarType.ATTN_LAYER_WRITE_NORM,
}


def make_previous_layer_resid_post_mlp_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Several DSTs require use of the residual stream activations immediately preceding some particular set of activations.
    If the activations are MLP activations, this is simple: just the resid.post_attn activations in the same layer. If the
    activations are attention activations, this is a bit more complicated: resid.post_mlp in the previous layer if layer_index > 0,
    or resid.post_emb (which has no layers) if layer_index == 0.
    As a result, this gives the residual stream activations just prior to attention with layer_index k at entry k of
    activations_by_layer_index.
    This ScalarDeriver is for the latter case."""

    def get_scalar_sources(pass_type: PassType) -> tuple[RawScalarSource, RawScalarSource]:
        return (
            RawScalarSource(
                activation_location_type=ActivationLocationType.RESID_POST_MLP,
                pass_type=pass_type,
                layer_indexer=OffsetLayerIndexer(-1),
            ),  # resid.post_mlp
            RawScalarSource(
                activation_location_type=ActivationLocationType.RESID_POST_EMBEDDING,
                pass_type=pass_type,
                layer_indexer=NoLayersLayerIndexer(),
            ),  # resid.post_emb
        )

    if dst_config.derive_gradients:
        sub_scalar_sources: tuple[RawScalarSource, ...] = get_scalar_sources(
            PassType.FORWARD
        ) + get_scalar_sources(PassType.BACKWARD)
    else:
        sub_scalar_sources = get_scalar_sources(PassType.FORWARD)

    def tensor_calculate_derived_scalar_fn(
        raw_activation_data_tuple: tuple[torch.Tensor, ...],
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        if len(raw_activation_data_tuple) == 2:
            assert pass_type == PassType.FORWARD
            resid_post_mlp, resid_post_emb = raw_activation_data_tuple
        else:
            assert len(raw_activation_data_tuple) == 4
            _, _, resid_post_mlp, resid_post_emb = raw_activation_data_tuple
        assert layer_index is not None
        if layer_index == 0:
            return resid_post_emb
        else:
            assert layer_index > 0
            return resid_post_mlp

    return ScalarDeriver(
        dst=DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP,
        dst_config=dst_config,
        sub_scalar_sources=sub_scalar_sources,
        tensor_calculate_derived_scalar_fn=tensor_calculate_derived_scalar_fn,
    )


def make_residual_norm_scalar_deriver_factory_for_activation_location_type(
    activation_location_type: ActivationLocationType,
) -> Callable[[DstConfig], ScalarDeriver]:
    """this is for DerivedScalarType's 1:1 with a ActivationLocationType, which can be generated from just the ActivationLocationType
    and no additional information"""

    dst = _residual_norm_dst_by_activation_location_type[activation_location_type]

    assert activation_location_type.shape_spec_per_token_sequence == (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ), f"Residual norm only defined for residual stream activations, not {activation_location_type=} with shape {activation_location_type.shape_spec_per_token_sequence=}"

    def make_scalar_deriver_fn(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        sub_scalar_sources = get_scalar_sources_for_activation_location_types(
            activation_location_type, dst_config.derive_gradients
        )

        def compute_norm(
            raw_activation_data_tuple: tuple[torch.Tensor, ...],
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            assert len(raw_activation_data_tuple) == 1
            raw_activation_data = raw_activation_data_tuple[0]
            return raw_activation_data.norm(dim=-1)[:, None]  # singleton final dimension

        return ScalarDeriver(
            dst=dst,
            dst_config=dst_config,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=compute_norm,
        )

    return make_scalar_deriver_fn


def make_token_attribution_scalar_deriver(dst_config: DstConfig) -> ScalarDeriver:
    """This computes an attribution value for each token in the sequence, the inner product of act and grad for the embedding.
    This corresponds to the estimated importance of this token to the final prediction."""
    activation_location_type = ActivationLocationType.RESID_POST_EMBEDDING
    dst = DerivedScalarType.TOKEN_ATTRIBUTION

    act_times_grad_scalar_deriver = make_scalar_deriver_factory_for_act_times_grad(
        activation_location_type=activation_location_type,
        dst=dst,
    )(dst_config)

    def sum_over_last_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)[:, None]  # singleton final dim

    return act_times_grad_scalar_deriver.apply_transform_fn_to_output(
        sum_over_last_dim, pass_type_to_transform=PassType.FORWARD, output_dst=dst
    )


_residual_projection_to_final_residual_grad_dst_by_activation_location_type: dict[
    ActivationLocationType, DerivedScalarType
] = {
    ActivationLocationType.RESID_POST_EMBEDDING: DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD,
    ActivationLocationType.RESID_POST_MLP: DerivedScalarType.RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD,
    ActivationLocationType.RESID_DELTA_MLP: DerivedScalarType.MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD,
    ActivationLocationType.RESID_POST_ATTN: DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD,
    ActivationLocationType.RESID_DELTA_ATTN: DerivedScalarType.ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD,
}


_residual_projection_to_final_activation_residual_grad_dst_by_activation_location_type: dict[
    ActivationLocationType, DerivedScalarType
] = {
    ActivationLocationType.RESID_POST_EMBEDDING: DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
}


def make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
    activation_location_type: ActivationLocationType,
    use_existing_backward_pass_for_final_residual_grad: bool,
) -> Callable[[DstConfig], ScalarDeriver]:
    """this is for DerivedScalarType's 1:1 with a ActivationLocationType, which can be generated from just the ActivationLocationType
    and no additional information"""

    if use_existing_backward_pass_for_final_residual_grad:
        dst = _residual_projection_to_final_residual_grad_dst_by_activation_location_type[
            activation_location_type
        ]
    else:
        dst = (
            _residual_projection_to_final_activation_residual_grad_dst_by_activation_location_type[
                activation_location_type
            ]
        )

    assert activation_location_type.shape_spec_per_token_sequence == (
        Dimension.SEQUENCE_TOKENS,
        Dimension.RESIDUAL_STREAM_CHANNELS,
    ), f"Residual norm only defined for residual stream activations, not {activation_location_type=} with shape {activation_location_type.shape_spec_per_token_sequence=}"

    def make_scalar_deriver_fn(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        residual_scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
            activation_location_type
        )(dst_config)

        return convert_scalar_deriver_to_write_to_final_residual_grad(
            scalar_deriver=residual_scalar_deriver,
            use_existing_backward_pass_for_final_residual_grad=use_existing_backward_pass_for_final_residual_grad,
            output_dst=dst,
        )

    return make_scalar_deriver_fn


def make_vocab_token_write_to_input_direction_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """
    This computes the inner product between each token's embedding vector, and the input
    weight for a particular autoencoder latent or MLP neuron. The MLP neuron case
    straightforwardly applies the logit-lens setup: Embedding * InputWeightWithLayerNormGain.
    For the MLP autoencoder case (so far only implemented for MLP latents), we first convert
    to the MLP neuron basis using MlpLayer(Embedding), then multiply by InputWeight (there is
    no associated layer norm gain). MlpLayer(Embedding) * InputWeight corresponds to the
    autoencoder latent pre-activation in the case where the residual stream vector were exactly
    the token embedding vector.
    """
    dst = DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION

    activation_index = dst_config.activation_index_for_fake_grad
    assert activation_index is not None
    layer_index = activation_index.layer_index
    assert layer_index is not None
    activation_location_type = activation_index.activation_location_type
    neuron_index = activation_index.tensor_indices[1]
    assert isinstance(neuron_index, int)

    model_context = dst_config.get_model_context()
    autoencoder_context = dst_config.get_autoencoder_context(NodeType.MLP_AUTOENCODER_LATENT)

    assert isinstance(model_context, StandardModelContext)
    transformer = model_context.get_or_create_model()
    transformer_layer = transformer.xf_layers[layer_index]

    assert activation_location_type in (
        ActivationLocationType.MLP_POST_ACT,
        ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT,
    ), f"Virtual input weight not defined for {activation_location_type=}"

    emb = get_embedding(model_context)  # resulting shape: n_vocab, d_model
    match activation_location_type:
        # only fwd pass is used, so detaching layer norm scale is not relevant
        case ActivationLocationType.MLP_POST_ACT:
            token_vector = apply_mlp_pre_act(
                transformer_layer, emb, detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE
            )[:, neuron_index]
            # ^ resulting shape: n_vocab,
        case (
            ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT
            | ActivationLocationType.ONLINE_AUTOENCODER_LATENT
        ):
            assert autoencoder_context is not None
            autoencoder = autoencoder_context.get_autoencoder(layer_index)
            # directly applies MLP layer + autoencoder to all token embedding
            # vectors
            token_vector = apply_autoencoder_pre_latent(
                transformer_layer,
                autoencoder,
                emb,
                autoencoder_dst=autoencoder_context.dst,
                detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
                # index the latent to avoid creating a matrix of shape (n_vocab, n_latents)
                latent_slice=slice(neuron_index, neuron_index + 1),
            )[:, 0]
            # indexed by zero because we already sliced to a single neuron
            # resulting shape: n_vocab,
        case _:
            raise NotImplementedError(activation_location_type)

    def replace_with_token_vector(
        resid_post_emb: torch.Tensor,
    ) -> torch.Tensor:
        # resid_post_emb only used for shape
        return token_vector.unsqueeze(0).expand(  # resulting shape: 1, n_vocab
            resid_post_emb.shape[0], -1
        )  # resulting shape: n_sequence_tokens, n_vocab

    resid_post_emb_scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING
    )(dst_config)

    return resid_post_emb_scalar_deriver.apply_transform_fn_to_output(
        replace_with_token_vector, pass_type_to_transform=PassType.FORWARD, output_dst=dst
    )


def make_unity_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    resid_post_emb_scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING
    )(dst_config)
    dst = DerivedScalarType.ALWAYS_ONE

    def convert_to_unity(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor[:, 0:1])  # (n_sequence_tokens, 1)

    return resid_post_emb_scalar_deriver.apply_transform_fn_to_output(
        convert_to_unity, pass_type_to_transform=PassType.FORWARD, output_dst=dst
    )
