"""
This file contains code to compute derived scalars related to autoencoder latents post-hoc
(that is, from pre-existing MLP activations). Typically, the derived scalars consist of the
autoencoder latent activation multiplied by some other quantity.
"""

from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.direct_effects import (
    convert_scalar_deriver_to_write_to_final_residual_grad,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    convert_scalar_deriver_to_write_norm,
    convert_scalar_deriver_to_write_vector,
    no_op_tensor_calculate_derived_scalar_fn,
)
from neuron_explainer.activations.derived_scalars.reconstituted import make_apply_autoencoder
from neuron_explainer.activations.derived_scalars.reconstituter_class import (
    WriteLatentReconstituter,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DerivedScalarSource,
    DstConfig,
    RawScalarSource,
    ScalarDeriver,
)
from neuron_explainer.activations.derived_scalars.utils import detach_and_clone
from neuron_explainer.activations.derived_scalars.write_tensors import (
    get_autoencoder_write_tensor_by_layer_index,
    get_mlp_write_tensor_by_layer_index_with_autoencoder_context,
)
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
    PassType,
)


def make_autoencoder_latent_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_autoencoder_latent_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        # import here to avoid circular import
        from neuron_explainer.activations.derived_scalars.make_scalar_derivers import (
            make_scalar_deriver,
        )

        layer_indices = dst_config.layer_indices
        if layer_indices is None:
            model_context = dst_config.get_model_context()
            layer_indices = list(range(model_context.n_layers))

        autoencoder_context = dst_config.get_autoencoder_context(node_type)
        assert autoencoder_context is not None

        autoencoder_dst = autoencoder_context.dst
        autoencoder_dst = maybe_convert_autoencoder_dst(autoencoder_dst)
        scalar_deriver = make_scalar_deriver(autoencoder_dst, dst_config)
        apply_autoencoder = make_apply_autoencoder(autoencoder_context)

        def new_tensor_calculate_derived_scalar_fn(
            derived_scalar_tensor: torch.Tensor,
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            assert pass_type == PassType.FORWARD
            return apply_autoencoder(derived_scalar_tensor, layer_index)

        output_dst = DerivedScalarType.AUTOENCODER_LATENT.update_from_autoencoder_node_type(
            node_type
        )
        new_scalar_deriver = scalar_deriver.apply_layerwise_transform_fn_to_output(
            layerwise_transform_fn=new_tensor_calculate_derived_scalar_fn,
            pass_type_to_transform=PassType.FORWARD,
            output_dst=output_dst,
        )
        return new_scalar_deriver

    return make_autoencoder_latent_scalar_deriver


def _make_autoencoder_latent_grad_wrt_input_scalar_deriver_helper(
    dst_config: DstConfig,
    output_dst: DerivedScalarType,
    node_type: NodeType | None = None,
) -> ScalarDeriver:
    """Compute the gradient from a particular autoencoder latent, with respect to the
    autoencoder input directions.

    Requires:
    >>> dst_config.layer_indices = [layer_index]
    >>> dst_config.trace_config.tensor_indices = ("All", latent_index)
    """
    # import here to avoid circular import
    from neuron_explainer.activations.derived_scalars.make_scalar_derivers import (
        make_scalar_deriver,
    )

    trace_config = dst_config.trace_config
    assert trace_config is not None
    assert trace_config.node_type in [
        NodeType.AUTOENCODER_LATENT,
        NodeType.MLP_AUTOENCODER_LATENT,
        NodeType.ATTENTION_AUTOENCODER_LATENT,
        NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR,
    ]
    assert trace_config.tensor_indices[0] == "All"
    assert isinstance(trace_config.tensor_indices[1], int)
    latent_index_for_grad = trace_config.tensor_indices[1]
    layer_index = trace_config.layer_index
    assert layer_index is not None
    if dst_config.layer_indices is not None:
        assert [layer_index] == dst_config.layer_indices

    autoencoder_context = dst_config.get_autoencoder_context(node_type)
    assert autoencoder_context is not None
    autoencoder_dst = autoencoder_context.dst
    autoencoder_dst = maybe_convert_autoencoder_dst(autoencoder_dst)
    scalar_deriver = make_scalar_deriver(autoencoder_dst, dst_config)
    apply_autoencoder = make_apply_autoencoder(autoencoder_context, use_no_grad=False)

    def new_tensor_calculate_derived_scalar_fn(
        derived_scalar_tensor: torch.Tensor,
    ) -> torch.Tensor:
        derived_scalar_tensor = detach_and_clone(derived_scalar_tensor, requires_grad=True)
        latents = apply_autoencoder(derived_scalar_tensor, layer_index)
        latents[:, latent_index_for_grad].sum(dim=0).backward()  # sum over tokens
        assert derived_scalar_tensor.grad is not None
        return derived_scalar_tensor.grad

    new_scalar_deriver = scalar_deriver.apply_transform_fn_to_output(
        transform_fn=new_tensor_calculate_derived_scalar_fn,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
    )
    return new_scalar_deriver


def make_autoencoder_latent_grad_wrt_residual_input_scalar_deriver(
    dst_config: DstConfig,
    node_type: NodeType | None = None,
) -> ScalarDeriver:
    autoencoder_context = dst_config.get_autoencoder_context(node_type)
    assert autoencoder_context is not None
    assert dst_config.trace_config is not None
    latent_index = dst_config.trace_config.node_index
    assert latent_index is not None
    latent_reconstituter = WriteLatentReconstituter(autoencoder_context)
    return latent_reconstituter.make_gradient_scalar_deriver_for_latent_index(
        latent_index=latent_index,
        dst_config=dst_config,
        output_dst=DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT,
    )


def make_autoencoder_latent_grad_wrt_residual_input_scalar_source(
    dst_config: DstConfig,
    node_type: NodeType | None = None,
) -> DerivedScalarSource:
    autoencoder_context = dst_config.get_autoencoder_context(node_type)
    assert autoencoder_context is not None
    assert dst_config.trace_config is not None
    latent_index = dst_config.trace_config.node_index
    assert latent_index is not None
    latent_reconstituter = WriteLatentReconstituter(autoencoder_context)
    return latent_reconstituter.make_gradient_scalar_source_for_latent_index(
        latent_index=latent_index,
        dst_config=dst_config,
        output_dst=DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT,
    )


def make_autoencoder_latent_grad_wrt_mlp_post_act_input_scalar_deriver(
    dst_config: DstConfig,
    node_type: NodeType | None = None,
) -> ScalarDeriver:
    """Output shape (n_tokens, n_neurons)"""
    autoencoder_context = dst_config.get_autoencoder_context(node_type)
    assert autoencoder_context is not None
    assert autoencoder_context.dst == DerivedScalarType.MLP_POST_ACT
    return _make_autoencoder_latent_grad_wrt_input_scalar_deriver_helper(
        dst_config,
        output_dst=DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT,
        node_type=node_type,
    )


def maybe_convert_autoencoder_dst(autoencoder_dst: DerivedScalarType) -> DerivedScalarType:
    if autoencoder_dst == DerivedScalarType.RESID_DELTA_MLP:
        # TODO: Consider removing this workaround and using RESID_DELTA_MLP directly.
        autoencoder_dst = DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT
    return autoencoder_dst


def make_autoencoder_write_norm_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_autoencoder_write_norm_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        model_context = dst_config.get_model_context()
        dst = DerivedScalarType.AUTOENCODER_WRITE_NORM.update_from_autoencoder_node_type(node_type)

        autoencoder_context = dst_config.get_autoencoder_context(node_type)
        assert autoencoder_context is not None

        write_tensor_by_layer_index = get_autoencoder_write_tensor_by_layer_index(
            autoencoder_context, model_context
        )

        scalar_deriver = make_autoencoder_latent_scalar_deriver_factory(node_type)(dst_config)

        return convert_scalar_deriver_to_write_norm(
            scalar_deriver, write_tensor_by_layer_index, dst
        )

    return make_autoencoder_write_norm_scalar_deriver


def get_autoencoder_alt_from_node_type(node_type: NodeType | None) -> ActivationLocationType:
    """Get the corresponding activation_location_type from a NodeType."""
    return {
        NodeType.AUTOENCODER_LATENT: ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
        NodeType.MLP_AUTOENCODER_LATENT: ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT,
        NodeType.ATTENTION_AUTOENCODER_LATENT: ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
    }[node_type or NodeType.AUTOENCODER_LATENT]


def make_online_autoencoder_latent_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_latent_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        autoencoder_context = dst_config.get_autoencoder_context(node_type)
        assert autoencoder_context is not None
        dst = DerivedScalarType.ONLINE_AUTOENCODER_LATENT.update_from_autoencoder_node_type(
            node_type
        )
        activation_location_type = get_autoencoder_alt_from_node_type(node_type)

        return ScalarDeriver(
            dst=dst,
            dst_config=dst_config,
            sub_scalar_sources=(
                RawScalarSource(
                    activation_location_type=activation_location_type,
                    pass_type=PassType.FORWARD,
                ),
            ),
            tensor_calculate_derived_scalar_fn=no_op_tensor_calculate_derived_scalar_fn,
        )

    return make_online_autoencoder_latent_scalar_deriver


def make_online_autoencoder_write_norm_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_write_norm_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        model_context = dst_config.get_model_context()

        autoencoder_context = dst_config.get_autoencoder_context(node_type)
        assert autoencoder_context is not None
        dst = DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM.update_from_autoencoder_node_type(
            node_type
        )

        write_tensor_by_layer_index = get_autoencoder_write_tensor_by_layer_index(
            autoencoder_context, model_context
        )

        scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(
            dst_config
        )

        return convert_scalar_deriver_to_write_norm(
            scalar_deriver, write_tensor_by_layer_index, dst
        )

    return make_online_autoencoder_write_norm_scalar_deriver


def make_online_autoencoder_latentwise_write_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_latentwise_write_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        model_context = dst_config.get_model_context()

        autoencoder_context = dst_config.get_autoencoder_context(node_type)
        assert autoencoder_context is not None
        dst = DerivedScalarType.ONLINE_AUTOENCODER_WRITE.update_from_autoencoder_node_type(
            node_type
        )

        write_tensor_by_layer_index = get_autoencoder_write_tensor_by_layer_index(
            autoencoder_context, model_context
        )

        scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(
            dst_config
        )

        return convert_scalar_deriver_to_write_vector(
            scalar_deriver, write_tensor_by_layer_index, dst
        )

    return make_online_autoencoder_latentwise_write_scalar_deriver


def make_online_autoencoder_write_to_final_residual_grad_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_write_to_final_residual_grad_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(
            dst_config
        )
        dst = DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD.update_from_autoencoder_node_type(
            node_type
        )
        return convert_scalar_deriver_to_write_to_final_residual_grad(
            scalar_deriver, dst, use_existing_backward_pass_for_final_residual_grad=True
        )

    return make_online_autoencoder_write_to_final_residual_grad_scalar_deriver


def make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(
            dst_config
        )
        dst = DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD.update_from_autoencoder_node_type(
            node_type
        )
        return convert_scalar_deriver_to_write_to_final_residual_grad(
            scalar_deriver, dst, use_existing_backward_pass_for_final_residual_grad=False
        )

    return make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver


def make_online_autoencoder_act_times_grad_scalar_deriver_factory(
    node_type: NodeType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_online_autoencoder_act_times_grad_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        act_scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(
            dst_config
        )
        dst = DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD.update_from_autoencoder_node_type(
            node_type
        )
        activity_location_type = get_autoencoder_alt_from_node_type(node_type)

        return act_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
            layerwise_transform_fn=lambda act, grad, layer_index, pass_type: act * grad,
            pass_type_to_transform=PassType.FORWARD,  # act
            other_scalar_source=RawScalarSource(
                activation_location_type=activity_location_type,
                pass_type=PassType.BACKWARD,  # grad
            ),
            output_dst=dst,
        )

    return make_online_autoencoder_act_times_grad_scalar_deriver


def make_online_autoencoder_error_scalar_deriver_factory(
    activation_location_type: ActivationLocationType,
) -> Callable[[DstConfig], ScalarDeriver]:
    required_node_type = {
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR: NodeType.MLP_NEURON,
        ActivationLocationType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR: NodeType.RESIDUAL_STREAM_CHANNEL,
        ActivationLocationType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR: NodeType.RESIDUAL_STREAM_CHANNEL,
    }[activation_location_type]

    required_autoencoder_node_type = {
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR: NodeType.MLP_AUTOENCODER_LATENT,
        ActivationLocationType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR: NodeType.MLP_AUTOENCODER_LATENT,
        ActivationLocationType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR: NodeType.ATTENTION_AUTOENCODER_LATENT,
    }[activation_location_type]

    dst = DerivedScalarType.from_activation_location_type(activation_location_type)

    def make_online_autoencoder_error_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        autoencoder_context = dst_config.get_autoencoder_context(required_autoencoder_node_type)
        assert autoencoder_context is not None
        assert autoencoder_context.dst.node_type == required_node_type, (
            autoencoder_context.dst,
            required_node_type,
            activation_location_type,
        )

        return ScalarDeriver(
            dst=dst,
            dst_config=dst_config,
            sub_scalar_sources=(
                RawScalarSource(
                    activation_location_type=activation_location_type,
                    pass_type=PassType.FORWARD,
                ),
            ),
            tensor_calculate_derived_scalar_fn=no_op_tensor_calculate_derived_scalar_fn,
        )

    return make_online_autoencoder_error_scalar_deriver


def make_online_mlp_autoencoder_error_act_times_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    act_scalar_deriver = make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR
    )(dst_config)

    return act_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
        layerwise_transform_fn=lambda act, grad, layer_index, pass_type: act * grad,
        pass_type_to_transform=PassType.FORWARD,  # act
        other_scalar_source=RawScalarSource(
            activation_location_type=ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR,
            pass_type=PassType.BACKWARD,  # grad
        ),
        output_dst=DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD,
    )


def make_online_mlp_autoencoder_error_write_norm_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    model_context = dst_config.get_model_context()

    autoencoder_context = dst_config.get_autoencoder_context(NodeType.MLP_AUTOENCODER_LATENT)
    assert autoencoder_context is not None

    write_tensor_by_layer_index = get_mlp_write_tensor_by_layer_index_with_autoencoder_context(
        autoencoder_context, model_context
    )

    scalar_deriver = make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR
    )(dst_config)

    return convert_scalar_deriver_to_write_norm(
        scalar_deriver,
        write_tensor_by_layer_index,
        DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_NORM,
    )


def make_online_mlp_autoencoder_error_write_to_final_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR
    )(dst_config)
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver,
        DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=True,
    )


# helpers for autoencoder gradient


def make_autoencoder_pre_act_encoder_derivative(
    autoencoder_context: AutoencoderContext,
    layer_index: int,
    latent_index: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    autoencoder = autoencoder_context.get_autoencoder(layer_index)

    if isinstance(autoencoder.encoder, torch.nn.Linear):
        # if the encoder is linear, then the derivative is just the encoder weight matrix
        encoder = autoencoder.encoder.weight  # shape (n_latents, n_inputs)

        if latent_index is not None:
            encoder = encoder[latent_index : latent_index + 1].clone()
            # ^ need to clone to avoid MPS backend crash

        def pre_act_encoder_derivative(autoencoder_input: torch.Tensor) -> torch.Tensor:
            return autoencoder_input @ encoder.T

        return pre_act_encoder_derivative
    else:
        raise NotImplementedError("Only implemented for linear encoder for now")


def make_autoencoder_activation_fn_derivative(
    autoencoder_context: AutoencoderContext,
    layer_index: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    autoencoder = autoencoder_context.get_autoencoder(layer_index)

    if _is_relu(autoencoder.activation):
        # if the activation is ReLU, then the derivative is just a step function

        def relu_derivative(post_activations: torch.Tensor) -> torch.Tensor:
            return (post_activations > 0).to(post_activations.dtype)

        return relu_derivative
    else:
        raise NotImplementedError("Only implemented for ReLU activation function for now")


def _is_relu(activation: Callable) -> bool:
    """More robust than isinstance(activation, torch.nn.ReLU), which doesn't always work."""
    if isinstance(activation, torch.nn.ReLU):
        return True
    else:
        test_input = torch.randn(10) * 10 ** torch.randn(10)
        return torch.equal(activation(test_input), torch.nn.ReLU()(test_input))
