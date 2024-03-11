from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.direct_effects import (
    convert_scalar_deriver_to_write_to_final_residual_grad,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    convert_scalar_deriver_to_write,
    convert_scalar_deriver_to_write_norm,
    convert_scalar_deriver_to_write_vector,
    make_scalar_deriver_factory_for_activation_location_type,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig, ScalarDeriver
from neuron_explainer.activations.derived_scalars.write_tensors import (
    get_mlp_write_tensor_by_layer_index,
)
from neuron_explainer.models.model_component_registry import ActivationLocationType

"""This file contains code to compute derived scalars related to MLP activations (typically an
MLP activation multiplied by some other value, such as the norm of that MLP neuron's output weight vector,
or the gradient of the loss with respect to that MLP activation)."""


def get_base_mlp_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the MLP activations."""
    return make_scalar_deriver_factory_for_activation_location_type(
        activation_location_type=ActivationLocationType.MLP_POST_ACT,
    )(dst_config)


def make_mlp_write_norm_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the write norm for each
    MLP neuron at each token."""
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))

    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    W_out_by_layer_index = get_mlp_write_tensor_by_layer_index(
        model_context=model_context, layer_indices=layer_indices
    )
    return convert_scalar_deriver_to_write_norm(
        scalar_deriver=scalar_deriver,
        write_tensor_by_layer_index=W_out_by_layer_index,
        output_dst=DerivedScalarType.MLP_WRITE_NORM,
    )


def make_mlp_write_to_final_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver=scalar_deriver,
        output_dst=DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=True,
    )


def make_mlp_write_to_final_activation_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver=scalar_deriver,
        output_dst=DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=False,
    )


def make_resid_delta_mlp_from_mlp_post_act_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the write vector for the MLP layer at each token."""
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))

    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    W_out_by_layer_index = get_mlp_write_tensor_by_layer_index(
        model_context=model_context, layer_indices=layer_indices
    )
    return convert_scalar_deriver_to_write(
        scalar_deriver=scalar_deriver,
        write_tensor_by_layer_index=W_out_by_layer_index,
        output_dst=DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT,
    )


def make_mlp_neuronwise_write_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the write vector of each MLP neuron at each token."""
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))

    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    W_out_by_layer_index = get_mlp_write_tensor_by_layer_index(
        model_context=model_context, layer_indices=layer_indices
    )
    return convert_scalar_deriver_to_write_vector(
        scalar_deriver=scalar_deriver,
        write_tensor_by_layer_index=W_out_by_layer_index,
        output_dst=DerivedScalarType.MLP_WRITE,
    )
