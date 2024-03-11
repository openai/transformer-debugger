from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import ActivationIndex, NodeIndex
from neuron_explainer.activations.derived_scalars.reconstituted import (
    make_apply_logits,
    make_apply_logprobs,
)
from neuron_explainer.activations.derived_scalars.reconstituter_class import Reconstituter
from neuron_explainer.models.model_component_registry import LayerIndex, NodeType, PassType
from neuron_explainer.models.model_context import ModelContext


class LogProbReconstituter(Reconstituter):
    """Reconstitute vocab token logprobs from final residual stream location. Can be used e.g. to compute
    effect of residual stream writes on token logprobs, rather than logits."""

    residual_dst: DerivedScalarType = DerivedScalarType.RESID_POST_MLP
    requires_other_scalar_source: bool = False

    def __init__(
        self,
        model_context: ModelContext,
        detach_layer_norm_scale: bool,
    ):
        super().__init__()
        self._model_context = model_context
        self.detach_layer_norm_scale = detach_layer_norm_scale
        transformer = self._model_context.get_or_create_model()
        self._reconstitute_activations_fn = make_apply_logprobs(
            transformer=transformer,
            detach_layer_norm_scale=self.detach_layer_norm_scale,
        )

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert other_arg is None
        assert layer_index == self._model_context.n_layers - 1
        assert pass_type == PassType.FORWARD
        return self._reconstitute_activations_fn(resid)


class LogitReconstituter(Reconstituter):
    """Reconstitute vocab token logprobs from final residual stream location. Can be used e.g. to compute
    effect of residual stream writes on token logprobs, rather than logits."""

    residual_dst: DerivedScalarType = DerivedScalarType.RESID_POST_MLP
    requires_other_scalar_source: bool = False

    def __init__(
        self,
        model_context: ModelContext,
        detach_layer_norm_scale: bool,
    ):
        super().__init__()
        self._model_context = model_context
        self.detach_layer_norm_scale = detach_layer_norm_scale
        transformer = self._model_context.get_or_create_model()
        self._reconstitute_activations_fn = make_apply_logits(
            transformer=transformer,
            detach_layer_norm_scale=self.detach_layer_norm_scale,
        )

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert other_arg is None
        assert layer_index == self._model_context.n_layers - 1
        assert pass_type == PassType.FORWARD
        return self._reconstitute_activations_fn(resid)

    def get_residual_activation_index(self) -> ActivationIndex:
        # this contains only the information that we're interested in the final residual stream layer
        dummy_node_index = NodeIndex(
            node_type=NodeType.LAYER,
            layer_index=self._model_context.n_layers - 1,
            tensor_indices=(),
            pass_type=PassType.FORWARD,
        )
        return self.get_residual_activation_index_for_node_index(
            node_index=dummy_node_index,
        )

    def make_reconstitute_gradient_of_loss_fn(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def scalar_hook(
            resid: torch.Tensor,
        ) -> torch.Tensor:
            # loss fn expects a batch dimension
            return loss_fn(resid.unsqueeze(0)).squeeze(0)

        def reconstitute_gradient(
            resid: torch.Tensor,
        ) -> torch.Tensor:
            return self.reconstitute_gradient(
                resid=resid,
                other_arg=None,
                layer_index=self._model_context.n_layers - 1,
                pass_type=PassType.FORWARD,
                scalar_hook=scalar_hook,
            )

        return reconstitute_gradient
