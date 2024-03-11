from dataclasses import dataclass

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    ActivationIndex,
    NodeIndex,
    TraceConfig,
)
from neuron_explainer.models.autoencoder_context import (
    AutoencoderConfig,
    AutoencoderContext,
    MultiAutoencoderContext,
)
from neuron_explainer.models.inference_engine_type_registry import InferenceEngineType
from neuron_explainer.models.model_component_registry import NodeType, PassType
from neuron_explainer.models.model_context import (
    ModelContext,
    StandardModelContext,
    get_default_device,
)
from neuron_explainer.models.transformer import Transformer


@dataclass(frozen=True)
class DstConfig:
    """Holds any information that's necessary to construct a ScalarDeriver for any
    DerivedScalarType. Note that which fields are set and used will depend on
    which DerivedScalarType is being used. For DerivedScalarType's that are 1:1 with
    HookLocationType's, all fields can be None."""

    # derive gradients specifies whether to compute the gradients of the loss function with
    # respect to the derived scalar. In general, this may require loading some combination of
    # activations and gradients from blobstore. If False, then we can get away with loading
    # fewer kinds of activations and gradients.
    derive_gradients: bool = False
    model_name: str | None = None
    """If model_name and model_context are both provided, then the model_name associated with model_context will be asserted
    to match model_name. Other than that, model_name is not used in that case (model_context overrides)."""
    inference_engine_type: InferenceEngineType = InferenceEngineType.STANDARD
    layer_indices: list[int] | None = None
    """Contains flags relevant to reading activations from blobstore (e.g. non-standard shape conventions of tensors)"""
    autoencoder_config: AutoencoderConfig | None = None
    """Contains paths to autoencoders that will be used to derive scalar values, as well as the activation location type
    which the autoencoder acts on.
    If autoencoder_config and autoencoder_context are both provided, then the autoencoder_config associated with autoencoder_context will be asserted
    to match autoencoder_config. Other than that, autoencoder_config is not used in that case (autoencoder_context overrides)."""
    model_context: ModelContext | None = None
    """Optionally, model_context can be used in place of model_name to pre-load model weights and use
    them to derive scalars. If provided, model_context will supersede model_name."""
    autoencoder_context: AutoencoderContext | None = None
    multi_autoencoder_context: MultiAutoencoderContext | None = None
    """Optionally, autoencoder_context can be used in place of autoencoder_config to pre-load autoencoders and use
    them to derive autoencoder latents."""
    trace_config: TraceConfig | None = None
    """Some DSTs depend on from what location and/or layer the backward pass was computed. "None" (the default)
    means to use some loss (a function of ActivationLocationType.LOGITS) to perform the backward pass, rather than
    some intermediate activation. This field can be used to override that default, for example by specifying
    a TraceConfig using an ActivationIndex with ActivationLocationType.MLP_POST_ACT if computing a backward pass from a particular MLP activation."""
    node_index_for_attention_write: NodeIndex | None = None
    """Some DSTs examine the direction being read from corresponding to some particular projection of an attention write vector.
    In this case, the node index of the attention write vector in question can be specified here. Note that if activation_index_for_grad
    is specified, this should be prior to the layer_index of the trace_config."""
    device_for_raw_activations: torch.device | None = None
    """In some cases, neither model_context nor autoencoder_context is provided, but we must still infer a device. This is for
    those cases"""
    activation_index_for_fake_grad: ActivationIndex | None = None
    """Some DSTs compute a "fake gradient" by running a full backward pass, computing the gradient of a loss function which is identically 0,
    but ablating the gradient corresponding to activation_index_for_fake_grad to be 1. This is useful for computing the "read direction" of a
    derived scalar even when its gradient and activation might be 0 for a particular prompt."""
    node_index_for_attribution: NodeIndex | None = None
    """Some DSTs compute attribution of edges into or out of a single node. This config option specifies which node."""
    detach_layer_norm_scale_for_attribution: bool = False
    """When computing act * grad attribution of an edge, one can choose whether to detach the layer norm scale immediately
    before the downstream node."""

    def get_device(self) -> torch.device:
        if self.model_context is not None:
            return self.model_context.device
        elif self.autoencoder_context is not None:
            return self.autoencoder_context.device
        elif self.device_for_raw_activations is not None:
            return self.device_for_raw_activations
        else:
            return get_default_device()

    def get_model_context(self) -> ModelContext:
        if self.model_context is not None:
            return self.model_context
        else:
            assert self.model_name is not None
            return ModelContext.from_model_type(
                self.model_name,
                device=get_default_device(),
                inference_engine_type=self.inference_engine_type,
            )

    def get_autoencoder_context(
        self, node_type: NodeType | None = None
    ) -> AutoencoderContext | None:
        autoencoder_context: AutoencoderContext | None = None
        if self.autoencoder_context is not None:
            autoencoder_context = self.autoencoder_context
        elif self.multi_autoencoder_context is not None:
            autoencoder_context = self.multi_autoencoder_context.get_autoencoder_context(node_type)
        elif self.autoencoder_config is not None:
            autoencoder_config = self.autoencoder_config
            assert autoencoder_config is not None
            autoencoder_context = AutoencoderContext(autoencoder_config, device=self.get_device())
        else:
            return None

        assert autoencoder_context is not None, str(node_type)
        if node_type is not None and node_type != NodeType.AUTOENCODER_LATENT:
            assert (
                node_type == autoencoder_context.autoencoder_node_type
            ), f"{node_type=} {autoencoder_context.autoencoder_node_type=}"
        return autoencoder_context

    def requires_grad_for_type(self, dst: DerivedScalarType) -> bool:
        if self.derive_gradients:
            required_pass_types = [PassType.FORWARD, PassType.BACKWARD]
        else:
            required_pass_types = [PassType.FORWARD]
        return any(
            dst.requires_grad_for_pass_type(required_pass_type)
            for required_pass_type in required_pass_types
        )

    def get_n_layers(self) -> int:
        """DstConfigs are always meant to be used on a single particular model. This
        function will error if that model has not been specified."""
        return self.get_model_context().n_layers

    def get_autoencoder_dst(self, node_type: NodeType | None = None) -> DerivedScalarType | None:
        """DstConfigs are not always used with autoencoders, so they don't need to specify
        the autoencoder. This function will return None if the autoencoder context has not
        been specified."""
        autoencoder_context = self.get_autoencoder_context(node_type)
        if autoencoder_context is None:
            return None
        else:
            return autoencoder_context.dst

    def get_or_create_model(self) -> Transformer:
        """
        Returns a transformer (only valid if a StandardModelContext was specified in the config).
        """
        model_context = self.get_model_context()
        assert isinstance(model_context, StandardModelContext)
        return model_context.get_or_create_model()

    def __post_init__(self) -> None:
        config_setting_to_device: dict[str, torch.device] = {}
        if self.model_context is not None:
            assert self.model_context.model_name == self.model_name or self.model_name is None
            config_setting_to_device["model"] = self.model_context.device

        if self.autoencoder_context is not None:
            assert (self.autoencoder_context.autoencoder_config == self.autoencoder_config) or (
                self.autoencoder_config is None
            )
            config_setting_to_device["autoencoder"] = self.autoencoder_context.device

        if self.multi_autoencoder_context is not None:
            for (
                node_type,
                autoencoder_context,
            ) in self.multi_autoencoder_context.autoencoder_context_by_node_type.items():
                config_setting_to_device[node_type] = autoencoder_context.device

        if self.device_for_raw_activations is not None:
            config_setting_to_device["raw activations"] = self.device_for_raw_activations

        if len(config_setting_to_device) > 1:
            assert (
                len(set(config_setting_to_device.values())) == 1
            ), f"All devices provided must match, but {config_setting_to_device=}"

        if self.node_index_for_attention_write is not None:
            assert self.node_index_for_attention_write.node_type == NodeType.ATTENTION_HEAD
            assert self.node_index_for_attention_write.layer_index is not None
            if (
                self.trace_config is not None
                and self.trace_config.node_type is not NodeType.VOCAB_TOKEN
            ):
                assert self.trace_config.layer_index is not None

        if self.trace_config is not None:
            # backward pass from a backward pass activation is not supported
            assert self.trace_config.pass_type == PassType.FORWARD
