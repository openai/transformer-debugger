import pytest
import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.indexing import (
    ActivationIndex,
    AttnSubNodeIndex,
    NodeIndex,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig
from neuron_explainer.activations.derived_scalars.tests.utils import get_activation_shape
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import StandardModelContext

REFERENCE_DS_STORE_PATH_BY_GRAD_LOCATION = {
    "mlp": "az://openaipublic/neuron-explainer/test-data/reference_ds_stores/test_all_dsts_reference_ds_store.pt",
    "attn": "az://openaipublic/neuron-explainer/test-data/reference_ds_stores/attn/test_all_dsts_reference_ds_store.pt",
}


DETACH_LAYER_NORM_SCALE_FOR_TEST = (
    False  # this sets whether to detach layer norm scale when computing these DSTs.
)
# Likely the desired value is True going forward, but saved activations implicitly used False here (TODO).


@pytest.mark.parametrize("grad_location", ["mlp", "attn"])
def test_dsts_consistency(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
    grad_location: str,
) -> None:
    reference_ds_store_path = REFERENCE_DS_STORE_PATH_BY_GRAD_LOCATION[grad_location]
    prompt = "This is a test"
    n_tokens = len(standard_model_context.encode(prompt))

    grad_layer_index = 5
    token_index = 1
    attended_to_token_index = 1

    if grad_location == "mlp":
        activation_index_for_grad = ActivationIndex(
            activation_location_type=ActivationLocationType.MLP_POST_ACT,
            layer_index=grad_layer_index,
            tensor_indices=(token_index, 0),
            pass_type=PassType.FORWARD,
        )
    else:
        assert grad_location == "attn"
        activation_index_for_grad = ActivationIndex(
            activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
            layer_index=grad_layer_index,
            tensor_indices=(token_index, attended_to_token_index, 0),
            pass_type=PassType.FORWARD,
        )

    # these require separate configs
    out_edge_attribution_dsts = [
        DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION,
        DerivedScalarType.ATTN_OUT_EDGE_ATTRIBUTION,
        DerivedScalarType.MLP_OUT_EDGE_ATTRIBUTION,
        DerivedScalarType.ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION,
        DerivedScalarType.TOKEN_OUT_EDGE_ATTRIBUTION,
    ]
    # these require separate configs
    in_edge_attribution_dsts = [
        DerivedScalarType.SINGLE_NODE_WRITE,
        DerivedScalarType.ATTN_QUERY_IN_EDGE_ATTRIBUTION,
        DerivedScalarType.ATTN_KEY_IN_EDGE_ATTRIBUTION,
        DerivedScalarType.ATTN_VALUE_IN_EDGE_ATTRIBUTION,
        DerivedScalarType.MLP_IN_EDGE_ATTRIBUTION,
        DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION,
        DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD,
        DerivedScalarType.ATTN_QUERY_IN_EDGE_ACTIVATION,
        DerivedScalarType.ATTN_KEY_IN_EDGE_ACTIVATION,
        DerivedScalarType.MLP_IN_EDGE_ACTIVATION,
        DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION,
    ]

    dst_list = list(DerivedScalarType.__members__.values())

    disallow_list = (
        [
            # not present based on autoencoder settings (mlp_post_act autoencoder only)
            DerivedScalarType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR,
            DerivedScalarType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR,
            DerivedScalarType.ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.ATTENTION_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
            # requires a different autoencoder or a different activation index
            DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT,
            DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT,
            DerivedScalarType.ATTN_WRITE_TO_LATENT,
            DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
            DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
            DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED,
            DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN,
            DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED,
            DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION,
            DerivedScalarType.ALWAYS_ONE,
        ]
        + out_edge_attribution_dsts
        + in_edge_attribution_dsts
    )
    starting_dst_list = [dst for dst in dst_list if dst not in disallow_list]

    standard_autoencoder_context.warmup()

    dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
        trace_config=TraceConfig.from_activation_index(
            activation_index_for_grad, detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST
        ),
    )
    dst_config_by_dst = {dst: dst_config for dst in starting_dst_list}

    attribution_layer_index = 3
    assert attribution_layer_index < grad_layer_index - 1
    # edge attribution DSTs involve 3 locations in the network (from most downstream to most upstream):
    # 1. the location from which the backward pass is run
    # 2. the location of the downstream node of the edge whose attribution is being computed
    # 3. the location of the upstream node of the edge whose attribution is being computed
    # in the case where attribution_layer_index refers to the upstream node (3.), we need it to be at least 2 layers
    # earlier than the grad_layer_index (1.), so that there is at least one layer between the attribution layer and
    # the grad layer, for nodes in category (2.) to exist

    in_edge_attribution_dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
        trace_config=TraceConfig.from_activation_index(
            activation_index_for_grad, detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST
        ),
        node_index_for_attribution=NodeIndex(
            # must be an earlier layer than activation_index_for_grad
            node_type=NodeType.ATTENTION_HEAD,
            layer_index=attribution_layer_index,
            tensor_indices=(token_index, attended_to_token_index, 0),
            pass_type=PassType.FORWARD,  # note: does not test autoencoder, MLP nodes
        ),
        detach_layer_norm_scale_for_attribution=DETACH_LAYER_NORM_SCALE_FOR_TEST,
    )
    dst_config_by_dst.update(
        {dst: in_edge_attribution_dst_config for dst in in_edge_attribution_dsts}
    )

    out_edge_attribution_dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
        trace_config=TraceConfig.from_activation_index(
            activation_index_for_grad, detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST
        ),
        node_index_for_attribution=AttnSubNodeIndex(
            # must be an earlier layer than activation_index_for_grad
            node_type=NodeType.ATTENTION_HEAD,
            layer_index=attribution_layer_index,
            tensor_indices=(token_index, attended_to_token_index, 0),
            q_k_or_v=ActivationLocationType.ATTN_VALUE,
            pass_type=PassType.FORWARD,  # note: does not test autoencoder, MLP, Q/K subnodes
        ),
        detach_layer_norm_scale_for_attribution=DETACH_LAYER_NORM_SCALE_FOR_TEST,
    )
    dst_config_by_dst.update(
        {dst: out_edge_attribution_dst_config for dst in out_edge_attribution_dsts}
    )

    dst_list = list(dst_config_by_dst.keys())

    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig | None]] = list(
        dst_config_by_dst.items()
    )

    current_ds_store, _, raw_store = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        prompt=prompt,
        trace_config=TraceConfig.from_activation_index(
            activation_index_for_grad, detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST
        ),
        dst_and_config_list=dst_and_config_list,
        autoencoder_context=standard_autoencoder_context,
    )

    # check that the derived scalar is not all zeros, and has the correct shape
    for (
        dst,
        _pass_type,
    ), derived_scalar in current_ds_store.activations_and_metadata_by_dst_and_pass_type.items():
        if Dimension.AUTOENCODER_LATENTS in dst.shape_spec_per_token_sequence:
            n_latents = standard_autoencoder_context.num_autoencoder_directions
        else:
            n_latents = None

        assert derived_scalar.shape == get_activation_shape(
            dst, standard_model_context, n_tokens, n_latents
        ), f"{dst}: {derived_scalar.shape} != {get_activation_shape(dst, standard_model_context, n_tokens, n_latents)}"

        if dst in in_edge_attribution_dsts and dst not in {
            DerivedScalarType.SINGLE_NODE_WRITE,
            DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD,
        }:
            last_tensor = derived_scalar.activations_by_layer_index[attribution_layer_index + 1]
            assert not torch.all(
                last_tensor[..., 0] == 0.0
            ), dst  # the first element is legitimately zero for these DSTs at or before attribution_layer_index
        else:
            first_tensor = next(iter(derived_scalar.activations_by_layer_index.values()))
            assert not torch.all(
                first_tensor[..., 0] == 0.0
            ), dst  # check only first element to limit memory

    # Next step: compare the DerivedScalarStore against a reference store in blob storage.
    # The goal is to notice if the derived scalar values change unexpectedly, which may
    # indicate a bug. Note that this test will also fail for PRs that fix existing bugs that
    # affect the derived scalar values!

    def truncate_acts(acts: torch.Tensor) -> torch.Tensor:
        for dim in range(acts.ndim):
            acts = acts.narrow(dim=dim, start=0, length=min(50, acts.size(dim)))
        return acts

    # Truncate all the activations in the store, in preparation for comparing them against the
    # reference data, which is also truncated. We truncate to avoid needing to save and load
    # excessively large tensors.
    current_ds_store = current_ds_store.apply_transform_fn_to_activations(truncate_acts)

    # If you need to update the reference data, do the following in exactly this order:
    #   1) Increment the version number in the reference data path to avoid changing the data
    #      used by the version of this test running on CI.
    #   2) Uncomment the "save_to_file" line below.
    #   3) Run the test.
    #   4) Comment the "save_to_file" line out again.
    #
    # Think carefully before doing this! Make sure you're confident that the new data is
    # correct. Note that changes to the autoencoder stored at AUTOENCODER_TEST_PATH will also
    # affect the derived scalar values.
    # current_ds_store.save_to_file(reference_ds_store_path)
    #
    # In case of a deleted DST, you can temporarily set skip_missing_dsts=True below to run the tests, confirming
    # that all other DSTs are still correct. Then, increment the version number in the reference data path as
    # described above, save the new data, and set skip_missing_dsts=False again.

    reference_ds_store = DerivedScalarStore.load_from_file(
        reference_ds_store_path, map_location=standard_model_context.device, skip_missing_dsts=False
    )

    current_dsts_and_pass_types = set(
        current_ds_store.activations_and_metadata_by_dst_and_pass_type.keys()
    )
    reference_dsts_and_pass_types = set(
        reference_ds_store.activations_and_metadata_by_dst_and_pass_type.keys()
    )

    something_has_failed = False
    failed_dsts_and_pass_types = []

    for dst_and_pass_type in current_dsts_and_pass_types.intersection(
        reference_dsts_and_pass_types
    ):
        print(f"Comparing {dst_and_pass_type}")
        current_activations_and_metadata = (
            current_ds_store.activations_and_metadata_by_dst_and_pass_type[dst_and_pass_type]
        )
        reference_activations_and_metadata = (
            reference_ds_store.activations_and_metadata_by_dst_and_pass_type[dst_and_pass_type]
        )
        # TODO(sbills): Refactor this to share code with the __eq__ method in
        # ActivationsAndMetadata.
        try:
            layer_index: int | None = None
            assert (
                current_activations_and_metadata.activations_by_layer_index.keys()
                == reference_activations_and_metadata.activations_by_layer_index.keys()
            ), (
                "current:",
                current_activations_and_metadata.activations_by_layer_index.keys(),
                "reference:",
                reference_activations_and_metadata.activations_by_layer_index.keys(),
            )
            for layer_index in current_activations_and_metadata.activations_by_layer_index.keys():
                # show the first value for every sequence token
                activations_ndim = current_activations_and_metadata.activations_by_layer_index[
                    layer_index
                ].ndim
                slices_to_show = tuple([slice(None)] + [0] * (activations_ndim - 1))
                assert torch.allclose(
                    current_activations_and_metadata.activations_by_layer_index[layer_index],
                    reference_activations_and_metadata.activations_by_layer_index[layer_index],
                    rtol=5e-4,
                    atol=5e-4,
                ), (
                    "current:",
                    current_activations_and_metadata.activations_by_layer_index[layer_index][
                        slices_to_show
                    ],
                    "reference:",
                    reference_activations_and_metadata.activations_by_layer_index[layer_index][
                        slices_to_show
                    ],
                )
        except AssertionError as e:
            print(f"Discrepancy in {dst_and_pass_type}, layer {layer_index}: {e}")
            something_has_failed = True
            failed_dsts_and_pass_types.append(dst_and_pass_type)

    assert not something_has_failed, (
        "Some derived scalar values have changed unexpectedly: "
        f"{failed_dsts_and_pass_types}\n"
        "See the comments in this test case for instructions on how to update the reference data."
    )

    # Calculate which dsts are missing from current_ds_store and which dsts are added to
    # current_ds_store, relative to reference_ds_store.
    missing_dsts = reference_ds_store.dsts - current_ds_store.dsts
    added_dsts = current_ds_store.dsts - reference_ds_store.dsts
    assert len(missing_dsts) == 0 and len(added_dsts) == 0, (
        f"The following DSTs have been removed relative to the reference data: {missing_dsts}\n"
        f"The following DSTs have been added relative to the reference data:   {added_dsts}\n"
        f"If these changes are intentional, update the reference data by following the process "
        f"documented in a comment in this test case."
    )
