import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
)
from neuron_explainer.activations.derived_scalars.indexing import DerivedScalarIndex, NodeIndex
from neuron_explainer.activations.derived_scalars.postprocessing import (
    TokenReadConverter,
    TokenWriteConverter,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DerivedScalarType, DstConfig
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import Dimension, NodeType, PassType
from neuron_explainer.models.model_context import StandardModelContext

AUTOENCODER_TEST_DST = DerivedScalarType.MLP_POST_ACT


def test_read_and_write(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    # test token-space read and write postprocessing, for defined MLP and autoencoder latents
    prompt = "This is a test"

    dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
    )
    starting_dst_and_config_list = [
        (DerivedScalarType.MLP_POST_ACT, dst_config),
        (DerivedScalarType.ONLINE_AUTOENCODER_LATENT, dst_config),
    ]
    postprocessor_by_kind = {
        "write": TokenWriteConverter(
            model_context=standard_model_context,
            multi_autoencoder_context=standard_autoencoder_context,
        ),
        "read": TokenReadConverter(
            model_context=standard_model_context,
            multi_autoencoder_context=standard_autoencoder_context,
        ),
    }

    post_dst_and_config_list_by_kind = {
        kind: postprocessor_by_kind[kind].get_input_dst_and_config_list(
            starting_dst_and_config_list
        )
        for kind in postprocessor_by_kind.keys()
    }

    total_post_dst_and_config_list = []
    for kind in post_dst_and_config_list_by_kind.keys():
        total_post_dst_and_config_list.extend(post_dst_and_config_list_by_kind[kind])

    unique_dst_and_config_list = []
    seen_dsts = set()
    for dst_and_config in total_post_dst_and_config_list:
        dst, _ = dst_and_config
        if dst not in seen_dsts:
            unique_dst_and_config_list.append(dst_and_config)
            seen_dsts.add(dst)

    ds_store, _, _ = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
        prompt=prompt,
        dst_and_config_list=unique_dst_and_config_list,  # type: ignore
    )

    layer_index = 5

    node_indices = [
        NodeIndex(
            node_type=NodeType.MLP_NEURON,
            layer_index=layer_index,
            tensor_indices=(0, 0),
            pass_type=PassType.FORWARD,
        ),
        NodeIndex(
            node_type=NodeType.AUTOENCODER_LATENT,
            layer_index=layer_index,
            tensor_indices=(0, 0),
            pass_type=PassType.FORWARD,
        ),
    ]

    for kind in ["write", "read"]:
        for node_index in node_indices:
            print(f"{kind} {node_index}")
            p = postprocessor_by_kind[kind]
            token_vector = p.postprocess(
                ds_store=ds_store,
                node_index=node_index,
            )
            topk = torch.topk(token_vector, 10)
            top_token_ints = topk.indices.tolist()
            top_token_strings = standard_model_context.decode_token_list(top_token_ints)
            top_token_scores = topk.values.tolist()
            activation = ds_store[
                DerivedScalarIndex.from_node_index(
                    node_index,
                    (
                        DerivedScalarType.MLP_POST_ACT
                        if node_index.node_type == NodeType.MLP_NEURON
                        else DerivedScalarType.ONLINE_AUTOENCODER_LATENT
                    ),
                )
            ]
            print(f"Activation: {activation}")
            print(f"Top tokens: {top_token_strings}")
            print(f"Top token scores: {top_token_scores}")
            assert token_vector.shape == (
                standard_model_context.get_dim_size(Dimension.VOCAB_SIZE),
            )
            if node_index.node_type == NodeType.MLP_NEURON:
                assert activation < 0.0  # for this prompt
                if kind == "write":
                    assert (
                        " infancy" in top_token_strings
                    )  # matches downvoted tokens in neuron-viewer page for neuron 5:0
                elif kind == "read":
                    assert (
                        " unison" in top_token_strings
                    )  # matches upvoting input tokens in neuron-viewer page for neuron 5:0

    ds_store, _, _ = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        autoencoder_context=standard_autoencoder_context,
        prompt=".\n\nA fake orca",
        dst_and_config_list=unique_dst_and_config_list,  # type: ignore
    )

    layer_index = 5

    node_indices = [
        NodeIndex(
            node_type=NodeType.MLP_NEURON,
            layer_index=layer_index,
            tensor_indices=(6, 0),
            pass_type=PassType.FORWARD,
        ),
        NodeIndex(
            node_type=NodeType.AUTOENCODER_LATENT,
            layer_index=layer_index,
            tensor_indices=(6, 0),
            pass_type=PassType.FORWARD,
        ),
    ]
