"""
Class for performing inference on a model in real time, i.e. in the context of interactive user
requests.

Requests consist of two parts: an InferenceRequestSpec and a XRequestSpec.
- InferenceRequestSpec contains the information necessary for computing a forward and backward
  pass on a model (prompt, loss function, in future ablation info).
- XRequestSpec contains the information necessary for computing a derived scalar of type X
  (activation, derived scalar type, layer index, etc.) This can include information necessary for
  inserting hooks into the model, e.g. in the case of the online autoencoder latent.

Functions for handling requests first compute DerivedScalarStore, then call a helper function that
takes XRequestSpec + DerivedScalarStore as input. CombinedRequestSpec contains
InferenceRequestSpec + a list of [XRequestSpec, YRequestSpec, ...]. It first computes
DerivedScalarStore, then calls relevant helper functions to generate a response containing sub
responses for the various sub request specs.
"""

import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, TypeVar

import torch
from fastapi import HTTPException

from neuron_explainer.activation_server.derived_scalar_computation import (
    DerivedScalarComputationParams,
    DstAndConfigsByProcessingStep,
    InferenceAndTokenData,
    InferenceData,
    compute_derived_scalar_groups_for_input_token_ints,
    maybe_construct_loss_fn_for_backward_pass,
)
from neuron_explainer.activation_server.dst_helpers import (
    assert_tensor,
    get_intermediate_sum_by_dst,
)
from neuron_explainer.activation_server.load_neurons import load_neuron_from_datasets
from neuron_explainer.activation_server.read_routes import (
    TokenAndRawAttentionScalars,
    normalize_attention_token_scalars,
    zip_tokens_and_attention_activations,
)
from neuron_explainer.activation_server.requests_and_responses import *
from neuron_explainer.activation_server.tdb_conversions import (
    convert_tdb_request_spec_to_inference_sub_request,
)
from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    DerivedScalarIndex,
    MirroredNodeIndex,
    NodeIndex,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.multi_group import (
    MultiGroupDerivedScalarStore,
    MultiGroupScalarDerivers,
)
from neuron_explainer.activations.derived_scalars.postprocessing import (
    DerivedScalarPostprocessor,
    TokenPairAttributionConverter,
    TokenReadConverter,
    TokenWriteConverter,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig
from neuron_explainer.activations.derived_scalars.tokens import (
    TopTokens,
    get_most_upvoted_and_downvoted_tokens_for_nodes,
)
from neuron_explainer.models.autoencoder_context import AutoencoderContext, MultiAutoencoderContext
from neuron_explainer.models.inference_engine_type_registry import InferenceEngineType
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import ModelContext, StandardModelContext
from neuron_explainer.models.transformer import Transformer
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable

T = TypeVar("T")


TOKEN_READ_DSTS: list[DerivedScalarType] = [
    DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION,
]


PROMPT_LENGTH_LIMIT = 500


@immutable
class TopKData(CamelCaseBaseModel):
    """The results of an individual top-k operation, which is represented as a TopKParams."""

    activations: list[float]
    node_indices: list[MirroredNodeIndex]
    vocab_token_strings_for_indices: list[str | None] | None
    # This is the total of all activations, including non-top-k activations.
    intermediate_sum_activations_by_dst: dict[DerivedScalarType, TensorND]


@dataclass(frozen=True)
class RequestResponseCorrespondence:
    request_class: type
    request_spec_class: type
    request_spec_name: str
    response_class: type
    response_data_class: type
    response_data_name: str


REQUEST_RESPONSE_CORRESPONDENCE_REGISTRY: list[RequestResponseCorrespondence] = [
    RequestResponseCorrespondence(
        request_class=DerivedScalarsRequest,
        request_spec_class=DerivedScalarsRequestSpec,
        request_spec_name="derived_scalars_request_spec",
        response_class=DerivedScalarsResponse,
        response_data_class=DerivedScalarsResponseData,
        response_data_name="derived_scalars_response_data",
    ),
    RequestResponseCorrespondence(
        request_class=DerivedAttentionScalarsRequest,
        request_spec_class=DerivedAttentionScalarsRequestSpec,
        request_spec_name="derived_attention_scalars_request_spec",
        response_class=DerivedAttentionScalarsResponse,
        response_data_class=DerivedAttentionScalarsResponseData,
        response_data_name="derived_attention_scalars_response_data",
    ),
    RequestResponseCorrespondence(
        request_class=MultipleTopKDerivedScalarsRequest,
        request_spec_class=MultipleTopKDerivedScalarsRequestSpec,
        request_spec_name="multiple_top_k_derived_scalars_request_spec",
        response_class=MultipleTopKDerivedScalarsResponse,
        response_data_class=MultipleTopKDerivedScalarsResponseData,
        response_data_name="multiple_top_k_derived_scalars_response_data",
    ),
    RequestResponseCorrespondence(
        request_class=ScoredTokensRequest,
        request_spec_class=ScoredTokensRequestSpec,
        request_spec_name="scored_tokens_request_spec",
        response_class=ScoredTokensResponse,
        response_data_class=ScoredTokensResponseData,
        response_data_name="scored_tokens_response_data",
    ),
    RequestResponseCorrespondence(
        request_class=TokenPairAttributionRequest,
        request_spec_class=TokenPairAttributionRequestSpec,
        request_spec_name="token_pair_attribution_request_spec",
        response_class=TokenPairAttributionResponse,
        response_data_class=TokenPairAttributionResponseData,
        response_data_name="token_pair_attribution_response_data",
    ),
]


def get_corresponding_object(
    object: str | type, object_category: str, desired_category: str
) -> str | type:
    correspondence_of_interest = [
        correspondence
        for correspondence in REQUEST_RESPONSE_CORRESPONDENCE_REGISTRY
        if getattr(correspondence, object_category) == object
    ]
    assert (
        len(correspondence_of_interest) == 1
    ), f"Found {len(correspondence_of_interest)} correspondences for {object_category} {object}"
    return getattr(correspondence_of_interest[0], desired_category)


def _make_vocab_token_string_for_node_index(
    model_context: ModelContext, node_index: NodeIndex
) -> str | None:
    if node_index.node_type == NodeType.VOCAB_TOKEN:
        last_index = node_index.tensor_indices[-1]
        if last_index is None or last_index >= model_context.n_vocab:
            return None
        return model_context.decode_token(last_index)
    return None


def _make_vocab_token_strings_for_indices(
    model_context: ModelContext, activation_indices: list[NodeIndex]
) -> list[str | None] | None:
    vocab_token_strings_for_indices = [
        _make_vocab_token_string_for_node_index(model_context, node_index)
        for node_index in activation_indices
    ]
    if all(vocab_token_string is None for vocab_token_string in vocab_token_strings_for_indices):
        return None
    else:
        return vocab_token_strings_for_indices


def _unique_list_in_original_order(original_list: list[T]) -> list[T]:
    """
    Returns a list containing the unique elements of the original list, in the same order as the
    original list. `list(set(original_list))` does not preserve the original order.
    """
    unique_list = []
    seen = set()
    for item in original_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


class InteractiveModel:
    def __init__(
        self,
        transformer: Transformer,
        standard_model_context: StandardModelContext,
        autoencoder_context: AutoencoderContext | MultiAutoencoderContext | None = None,
    ) -> None:
        self.transformer = transformer
        self._standard_model_context = standard_model_context
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            autoencoder_context
        )
        # We only allow one batched request to be handled at a time. Concurrent batched requests
        # tend to result in cuda OOMs.
        self._batched_request_lock = asyncio.Lock()

    @property
    def has_mlp_autoencoder(self) -> bool:
        return self._multi_autoencoder_context is not None and any(
            node_type == NodeType.MLP_AUTOENCODER_LATENT
            for node_type in self._multi_autoencoder_context.autoencoder_context_by_node_type.keys()
        )

    @property
    def has_attention_autoencoder(self) -> bool:
        return self._multi_autoencoder_context is not None and any(
            node_type == NodeType.ATTENTION_AUTOENCODER_LATENT
            for node_type in self._multi_autoencoder_context.autoencoder_context_by_node_type.keys()
        )

    def get_model_info(
        self, mlp_autoencoder_name: str | None, attn_autoencoder_name: str | None
    ) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=self._standard_model_context.model_name,
            has_mlp_autoencoder=self.has_mlp_autoencoder,
            has_attention_autoencoder=self.has_attention_autoencoder,
            n_layers=self._standard_model_context.n_layers,
            mlp_autoencoder_name=mlp_autoencoder_name,
            attention_autoencoder_name=attn_autoencoder_name,
        )

    def encode(self, string: str) -> list[int]:
        return self._standard_model_context.encode(string)

    @classmethod
    def from_model_name(cls, model_name: str) -> "InteractiveModel":
        standard_model_context = StandardModelContext(model_name=model_name)
        return cls.from_standard_model_context(standard_model_context)

    @classmethod
    def from_standard_model_context(
        cls, standard_model_context: StandardModelContext
    ) -> "InteractiveModel":
        return cls(
            transformer=standard_model_context.get_or_create_model(),
            standard_model_context=standard_model_context,
        )

    @classmethod
    def from_standard_model_context_and_autoencoder_context(
        cls,
        standard_model_context: StandardModelContext,
        autoencoder_context: AutoencoderContext | MultiAutoencoderContext,
    ) -> "InteractiveModel":
        return cls(
            transformer=standard_model_context.get_or_create_model(),
            standard_model_context=standard_model_context,
            autoencoder_context=autoencoder_context,
        )

    async def _handle_inference_request(
        self, inference_request: InferenceRequest
    ) -> InferenceResponse:
        request_type = type(inference_request)
        processing_spec_name = get_corresponding_object(
            request_type, "request_class", "request_spec_name"
        )
        assert isinstance(processing_spec_name, str)
        response_class = get_corresponding_object(request_type, "request_class", "response_class")
        assert isinstance(response_class, type)
        response_data_name = get_corresponding_object(
            request_type, "request_class", "response_data_name"
        )
        assert isinstance(response_data_name, str)

        processing_request_spec = getattr(inference_request, processing_spec_name)
        # We handle the singular case by wrapping the request in a batched request, to avoid the need to
        # special-case non-batched requests.
        batched_request = BatchedRequest(
            inference_sub_requests=[
                InferenceSubRequest(
                    inference_request_spec=inference_request.inference_request_spec,
                    processing_request_spec_by_name={processing_spec_name: processing_request_spec},
                )
            ]
        )
        batched_response = await self.handle_batched_request(batched_request)
        assert len(batched_response.inference_sub_responses) == 1
        sub_response = batched_response.inference_sub_responses[0]
        return response_class(
            inference_and_token_data=sub_response.inference_response.inference_and_token_data,
            **{
                response_data_name: sub_response.processing_response_data_by_name[
                    processing_spec_name
                ]
            },
        )

    async def get_derived_scalars(
        self, inference_request: DerivedScalarsRequest
    ) -> DerivedScalarsResponse:
        response = await self._handle_inference_request(inference_request)
        assert isinstance(response, DerivedScalarsResponse)
        return response

    async def _get_derived_scalars_from_ds_store(
        self,
        request_spec: DerivedScalarsRequestSpec,
        ds_store: DerivedScalarStore,
    ) -> DerivedScalarsResponseData:
        ds_index = DerivedScalarIndex(
            dst=request_spec.dst,
            pass_type=request_spec.pass_type,
            layer_index=request_spec.layer_index,
            tensor_indices=(None, request_spec.activation_index),
        )
        activations = ds_store[ds_index]

        index_of_sequence = NodeIndex.from_ds_index(ds_index)
        index_base_dict = asdict(index_of_sequence)
        index_base_dict.pop("tensor_indices")
        activations_to_return = assert_tensor(activations)
        assert activations_to_return.ndim == 1, activations_to_return.shape
        indices_to_return = []
        for token_index in range(activations_to_return.shape[0]):
            indices_to_return.append(
                MirroredNodeIndex(
                    **index_base_dict,
                    tensor_indices=(token_index,) + index_of_sequence.tensor_indices[1:],
                )
            )

        if request_spec.normalize_activations_using_neuron_record is None:
            normalized_activations = None
        else:
            _, neuron_record = await load_neuron_from_datasets(
                request_spec.normalize_activations_using_neuron_record
            )
            normalized_activations = (
                torch.clamp(activations_to_return, min=0) / neuron_record.max_activation
            ).tolist()

        return DerivedScalarsResponseData(
            activations=activations_to_return.tolist(),
            normalized_activations=normalized_activations,
            node_indices=indices_to_return,
            top_tokens=self._get_top_tokens(
                request_spec, ds_store, indices_to_return, activations_to_return
            ),
        )

    def _get_top_tokens(
        self,
        request_spec: DerivedScalarsRequestSpec,
        ds_store: DerivedScalarStore,
        indices_to_return: list[MirroredNodeIndex],
        activations_to_return: torch.Tensor,
    ) -> TopTokens | None:
        top_and_bottom_t_tokens_upvoted = request_spec.num_top_tokens
        if top_and_bottom_t_tokens_upvoted is None:
            # This data wasn't requested.
            return None
        else:
            assert top_and_bottom_t_tokens_upvoted > 0
            if request_spec.dst in TOKEN_READ_DSTS:
                # This DST is used for calculating token reads: basically, which vocab tokens most
                # "upvote" the node of interest. In this case, we don't use TokenWriteConverter.
                # Instead, we perform the usual top-t logic on the raw activations from the
                # DerivedScalarStore.
                token_write = ds_store[
                    DerivedScalarIndex(
                        dst=request_spec.dst,
                        pass_type=request_spec.pass_type,
                        layer_index=None,
                        tensor_indices=(0, None),  # (First sequence token, all activations)
                    )
                ]

                # In some cases the token write may be all 0s, for example if we're handling an
                # autoencoder latent whose activation on this token is zero. Return None in this
                # case.
                if torch.all(token_write == 0):
                    return None

                # We never need to do the flipping logic for token reads, since we use an ablation
                # to force the gradient to be positive.
                flip_upvoted_and_downvoted = False
            else:
                # The request is using a regular DST. We apply the TokenWriteConverter to get the
                # token write for one of the requested nodes.
                token_write_converter = TokenWriteConverter(
                    model_context=self._standard_model_context,
                    multi_autoencoder_context=self._multi_autoencoder_context,
                )
                token_write = token_write_converter.postprocess(
                    # We can use any of the indices to return, since they all match in all aspects
                    # except the sequence token index.
                    indices_to_return[0],
                    ds_store,
                )
                if activations_to_return[0] == 0:
                    # If the activation is 0, we don't get any information about the top and bottom
                    # tokens. In GELU models this should only happen for autoencoder latents, which
                    # tend to have sparse activations.
                    return None
                else:
                    # If the activation is negative, the top and bottom tokens need to be flipped.
                    # This means that we swap upvoted/downvoted and positive/negative for the
                    # associated scalars.
                    flip_upvoted_and_downvoted = activations_to_return[0].item() < 0

        # Unsqueeze to get the shape expected by the helper functions that do the top-t logic.
        token_write = token_write.unsqueeze(0)
        assert (
            token_write.ndim == 2
        ), f"Expected token_write.ndim == 2, but got {token_write.shape=}"
        assert torch.isfinite(
            token_write
        ).all(), "token_write tensor should only contain finite values"
        return get_most_upvoted_and_downvoted_tokens_for_nodes(
            self._standard_model_context,
            token_write,
            top_and_bottom_t_tokens_upvoted,
            flip_upvoted_and_downvoted,
        )[0]

    async def get_derived_attention_scalars(
        self, inference_request: DerivedAttentionScalarsRequest
    ) -> DerivedAttentionScalarsResponse:
        response = await self._handle_inference_request(inference_request)
        assert isinstance(response, DerivedAttentionScalarsResponse)
        return response

    async def _get_derived_attention_scalars_from_ds_store(
        self,
        request_spec: DerivedAttentionScalarsRequestSpec,
        ds_store: DerivedScalarStore,
        tokens_as_ints: list[int],
    ) -> DerivedAttentionScalarsResponseData:
        if request_spec.dst == DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM:
            # Dimensions for DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM: (
            #     Dimension.SEQUENCE_TOKENS,
            #     Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
            #     Dimension.ATTN_HEADS,
            # )
            head_index = request_spec.activation_index
            tensor_indices = (None, None, head_index)  # type: tuple[int | None, ...]
        elif request_spec.dst == DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS:
            # Dimensions for DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS: (
            #     Dimension.SEQUENCE_TOKENS,
            #     Dimension.ATTENDED_TO_SEQUENCE_TOKENS,
            #     Dimension.SINGLETON,
            # )
            tensor_indices = (None, None, 0)
        else:
            raise NotImplementedError(f"Unsupported DST: {request_spec.dst}")
        ds_index = DerivedScalarIndex(
            dst=request_spec.dst,
            pass_type=PassType.FORWARD,
            layer_index=request_spec.layer_index,
            tensor_indices=tensor_indices,
        )
        activations = assert_tensor(ds_store[ds_index])
        assert activations.ndim == 2

        token_and_raw_attention_scalars_list: list[TokenAndRawAttentionScalars] = []
        tokens_as_strings = [
            self._standard_model_context.decode_token(token) for token in tokens_as_ints
        ]
        assert len(tokens_as_strings) == activations.shape[0] == activations.shape[1]
        for i in range(len(tokens_as_strings)):
            # We already indexed by the attention head (last dimension), so now we index by sequence
            # token and attended-to token. For the attended-to token, we want the current token and
            # all preceding tokens. (Subsequent tokens are masked.) This flattened representation is
            # a bit odd, but it's what the normalization function expects.
            scalars = activations[i, : i + 1]
            assert scalars.ndim == 1, scalars.ndim
            token_and_raw_attention_scalars_list.append(
                TokenAndRawAttentionScalars(
                    token=tokens_as_strings[i],
                    scalars=scalars.tolist(),
                )
            )

        list_of_sequence_lists = [[token_and_raw_attention_scalars_list]]
        if request_spec.normalize_activations_using_neuron_record is not None:
            _, neuron_record = await load_neuron_from_datasets(
                request_spec.normalize_activations_using_neuron_record
            )
            # We add the most positive activation records to the list of sequence lists used for
            # normalization. We don't care about the results for those sequences, but including them
            # means that we'll get the appropriate max values for normalization.
            list_of_sequence_lists.append(
                zip_tokens_and_attention_activations(neuron_record.most_positive_activation_records)
            )

        # This function handles nested lists, so we need to nest and unnest when invoking it.
        # If we added the most positive activation records to the list of sequence lists, they will
        # effectively be dropped when we index into the result (they're at index 1).
        token_and_attention_scalars_list = normalize_attention_token_scalars(
            list_of_sequence_lists
        )[0][0]
        return DerivedAttentionScalarsResponseData(
            token_and_attention_scalars_list=token_and_attention_scalars_list
        )

    async def get_scored_tokens(self, request: ScoredTokensRequest) -> ScoredTokensResponse:
        response = await self._handle_inference_request(request)
        assert isinstance(response, ScoredTokensResponse)
        return response

    def _get_token_scoring_postprocessor(
        self, token_scoring_type: TokenScoringType
    ) -> DerivedScalarPostprocessor:
        match token_scoring_type:
            case TokenScoringType.UPVOTED_OUTPUT_TOKENS:
                return TokenWriteConverter(
                    model_context=self._standard_model_context,
                    multi_autoencoder_context=self._multi_autoencoder_context,
                )
            case (
                TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_Q
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_K
            ):
                return TokenReadConverter(
                    model_context=self._standard_model_context,
                    multi_autoencoder_context=self._multi_autoencoder_context,
                )
            case _:
                raise NotImplementedError(f"Unsupported token_scoring_type: {token_scoring_type}")

    def _should_score_node(self, node_type: NodeType, token_scoring_type: TokenScoringType) -> bool:
        match token_scoring_type:
            case TokenScoringType.UPVOTED_OUTPUT_TOKENS:
                return True
            case TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP:
                return node_type in [
                    NodeType.MLP_NEURON,
                    NodeType.AUTOENCODER_LATENT,
                    NodeType.MLP_AUTOENCODER_LATENT,
                ]
            case (
                TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_Q
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_K
            ):
                # TODO(dan): These token scoring types are currently disabled. Work through the
                # errors, then re-enable them.
                # return node_type in [NodeType.ATTENTION_HEAD, NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR]
                return False
            case _:
                raise NotImplementedError(f"Unsupported token_scoring_type: {token_scoring_type}")

    def _transform_node_indices_for_attn_q_or_k(
        self,
        all_node_indices: list[NodeIndex],
        activation_location_type: ActivationLocationType,
    ) -> list[NodeIndex]:
        return [
            (
                node_index.to_subnode_index(activation_location_type)
                # mypy has trouble figuring out the type of this list comprehension.
                if node_index.node_type == NodeType.ATTENTION_HEAD
                else node_index
            )
            for node_index in all_node_indices
        ]

    def _transform_node_indices(
        self, all_node_indices: list[NodeIndex], token_scoring_type: TokenScoringType
    ) -> list[NodeIndex]:
        match token_scoring_type:
            case (
                TokenScoringType.UPVOTED_OUTPUT_TOKENS
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP
            ):
                return all_node_indices
            case TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_Q:
                return self._transform_node_indices_for_attn_q_or_k(
                    all_node_indices, ActivationLocationType.ATTN_QUERY
                )
            case TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_K:
                return self._transform_node_indices_for_attn_q_or_k(
                    all_node_indices, ActivationLocationType.ATTN_KEY
                )
            case _:
                raise NotImplementedError(f"Unhandled token_scoring_type: {token_scoring_type}")

    def _get_group_id_for_token_scoring(self, token_scoring_type: TokenScoringType) -> GroupId:
        match token_scoring_type:
            case TokenScoringType.UPVOTED_OUTPUT_TOKENS:
                return GroupId.TOKEN_WRITE
            case (
                TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_Q
                | TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_ATTN_K
            ):
                return GroupId.TOKEN_READ
            case _:
                raise NotImplementedError(f"Unhandled token_scoring_type: {token_scoring_type}")

    def _get_scored_tokens(
        self,
        request_spec: ScoredTokensRequestSpec,
        multi_group_ds_store: MultiGroupDerivedScalarStore,
        all_node_indices: list[NodeIndex],
    ) -> ScoredTokensResponseData:
        # Do postprocessing for all of the nodes to get the top and bottom tokens for that node.
        # Use the derived scalars associated with the TOKEN_WRITE or TOKEN_READ group.
        token_scoring_type = request_spec.token_scoring_type
        ds_store = multi_group_ds_store.get_ds_store(
            self._get_group_id_for_token_scoring(token_scoring_type)
        )
        postprocessor = self._get_token_scoring_postprocessor(token_scoring_type)
        should_score_node = [
            self._should_score_node(node_index.node_type, token_scoring_type)
            for node_index in all_node_indices
        ]
        all_node_indices_to_score = [
            node_index
            for node_index, should_score in zip(all_node_indices, should_score_node)
            if should_score
        ]
        if len(all_node_indices_to_score) > 0:
            scored_token_scalars_list = postprocessor.postprocess_multiple_nodes(
                all_node_indices_to_score, ds_store
            )
            token_scalars_list = [
                scored_token_scalars_list.pop(0) if should_score else None
                for should_score in should_score_node
            ]
        else:
            token_scalars_list = [None for _ in range(len(all_node_indices))]

        non_none_token_scalars = [ts for ts in token_scalars_list if ts is not None]
        if len(non_none_token_scalars) == 0:
            # This can happen in situations where the token scoring type doesn't apply to any of the
            # nodes.
            top_tokens_list = []
        else:
            token_scalars_tensor = torch.stack(non_none_token_scalars)
            assert (
                token_scalars_tensor.ndim == 2
            ), f"Expected token_writes.ndim == 2, but got {token_scalars_tensor.shape=}"
            assert torch.isfinite(
                token_scalars_tensor
            ).all(), "token_scalars_tensor should only contain finite values"
            top_tokens_list = get_most_upvoted_and_downvoted_tokens_for_nodes(
                self._standard_model_context, token_scalars_tensor, request_spec.num_tokens
            )

        # Now create a version of top_tokens_list that has the same length as all_node_indices, with
        # None at the indices for the nodes we didn't score.
        final_top_tokens_list: list[TopTokens | None] = []
        top_tokens_index = 0
        for token_scalars in token_scalars_list:
            if token_scalars is None:
                # If the node wasn't scored, add None.
                final_top_tokens_list.append(None)
            else:
                # If the node was scored, add the corresponding TopTokens from top_tokens_list.
                final_top_tokens_list.append(top_tokens_list[top_tokens_index])
                top_tokens_index += 1

        assert top_tokens_index == len(top_tokens_list)
        assert len(final_top_tokens_list) == len(all_node_indices)

        return ScoredTokensResponseData(
            node_indices=[
                MirroredNodeIndex.from_node_index(node_index) for node_index in all_node_indices
            ],
            top_tokens_list=final_top_tokens_list,
        )

    def _get_token_pair_attribution(
        self,
        request_spec: TokenPairAttributionRequestSpec,
        multi_group_ds_store: MultiGroupDerivedScalarStore,
        all_node_indices: list[NodeIndex],
    ) -> TokenPairAttributionResponseData:
        """Returns attended-to tokens with most positive attributions for attention write autoencoder latents."""
        ds_store = multi_group_ds_store.get_ds_store(group_id=GroupId.TOKEN_PAIR_ATTRIBUTION)
        postprocessor = TokenPairAttributionConverter(
            model_context=self._standard_model_context,
            multi_autoencoder_context=self._multi_autoencoder_context,
            num_tokens_attended_to=request_spec.num_tokens_attended_to,
        )

        # sort the top token-attended-to by the value of the attribution
        node_indices = []  # type: list[MirroredNodeIndex]
        top_tokens_attended_to_list = []  # type: list[TopTokensAttendedTo | None]
        for node_index in all_node_indices:
            node_indices.append(MirroredNodeIndex.from_node_index(node_index))
            try:
                postprocessed = postprocessor.postprocess(node_index, ds_store)
                top_tokens_attended_to = postprocessed.topk(k=request_spec.num_tokens_attended_to)
                top_tokens_attended_to_list.append(
                    TopTokensAttendedTo(
                        token_indices=top_tokens_attended_to.indices.cpu().numpy().tolist(),
                        attributions=top_tokens_attended_to.values.cpu().numpy().tolist(),
                    )
                )
            except ValueError:
                top_tokens_attended_to_list.append(None)
                continue

        return TokenPairAttributionResponseData(
            node_indices=node_indices, top_tokens_attended_to_list=top_tokens_attended_to_list
        )

    async def get_multiple_top_k_derived_scalars(
        self, request: MultipleTopKDerivedScalarsRequest
    ) -> MultipleTopKDerivedScalarsResponse:
        """This request is assumed to have multiple group_ids, where values within each group_id
        are comparable (for example a group called "write_norm" might have MLP write norm and attention write norm;
        or "act_times_grad" might have MLP post-act act*grad and attention post-softmax act*grad). Across group IDs,
        the values are assumed to be attributable to the same set of node types (e.g. MLP neurons; or attention heads).

        Example:
        ---------------------------------------------------------
        | Group ID       | DerivedScalarType   | NodeType       |
        ---------------------------------------------------------
        | write_norm     | mlp_write_norm      | mlp_neuron     |
        | write_norm     | attn_write_norm     | attention_head |
        | act_times_grad | mlp_act_times_grad  | mlp_neuron     |
        | act_times_grad | attn_act_times_grad | attention_head |
        ---------------------------------------------------------

        The response contains a list of NodeIndex objects, which identify a NodeType and e.g. token_index, neuron_index
        tuple. These indices correspond to derived scalar values that are extremal for some derived scalar type. It also contains
        a dict of the corresponding derived scalar values, keyed by group_id, where the i'th element of each list corresponds to
        the i'th NodeIndex in the list of ActivationIndices.
        """

        response = await self._handle_inference_request(request)
        assert isinstance(response, MultipleTopKDerivedScalarsResponse)
        return response

    def _compute_multi_group_ds_store(
        self,
        # In the future we may want to pass in a single list of compound types instead of three parallel lists.
        batched_inference_request_spec: list[InferenceRequestSpec],
        # Sometimes T will be GroupId; sometimes it will be tuple[str, GroupId].
        batched_dst_and_configs_by_processing_step: list[DstAndConfigsByProcessingStep],
        # Return three parallel lists:
        #   1) a batched list of input token ints
        #   2) a batched list of derived scalar stores
        #   3) a batched list of inference data objects
        # Each list should be the same length.
    ) -> tuple[
        list[list[int]],
        list[dict[str, MultiGroupDerivedScalarStore]],
        list[InferenceData],
    ]:
        """
        Helper method (first step) that computes a DerivedScalarStore for each group ID. The
        quantities within each group ID are intended to be comparable (e.g. the write vector norms
        of attention heads, and the write vector norms of MLP neurons).
        """
        assert len(batched_inference_request_spec) == len(
            batched_dst_and_configs_by_processing_step
        )
        batched_ds_computation_params = []
        for inference_request_spec, dst_and_configs_by_processing_step in zip(
            batched_inference_request_spec,
            batched_dst_and_configs_by_processing_step,
        ):
            prompt = inference_request_spec.prompt
            unpadded_tokens_as_ints = self.encode(prompt)
            tokens_as_ints = unpadded_tokens_as_ints

            multi_group_scalar_derivers_by_processing_step = {
                spec_name: (
                    MultiGroupScalarDerivers.from_dst_and_config_list_by_group_id(
                        dst_and_config_list_by_group_id=dst_and_configs_by_processing_step[
                            spec_name
                        ],
                    )
                )
                for spec_name in dst_and_configs_by_processing_step.keys()
            }

            # if at least one of the scalar derivers is intended to operate on GPU,
            # then we'll use GPU for the raw activations. Otherwise the CPU.
            devices_for_raw_activations = []
            for (
                multi_group_scalar_derivers
            ) in multi_group_scalar_derivers_by_processing_step.values():
                devices_for_raw_activations += (
                    multi_group_scalar_derivers.devices_for_raw_activations
                )
            if any(device.type == "cuda" for device in devices_for_raw_activations):
                device_for_raw_activations = torch.device("cuda", 0)
            elif any(device.type == "mps" for device in devices_for_raw_activations):
                device_for_raw_activations = torch.device("mps")
            else:
                device_for_raw_activations = torch.device("cpu")
            trace_config = (  # mirrored -> non-mirrored TraceConfig
                inference_request_spec.trace_config.to_trace_config()
                if inference_request_spec.trace_config is not None
                else None
            )
            ds_computation_params = DerivedScalarComputationParams(
                input_token_ints=tokens_as_ints,
                multi_group_scalar_derivers_by_processing_step=multi_group_scalar_derivers_by_processing_step,
                loss_fn_for_backward_pass=maybe_construct_loss_fn_for_backward_pass(
                    model_context=self._standard_model_context,
                    config=inference_request_spec.loss_fn_config,
                ),
                trace_config=trace_config,
                ablation_specs=inference_request_spec.ablation_specs,
                device_for_raw_activations=device_for_raw_activations,
            )
            batched_ds_computation_params.append(ds_computation_params)

        batched_multi_group_ds_store_by_processing_step: list[
            dict[str, MultiGroupDerivedScalarStore]
        ]
        (
            batched_multi_group_ds_store_by_processing_step,
            batched_inference_data,
            _,
        ) = compute_derived_scalar_groups_for_input_token_ints(
            model_context=self._standard_model_context,
            batched_ds_computation_params=batched_ds_computation_params,
            multi_autoencoder_context=self._multi_autoencoder_context,
        )

        return (
            [params.input_token_ints for params in batched_ds_computation_params],
            batched_multi_group_ds_store_by_processing_step,
            batched_inference_data,
        )

    def _get_multiple_top_k_derived_scalars_from_multi_group_ds_store(
        self,
        request_spec: MultipleTopKDerivedScalarsRequestSpec,
        multi_group_ds_store: MultiGroupDerivedScalarStore,
        all_node_indices: list[NodeIndex],
    ) -> MultipleTopKDerivedScalarsResponseData:
        """
        Helper method (the second and final step) that computes top k activations for each group
        name starting from a pre-computed DerivedScalarStore, per group ID. The quantities within
        each group ID are intended to be comparable (e.g. the write vector norms of attention
        heads, and the write vector norms of MLP neurons).

        This computes the top k model component, token combinations, and returns them in a
        MultipleTopKDerivedScalarsResponseData object.
        """
        assert len(all_node_indices) > 0, "Expected at least one node index"

        activations_by_group_id = (
            multi_group_ds_store.get_derived_scalars_by_group_id_for_node_indices(all_node_indices)
        )

        intermediate_sum_by_dst_by_group_id: dict[GroupId, dict[DerivedScalarType, TensorND]] = {}

        for group_id in activations_by_group_id.keys():
            intermediate_sum_by_dst_by_group_id[group_id] = compute_intermediate_sum_by_dst(
                ds_store=multi_group_ds_store.get_ds_store(group_id),
                dimensions_to_keep_for_intermediate_sum=request_spec.dimensions_to_keep_for_intermediate_sum,
            )

        return MultipleTopKDerivedScalarsResponseData(
            activations_by_group_id={
                group_id: activations.tolist()
                for group_id, activations in activations_by_group_id.items()
            },
            node_indices=[
                MirroredNodeIndex.from_node_index(node_index) for node_index in all_node_indices
            ],
            vocab_token_strings_for_indices=_make_vocab_token_strings_for_indices(
                self._standard_model_context,
                all_node_indices,
            ),
            intermediate_sum_activations_by_dst_by_group_id=intermediate_sum_by_dst_by_group_id,
        )

    def _get_dst_and_configs_by_processing_step_for_singular_request(
        self, request: InferenceSubRequest
    ) -> DstAndConfigsByProcessingStep:
        """
        Wrapper that calls the helper to infer the correct configs for each sub_request_spec, and then
        performs a sanity check to confirm that backward pass activations are not being requested at any
        layers deeper than the layer from which the backward pass is being computed.

        Returns:
            dst_and_config_list_dict: a nested dict of lists of tuples, where each tuple
                contains a DerivedScalarType and a DerivedScalarTypeConfig. The nested dict is keyed first by
                spec_name and then by group_id, where spec_name is the name of the processing_request_spec, and group_id
                refers to a GroupId enum value (each GroupId referring to a set of DSTs).
        """
        inference_request_spec = request.inference_request_spec
        processing_request_spec_by_name = request.processing_request_spec_by_name

        dst_and_configs_by_processing_step: DstAndConfigsByProcessingStep = {}
        for spec_name, processing_request_spec in processing_request_spec_by_name.items():
            dst_and_configs_by_processing_step[
                spec_name
            ] = self._get_dst_and_config_list_by_group_id_from_request_spec(
                inference_request_spec=inference_request_spec,
                processing_request_spec=processing_request_spec,
                preceding_dst_and_config_lists=dst_and_configs_by_processing_step,
            )

        return dst_and_configs_by_processing_step

    async def handle_batched_tdb_request(
        self, batched_tdb_request: BatchedTdbRequest
    ) -> BatchedResponse:
        inference_sub_requests = [
            convert_tdb_request_spec_to_inference_sub_request(tdb_request_spec)
            for tdb_request_spec in batched_tdb_request.sub_requests
        ]
        # TODO(sbills): Return a TDB-specific response rather than just returning a regular
        # BatchedResponse.
        return await self.handle_batched_request(
            BatchedRequest(inference_sub_requests=inference_sub_requests)
        )

    async def handle_batched_request(self, batched_request: BatchedRequest) -> BatchedResponse:
        """For high level overview, see STEP 0, 1, 2 below"""
        async with self._batched_request_lock:
            # STEP 0: infer the derived scalar types and configs needed for each group ID, by
            # examining the processing_request_spec for each group ID, and the information in the
            # inference_request_spec shared by all group IDs.
            batched_dst_and_config_list_by_processing_step = []
            for inference_request in batched_request.inference_sub_requests:
                batched_dst_and_config_list_by_processing_step.append(
                    self._get_dst_and_configs_by_processing_step_for_singular_request(
                        inference_request
                    )
                )

            # Confirm that the prompts for all of the batched (multiple) top-k sub-requests have the
            # the same length (in tokens). If they don't, we can't aggregate node indices across
            # them. They should also be less than PROMPT_LENGTH_LIMIT tokens long.
            prompt_lengths = []
            batched_tokens_as_ints = []
            for inference_request in batched_request.inference_sub_requests:
                tokens_as_ints = self._standard_model_context.encode(
                    inference_request.inference_request_spec.prompt
                )
                if any(
                    isinstance(spec, MultipleTopKDerivedScalarsRequestSpec)
                    for spec in inference_request.processing_request_spec_by_name.values()
                ):
                    batched_tokens_as_ints.append(tokens_as_ints)
                    prompt_lengths.append(len(tokens_as_ints))

            if any(prompt_length > PROMPT_LENGTH_LIMIT for prompt_length in prompt_lengths):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompts must be less than {PROMPT_LENGTH_LIMIT} tokens long for batched top-k requests. "
                        f"Got these prompt lengths: {prompt_lengths}"
                    ),
                )

            if len(set(prompt_lengths)) > 1:
                # Build an error message that gives the tokenized prompts (as strings) with their
                # lengths.
                tokens_as_strings_list = [
                    self._standard_model_context.decode_token_list(tokens_as_ints)
                    for tokens_as_ints in batched_tokens_as_ints
                ]
                prompt_lengths_str = ", ".join(
                    f"{tokens_as_strings} ({len(tokens_as_strings)} tokens)\n"
                    for tokens_as_strings in tokens_as_strings_list
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"All prompts must have the same length for batched top-k requests. "
                        f"Got these prompts:\n{prompt_lengths_str}"
                    ),
                )

            # STEP 1: run the forward pass, saving the activations and metadata required for each of
            # the derived scalar types and configs needed for each group ID; then compute the
            # derived scalars for each group ID from the common set of activations and metadata
            # objects.
            (
                batched_tokens_as_ints,
                batched_multi_group_ds_store_by_processing_step,
                batched_inference_data,
            ) = self._compute_multi_group_ds_store(
                batched_inference_request_spec=[
                    inference_request.inference_request_spec
                    for inference_request in batched_request.inference_sub_requests
                ],
                batched_dst_and_configs_by_processing_step=batched_dst_and_config_list_by_processing_step,
            )

            # Identify the node indices of the top (and possibly bottom) k activations for each
            # group sub-request, collated by spec name. We need to know all of the indices before we
            # can put together the results, since we ensure that all sub-requests cover all nodes
            # that were in the top k for any sub-request.
            # NOTE: all_node_indices is separated by spec name (i.e. processing step) but cumulative
            # across the batch index
            all_node_indices_by_spec_name: dict[str, list[NodeIndex]] = defaultdict(list)
            for (
                request,
                multi_group_ds_store_by_processing_step,
            ) in zip(
                batched_request.inference_sub_requests,
                batched_multi_group_ds_store_by_processing_step,
            ):
                for (
                    processing_spec_name,
                    processing_request_spec,
                ) in request.processing_request_spec_by_name.items():
                    if isinstance(processing_request_spec, MultipleTopKDerivedScalarsRequestSpec):
                        (
                            transform_activations_fn_for_top_k,
                            transform_indices_fn_for_top_k,
                        ) = maybe_construct_transform_fns_for_top_k(
                            processing_request_spec.token_index,
                        )
                        multi_group_ds_store = multi_group_ds_store_by_processing_step[
                            processing_spec_name
                        ]
                        # if top_and_bottom_k is not provided, returns all elements
                        node_indices, _ = multi_group_ds_store.topk(
                            top_and_bottom_k=processing_request_spec.top_and_bottom_k,
                            transform_activations_fn=transform_activations_fn_for_top_k,
                            transform_indices_fn=transform_indices_fn_for_top_k,
                        )
                        all_node_indices_by_spec_name[processing_spec_name] += node_indices
                    elif isinstance(
                        processing_request_spec,
                        (ScoredTokensRequestSpec, TokenPairAttributionRequestSpec),
                    ):
                        # These request specs always depend on another spec in the top-level
                        # request and use the same node indices.
                        all_node_indices_by_spec_name[
                            processing_spec_name
                        ] = all_node_indices_by_spec_name[
                            processing_request_spec.depends_on_spec_name
                        ]
            # Dedupe the node indices, maintaining the original order.
            all_node_indices_by_spec_name = {
                spec_name: _unique_list_in_original_order(all_node_indices)
                for spec_name, all_node_indices in all_node_indices_by_spec_name.items()
            }

            # STEP 2: compute the response data for each sub request spec; this can be simply
            # extracting a particular derived scalar at a particular activation index, or can be
            # computing the top k derived scalars, or can be computing the top k of multiple
            # scalars, and combining them.
            batched_inference_responses = []
            assert (
                len(batched_request.inference_sub_requests)
                == len(batched_multi_group_ds_store_by_processing_step)
                == len(batched_dst_and_config_list_by_processing_step)
                == len(batched_tokens_as_ints)
                == len(batched_inference_data)
            )
            for (
                inference_sub_request,
                multi_group_ds_store_by_processing_step,
                tokens_as_ints,
                inference_data,
            ) in zip(
                batched_request.inference_sub_requests,
                batched_multi_group_ds_store_by_processing_step,
                batched_tokens_as_ints,
                batched_inference_data,
            ):
                processing_response_data_by_name: dict[str, ProcessingResponseData] = {}
                for (
                    processing_spec_name,
                    processing_request_spec,
                ) in inference_sub_request.processing_request_spec_by_name.items():
                    relevant_multi_group_ds_store = multi_group_ds_store_by_processing_step[
                        processing_spec_name
                    ]
                    # The key may not present in the case where a non-top-k request uses a spec name
                    # that is not present in any of the (multi) top-k requests, or in the case where
                    # there are no (multi) top-k requests at all.
                    all_node_indices: list[NodeIndex] = all_node_indices_by_spec_name.get(
                        processing_spec_name, []
                    )
                    processing_response_data = await self._handle_processing_request(
                        processing_request_spec=processing_request_spec,
                        multi_group_ds_store=relevant_multi_group_ds_store,
                        all_node_indices=all_node_indices,
                        tokens_as_ints=tokens_as_ints,
                    )
                    expected_response_data_class = get_corresponding_object(
                        type(processing_request_spec), "request_spec_class", "response_data_class"
                    )
                    assert isinstance(expected_response_data_class, type)
                    assert isinstance(
                        processing_response_data,
                        expected_response_data_class,
                    )
                    processing_response_data_by_name[
                        processing_spec_name
                    ] = processing_response_data
                # put the sub responses together in one object to be returned
                combined_response = InferenceResponseAndResponseDict(
                    inference_response=InferenceResponse(
                        inference_and_token_data=self._get_inference_response_data(
                            tokens_as_ints=tokens_as_ints,
                            inference_data=inference_data,
                        ),
                    ),
                    processing_response_data_by_name=processing_response_data_by_name,
                )
                # check that the type of each sub response data is as expected given the type of the
                # corresponding sub request spec
                for name in combined_response.processing_response_data_by_name.keys():
                    processing_request_spec = inference_sub_request.processing_request_spec_by_name[
                        name
                    ]
                    expected_response_data_class = get_corresponding_object(
                        type(processing_request_spec), "request_spec_class", "response_data_class"
                    )
                    assert isinstance(expected_response_data_class, type)
                    assert isinstance(
                        combined_response.processing_response_data_by_name[name],
                        expected_response_data_class,
                    ), "Response data class mismatch; perhaps cast to wrong type by Pydantic?"
                batched_inference_responses.append(combined_response)

            return BatchedResponse(inference_sub_responses=batched_inference_responses)

    async def _handle_processing_request(
        self,
        processing_request_spec: ProcessingRequestSpec,
        multi_group_ds_store: MultiGroupDerivedScalarStore,
        all_node_indices: list[NodeIndex],
        tokens_as_ints: list[int],
    ) -> ProcessingResponseData:
        """
        Figure out the type of the processing_request_spec and handle it with the appropriate helper method
        to return ResponseData of the appropriate type (e.g. DerivedScalarsRequestSpec ->
        DerivedScalarsResponseData).
        """
        if isinstance(processing_request_spec, DerivedAttentionScalarsRequestSpec):
            ds_store = multi_group_ds_store.to_single_ds_store()
            return await self._get_derived_attention_scalars_from_ds_store(
                request_spec=processing_request_spec,
                ds_store=ds_store,
                tokens_as_ints=tokens_as_ints,
            )
        elif isinstance(processing_request_spec, DerivedScalarsRequestSpec):
            ds_store = multi_group_ds_store.to_single_ds_store()
            return await self._get_derived_scalars_from_ds_store(
                request_spec=processing_request_spec,
                ds_store=ds_store,
            )
        elif isinstance(processing_request_spec, MultipleTopKDerivedScalarsRequestSpec):
            return self._get_multiple_top_k_derived_scalars_from_multi_group_ds_store(
                request_spec=processing_request_spec,
                multi_group_ds_store=multi_group_ds_store,
                all_node_indices=all_node_indices,
            )
        elif isinstance(processing_request_spec, ScoredTokensRequestSpec):
            return self._get_scored_tokens(
                request_spec=processing_request_spec,
                multi_group_ds_store=multi_group_ds_store,
                all_node_indices=all_node_indices,
            )
        elif isinstance(processing_request_spec, TokenPairAttributionRequestSpec):
            return self._get_token_pair_attribution(
                request_spec=processing_request_spec,
                multi_group_ds_store=multi_group_ds_store,
                all_node_indices=all_node_indices,
            )
        else:
            raise ValueError(f"Unhandled request_spec type: {type(processing_request_spec)}")

    def _get_inference_response_data(
        self, tokens_as_ints: list[int], inference_data: InferenceData
    ) -> InferenceAndTokenData:
        return InferenceAndTokenData(
            tokens_as_ints=tokens_as_ints,
            tokens_as_strings=self._standard_model_context.decode_token_list(tokens_as_ints),
            **inference_data.dict(),
        )

    def _get_dst_config(
        self,
        inference_request_spec: InferenceRequestSpec,
        dsts: list[DerivedScalarType],
        pass_type: PassType,
    ) -> DstConfig:
        """
        Infers the DerivedScalarTypeConfig required to compute the given DerivedScalarType(s) and pass_type.
        This auto-populates:
        derive_gradients based on the requested pass_type
        model_context and autoencoder context based on the InteractiveModel properties themselves
        layer_index_for_grad based on whether the backward pass originates from the loss at the model's outputs,
        or based on a particular activation at an intermediate location in the network
        """
        if inference_request_spec.activation_index_for_within_layer_grad is not None:
            trace_config = TraceConfig.from_activation_index(
                activation_index=(
                    inference_request_spec.activation_index_for_within_layer_grad.to_activation_index()
                )  # convert mirrored to un-mirrored trace config
            )
        elif inference_request_spec.trace_config is not None:
            # starts as MirroredTraceConfig, but DSTConfig takes TraceConfig
            trace_config = inference_request_spec.trace_config.to_trace_config()
        else:
            trace_config = None

        if DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS in dsts:
            assert inference_request_spec.activation_index_for_within_layer_grad is not None
            layer_index = inference_request_spec.activation_index_for_within_layer_grad.layer_index
            assert layer_index is not None
            layer_indices = [layer_index]  # type: list[int] | None
        else:
            layer_indices = None

        if any(
            dsts == [read_dst] for read_dst in TOKEN_READ_DSTS
        ):  # requesting just one token read dst
            # This is a special DST used for calculating top "input tokens" / token reads. When it's
            # used, there should always be an ablation that specifies the activation index of
            # interest, which needs to be plumbed through the DstConfig using a dedicated field.
            ablation_specs = inference_request_spec.ablation_specs
            assert ablation_specs is not None
            assert len(ablation_specs) == 1
            activation_index_for_fake_grad = ablation_specs[0].index.to_activation_index()
        else:
            activation_index_for_fake_grad = None

        return DstConfig(
            derive_gradients=(pass_type == PassType.BACKWARD),
            inference_engine_type=InferenceEngineType.STANDARD,
            model_context=self._standard_model_context,
            layer_indices=layer_indices,
            multi_autoencoder_context=self._multi_autoencoder_context,
            trace_config=trace_config,
            activation_index_for_fake_grad=activation_index_for_fake_grad,
        )

    def _singleton_dst_and_config_list_by_group_id(
        self,
        dst: DerivedScalarType,
        inference_request_spec: InferenceRequestSpec,
        pass_type: PassType,
    ) -> dict[GroupId, list[tuple[DerivedScalarType, DstConfig]]]:
        return {
            GroupId.SINGLETON: [
                (
                    dst,
                    self._get_dst_config(
                        inference_request_spec=inference_request_spec,
                        dsts=[dst],
                        pass_type=pass_type,
                    ),
                )
            ]
        }

    def _get_dst_and_config_list_by_group_id_from_request_spec(
        self,
        inference_request_spec: InferenceRequestSpec,
        processing_request_spec: ProcessingRequestSpec,
        # Some request specs piggyback on the same DSTs and configs as others in the request.
        # Those request specs must be ordered after their dependencies.
        preceding_dst_and_config_lists: DstAndConfigsByProcessingStep,
    ) -> dict[GroupId, list[tuple[DerivedScalarType, DstConfig]]]:
        """
        Wraps the helper that infers DerivedScalarTypeConfig, parsing the information on requested
        DerivedScalarType objects from the request_spec using logic that depends
        on the type of the request_spec (i.e. the kinds of information on model internals requested).

        This auto-populates:
        derive_gradients based on the requested pass_type
        model_context and autoencoder context based on the InteractiveModel properties themselves
        layer_index_for_grad based on whether the backward pass originates from the loss at the model's outputs,
        or based on a particular activation at an intermediate location in the network
        """
        pass_type = getattr(processing_request_spec, "pass_type", PassType.FORWARD)

        if isinstance(
            processing_request_spec, (DerivedScalarsRequestSpec, DerivedAttentionScalarsRequestSpec)
        ):
            dst_and_config_list_by_group_id = self._singleton_dst_and_config_list_by_group_id(
                dst=processing_request_spec.dst,
                inference_request_spec=inference_request_spec,
                pass_type=pass_type,
            )
        elif isinstance(processing_request_spec, MultipleTopKDerivedScalarsRequestSpec):
            dst_config_by_group_id = {
                group_id: self._get_dst_config(
                    inference_request_spec=inference_request_spec,
                    dsts=dst_list,
                    pass_type=pass_type,
                )
                for group_id, dst_list in processing_request_spec.dst_list_by_group_id.items()
            }
            dst_and_config_list_by_group_id = {
                group_id: [
                    (
                        dst,
                        dst_config_by_group_id[group_id],
                    )
                    for dst in dst_list
                ]
                for group_id, dst_list in processing_request_spec.dst_list_by_group_id.items()
            }
        elif isinstance(processing_request_spec, ScoredTokensRequestSpec):
            token_scoring_type = processing_request_spec.token_scoring_type
            postprocessor = self._get_token_scoring_postprocessor(token_scoring_type)
            group_id_for_request = self._get_group_id_for_token_scoring(token_scoring_type)
            dst_and_config_list_by_group_id = self._get_dst_and_config_list_from_preceding_requests(
                processing_request_spec=processing_request_spec,
                preceding_dst_and_config_lists=preceding_dst_and_config_lists,
                postprocessor=self._get_token_scoring_postprocessor(token_scoring_type),
                group_id_for_request=self._get_group_id_for_token_scoring(token_scoring_type),
            )
        elif isinstance(processing_request_spec, TokenPairAttributionRequestSpec):
            postprocessor = TokenPairAttributionConverter(
                model_context=self._standard_model_context,
                multi_autoencoder_context=self._multi_autoencoder_context,
                num_tokens_attended_to=processing_request_spec.num_tokens_attended_to,
            )
            group_id_for_request = GroupId.TOKEN_PAIR_ATTRIBUTION
            dst_and_config_list_by_group_id = self._get_dst_and_config_list_from_preceding_requests(
                processing_request_spec=processing_request_spec,
                preceding_dst_and_config_lists=preceding_dst_and_config_lists,
                postprocessor=postprocessor,
                group_id_for_request=group_id_for_request,
            )
        else:
            raise NotImplementedError(f"Unknown request spec type: {type(processing_request_spec)}")

        return dst_and_config_list_by_group_id

    def _get_dst_and_config_list_from_preceding_requests(
        self,
        processing_request_spec: ScoredTokensRequestSpec | TokenPairAttributionRequestSpec,
        preceding_dst_and_config_lists: DstAndConfigsByProcessingStep,
        postprocessor: DerivedScalarPostprocessor,
        group_id_for_request: GroupId,
    ) -> dict[GroupId, list[tuple[DerivedScalarType, DstConfig]]]:
        # Find the first dst_and_config_list that has the spec_name we're depending on.
        # NOTE(dan): should this be union rather than first dst_and_config_list?
        assert hasattr(processing_request_spec, "depends_on_spec_name")
        dst_and_config_list = None
        for spec_name in preceding_dst_and_config_lists.keys():
            if spec_name == processing_request_spec.depends_on_spec_name:
                dst_and_config_list = next(iter(preceding_dst_and_config_lists[spec_name].values()))
                break
        assert dst_and_config_list is not None
        dst_and_config_list_by_group_id = {
            group_id_for_request: postprocessor.get_input_dst_and_config_list(
                requested_dst_and_config_list=dst_and_config_list
            )
        }
        return dst_and_config_list_by_group_id


def maybe_construct_transform_fns_for_top_k(
    token_index: int | None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, Callable[[NodeIndex], NodeIndex] | None]:
    """
    In some cases, we may want to compute the top k activations on a transformed version of the DerivedScalarStore.
    This function constructs a transformation to apply to the activations, and a corresponding transformation to
    apply to the NodeIndex objects returned by the .topk operation, if any, to index the un-transformed DerivedScalarStore.
    `None` means no transformation is to be performed.
    Currently, we use this for computing top k over a single token index only, and converting the NodeIndex objects
    to index the original, un-transformed DerivedScalarStore.
    """
    if token_index is None:
        return None, None
    else:

        def access_token_index(x: torch.Tensor) -> torch.Tensor:
            return x[token_index].unsqueeze(0)

        def convert_to_original_token_index(node_index: NodeIndex) -> NodeIndex:
            assert (
                node_index.tensor_indices[0] == 0
            )  # assumed that token dimension was converted to singleton
            # replace the 0 token index with the original token_index
            return node_index.with_updates(
                tensor_indices=(token_index, *node_index.tensor_indices[1:])
            )

        return access_token_index, convert_to_original_token_index


def compute_intermediate_sum_by_dst(
    ds_store: DerivedScalarStore,
    dimensions_to_keep_for_intermediate_sum: list[Dimension],
) -> dict[DerivedScalarType, TensorND]:
    # Calculate intermediate sums.
    def keep_dimension_fn(dim: Dimension) -> bool:
        return dim in dimensions_to_keep_for_intermediate_sum

    intermediate_sum_by_dst = get_intermediate_sum_by_dst(
        ds_store=ds_store,
        keep_dimension_fn=keep_dimension_fn,
    )

    return intermediate_sum_by_dst
