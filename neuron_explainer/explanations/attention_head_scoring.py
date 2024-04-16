"""Uses API calls to score attention head explanations."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from neuron_explainer.activations.activations import (
    ActivationRecord,
    ActivationRecordSliceParams,
    load_neuron,
)
from neuron_explainer.activations.attention_utils import (
    convert_flattened_index_to_unflattened_index,
)
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.explainer import (
    ATTENTION_EXPLANATION_PREFIX,
    ContextSize,
    format_attention_head_token_pair_string,
)
from neuron_explainer.explanations.explanations import (
    AttentionSimulation,
    ScoredAttentionSimulation,
)
from neuron_explainer.explanations.few_shot_examples import ATTENTION_HEAD_FEW_SHOT_EXAMPLES
from neuron_explainer.explanations.prompt_builder import (
    ChatMessage,
    PromptBuilder,
    PromptFormat,
    Role,
)


class AttentionHeadOneAtATimeScorer:
    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.CHAT_MESSAGES,
        # This parameter lets us adjust the length of the prompt when we're generating explanations
        # using older models with shorter context windows. In the future we can use it to experiment
        # with longer context windows.
        context_size: ContextSize = ContextSize.FOUR_K,
        repeat_strongly_attending_pairs: bool = False,
        max_concurrent: int | None = 10,
        cache: bool = False,
    ):
        assert (
            prompt_format == PromptFormat.CHAT_MESSAGES
        ), f"Unhandled prompt format {prompt_format}"

        self.prompt_format = prompt_format
        self.context_size = context_size
        self.client = ApiClient(model_name=model_name, max_concurrent=max_concurrent, cache=cache)
        self.repeat_strongly_attending_pairs = repeat_strongly_attending_pairs

    async def score_explanation(
        self,
        explanation: str,
        activation_records: list[ActivationRecord],
        max_activation: float,
        # The number of high and low activating token pairs to sample for simulation
        num_activations_for_scoring: int = 5,
        # The activation threshold below which a token pair is eligible for sampling
        # as a low activating pair.
        low_activation_threshold: float = 0.1,
    ) -> ScoredAttentionSimulation:
        """Score explanations based on how well they predict attention between
        top attending token pairs and random low attending token pairs."""
        # Use the activation records to generate a set of pairs for scoring.
        # 10 pairs: the five top activating pairs, and five randomly chosen pairs
        # where the activations are below 0.1 * the max value.
        candidates = []
        for i, record in enumerate(activation_records):
            sorted_activation_flat_indices = np.argsort(record.activations)[::-1]
            sorted_vals = [record.activations[idx] for idx in sorted_activation_flat_indices]
            coordinates = [
                convert_flattened_index_to_unflattened_index(flat_index)
                for flat_index in sorted_activation_flat_indices
            ]
            candidates.extend([(i, val, coords) for val, coords in zip(sorted_vals, coordinates)])
        top_activation_coordinates = [
            (candidate[0], candidate[2])
            for candidate in sorted(candidates, key=lambda x: x[1], reverse=True)
        ][:num_activations_for_scoring]

        filtered_low_activation_coordinates = [
            (candidate[0], candidate[2])
            for candidate in candidates
            if candidate[1] < low_activation_threshold * max_activation
        ]
        selected_low_activation_coordinates = random.sample(
            filtered_low_activation_coordinates,
            min(len(filtered_low_activation_coordinates), num_activations_for_scoring),
        )

        attention_simulations = []
        true_labels = [1 for _ in range(len(top_activation_coordinates))] + [
            0 for _ in range(len(selected_low_activation_coordinates))
        ]
        # No need to shuffle because the model only sees one at a time anyway.
        for coords, label in zip(
            top_activation_coordinates + selected_low_activation_coordinates, true_labels
        ):
            activation_record = activation_records[coords[0]]
            # for each pair, generate a prompt where the model is asked to predict if the token pair has a strong
            # or weak activation.
            prompt = self.make_token_pair_prompt(explanation, activation_record.tokens, coords[1])

            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)  # Really a ChatMessage
            generate_kwargs: dict[str, Any] = {
                # Using a timeout prevents the scorer from hanging if the API server is overloaded.
                "timeout": 60,
                "n": 1,
                "max_tokens": 1,  # we only want to sample one token.
                "logprobs": True,
                "top_logprobs": 15,
                "messages": prompt,
            }
            response = await self.client.async_generate(**generate_kwargs)
            assert len(response["choices"]) == 1

            # from the response, extract the logit values for "0" (for weak) and "1" (for strong) to obtain
            # a float.
            choice = response["choices"][0]
            # for whatever reason `choice["logprobs"]["top_logprobs"]` is a list of dicts
            logprobs_dicts = choice["logprobs"]["content"][0]["top_logprobs"]
            extracted_probs = {d["token"]: d["logprob"] for d in logprobs_dicts}
            zero_prob = np.exp(extracted_probs["0"]) if "0" in extracted_probs else 0.0
            one_prob = np.exp(extracted_probs["1"]) if "1" in extracted_probs else 0.0
            total_prob = zero_prob + one_prob
            # The score is 0 * normalized probability of "0" + 1 * normalized probability of "1", which
            # reduces to just the normalized probability of "1".
            normalized_one_prob = one_prob / total_prob
            # print(f"zero_prob: {zero_prob/total_prob}, one_prob: {normalized_one_prob}")
            attention_simulations.append(
                AttentionSimulation(
                    tokens=activation_record.tokens,
                    token_pair_coords=coords[1],
                    token_pair_label=label,
                    simulation_prediction=normalized_one_prob,
                )
            )

        assert (
            len(attention_simulations)
            == len(true_labels)
            == len(top_activation_coordinates) + len(selected_low_activation_coordinates)
        )

        # ROC AUC awards a perfect score to explanations that order all of the scores
        # for pairs labeled "1" above the scores for pairs labeled "0" (even if the scores
        # for pairs labeled "0" are well above 0).
        score = roc_auc_score(
            y_true=true_labels, y_score=[sim.simulation_prediction for sim in attention_simulations]
        )
        return ScoredAttentionSimulation(
            attention_simulations=attention_simulations,
            roc_auc_score=score,
        )

    def make_token_pair_prompt(
        self, explanation: str, tokens: list[str], coords: tuple[int, int]
    ) -> str | list[ChatMessage]:
        """
        Create a prompt to send to the API to simulate the model predicting whether a token pair
        has a strong attention write norm according to the given explanation.
        """
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying attention heads in a neural network. Each head looks at every pair of tokens "
            "in a short token sequence and activates for pairs of tokens that fit what it is looking for. "
            "Attention heads always attend from a token to a token earlier in the sequence (or from a "
            "token to itself). We will display a token sequence and indicate a particular token pair within "
            'that sequence. The "to" token of the pair will be marked with double asterisks (e.g., **token**) '
            'and the "from" token will be marked with double square brackets (e.g., [[token]]). If the token pair '
            "consists of a token paired with itself, it will be marked with both (e.g., [[**token**]]) and "
            "no other token in the sequence will be marked. We present an explanation of what the "
            "attention head is looking for. Output 1 if the head activates for the token pair, and 0 otherwise.",
        )
        num_few_shot = 0
        for few_shot_example in ATTENTION_HEAD_FEW_SHOT_EXAMPLES:
            if not few_shot_example.simulation_examples:
                continue
            for simulation_example in few_shot_example.simulation_examples:
                self._add_per_token_pair_attention_simulation_prompt(
                    prompt_builder=prompt_builder,
                    tokens=few_shot_example.token_pair_examples[
                        simulation_example.token_pair_example_index
                    ].tokens,
                    explanation=few_shot_example.explanation,
                    simulation_coords=simulation_example.token_pair_coordinates,
                    index=num_few_shot,
                    label=simulation_example.label,
                )
                num_few_shot += 1

        self._add_per_token_pair_attention_simulation_prompt(
            prompt_builder=prompt_builder,
            tokens=tokens,
            explanation=explanation,
            simulation_coords=coords,
            index=num_few_shot,
            label=None,
        )

        return prompt_builder.build(self.prompt_format)

    def _add_per_token_pair_attention_simulation_prompt(
        self,
        prompt_builder: PromptBuilder,
        tokens: list[str],
        explanation: str,
        simulation_coords: tuple[int, int],
        index: int,
        label: int | None,  # None means this is the end of the full prompt.
    ) -> None:
        user_message = f"""

Example {index + 1}
Explanation: {ATTENTION_EXPLANATION_PREFIX} {explanation.strip()}
Sequence:\n{format_attention_head_token_pair_string(tokens, simulation_coords)}"""
        if self.repeat_strongly_attending_pairs:
            user_message += (
                f"\nThe same token pair, presented as (to_token, from_token): "
                f"({tokens[simulation_coords[1]]}, {tokens[simulation_coords[0]]})"
            )

        user_message += (
            f"\nPrediction of whether attention head {index + 1} activates on the token pair: "
        )
        prompt_builder.add_message(Role.USER, user_message)

        if label is not None:
            prompt_builder.add_message(Role.ASSISTANT, f"{label}")


if __name__ == "__main__":
    # Example usage
    async def main() -> None:
        scorer = AttentionHeadOneAtATimeScorer("gpt-4-turbo")
        explanation = "attends from tokens to the first token in the sequence"
        attention_head = load_neuron(
            "https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2_small/attn_write_norm/collated_activations_by_token_pair",
            "0",
            "5",
        )
        train_records = attention_head.train_activation_records(
            activation_record_slice_params=ActivationRecordSliceParams(n_examples_per_split=5)
        )
        scored_simulation = await scorer.score_explanation(
            explanation, train_records, max([max(record.activations) for record in train_records])
        )
        print(scored_simulation.roc_auc_score)

    import asyncio

    asyncio.run(main())
