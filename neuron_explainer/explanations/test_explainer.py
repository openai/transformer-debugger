import asyncio
from typing import Any

from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.few_shot_examples import TEST_EXAMPLES, FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import ChatMessage, PromptFormat, Role


def setup_module(unused_module: Any) -> None:
    # Make sure we have an event loop, since the attempt to create the Semaphore in
    # ApiClient will fail without it.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def test_if_formatting() -> None:
    expected_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Neuron 1
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 1 behavior: this neuron activates for vowels.

Neuron 2
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 2 behavior:<|endofprompt|> this neuron activates for"""

    explainer = TokenActivationPairExplainer(
        model_name="gpt-4",
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        few_shot_example_set=FewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        all_activations=TEST_EXAMPLES[0].activation_records,
        max_activation=1.0,
        max_tokens_for_completion=20,
    )

    assert prompt == expected_prompt


def test_chat_format() -> None:
    expected_prompt = [
        ChatMessage(
            role=Role.SYSTEM,
            content="""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.""",
        ),
        ChatMessage(
            role=Role.USER,
            content="""

Neuron 1
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 1 behavior: this neuron activates for""",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content=" vowels.",
        ),
        ChatMessage(
            role=Role.USER,
            content="""

Neuron 2
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 2 behavior: this neuron activates for""",
        ),
    ]

    explainer = TokenActivationPairExplainer(
        model_name="gpt-4",
        prompt_format=PromptFormat.CHAT_MESSAGES,
        few_shot_example_set=FewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        all_activations=TEST_EXAMPLES[0].activation_records,
        max_activation=1.0,
        max_tokens_for_completion=20,
    )

    assert isinstance(prompt, list)
    assert isinstance(prompt[0], dict)  # Really a ChatMessage
    for actual_message, expected_message in zip(prompt, expected_prompt):
        assert actual_message["role"] == expected_message["role"]
        assert actual_message["content"] == expected_message["content"]
    assert prompt == expected_prompt
