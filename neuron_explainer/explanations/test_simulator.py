# ruff: noqa: W291

from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import ChatMessage, PromptFormat, Role
from neuron_explainer.explanations.simulator import (
    ExplanationNeuronSimulator,
    ExplanationTokenByTokenSimulator,
)


def test_make_explanation_simulation_prompt_if_format() -> None:
    expected_prompt = """We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.


Neuron 1
Explanation of neuron 1 behavior: this neuron activates for vowels
Activations: 
<start>
a	10
b	0
c	0
<end>
<start>
d	unknown
e	10
f	0
<end>



Neuron 2
Explanation of neuron 2 behavior: this neuron activates for EXPLANATION<|endofprompt|>
Activations: 
<start>
0	unknown
1	unknown
2	unknown
<end>
"""
    prompt = ExplanationNeuronSimulator(
        client=None,  # type: ignore
        explanation="EXPLANATION",
        few_shot_example_set=FewShotExampleSet.TEST,
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
    ).make_simulation_prompt(
        tokens=[str(x) for x in range(3)],
    )
    assert prompt == expected_prompt


def test_make_explanation_simulation_prompt_chat_format() -> None:
    expected_prompt = [
        ChatMessage(
            role=Role.SYSTEM,
            content="""We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
""",
        ),
        ChatMessage(
            role=Role.USER,
            content="""

Neuron 1
Explanation of neuron 1 behavior: this neuron activates for vowels""",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="""
Activations: 
<start>
a	10
b	0
c	0
<end>
<start>
d	unknown
e	10
f	0
<end>

""",
        ),
        ChatMessage(
            role=Role.USER,
            content="""

Neuron 2
Explanation of neuron 2 behavior: this neuron activates for EXPLANATION""",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="""
Activations: 
<start>
0	unknown
1	unknown
2	unknown
<end>
""",
        ),
    ]
    prompt = ExplanationNeuronSimulator(
        client=None,  # type: ignore
        explanation="EXPLANATION",
        few_shot_example_set=FewShotExampleSet.TEST,
        prompt_format=PromptFormat.CHAT_MESSAGES,
    ).make_simulation_prompt(
        tokens=[str(x) for x in range(3)],
    )

    assert isinstance(prompt, list)
    assert isinstance(prompt[0], dict)  # Really a ChatMessage
    for actual_message, expected_message in zip(prompt, expected_prompt):
        assert actual_message["role"] == expected_message["role"]
        assert actual_message["content"] == expected_message["content"]
    assert prompt == expected_prompt


def test_make_token_by_token_simulation_prompt_if_format() -> None:
    expected_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at  an explanation of what the neuron does, and try to predict its activations on a particular token.

The activation format is token<tab>activation, and activations range from 0 to 10. Most activations will be 0.

Neuron 1
Explanation of neuron 1 behavior: this neuron activates for vowels
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


Now, we're going predict the activation of a new neuron on a single token, following the same rules as the examples above. Activations still range from 0 to 10.
Neuron 2
Explanation of neuron 2 behavior: this neuron activates for numbers and nothing else
Text:
ghi

Last token in the text:
i

Last token activation, considering the token in the context in which it appeared in the text:
10


Neuron 3
Explanation of neuron 3 behavior: this neuron activates for numbers and nothing else
Text:
01

Last token in the text:
1

Last token activation, considering the token in the context in which it appeared in the text:
<|endofprompt|>"""
    prompt = ExplanationTokenByTokenSimulator(
        client=None,  # type: ignore
        explanation="EXPLANATION",
        few_shot_example_set=FewShotExampleSet.TEST,
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
    ).make_single_token_simulation_prompt(
        tokens=[str(x) for x in range(3)],
        explanation="numbers and nothing else",
        token_index_to_score=1,
    )
    assert prompt == expected_prompt


def test_make_token_by_token_simulation_prompt_chat_format() -> None:
    expected_prompt = [
        ChatMessage(
            role=Role.SYSTEM,
            content="""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at  an explanation of what the neuron does, and try to predict its activations on a particular token.

The activation format is token<tab>activation, and activations range from 0 to 10. Most activations will be 0.

""",
        ),
        ChatMessage(
            role=Role.USER,
            content="""Neuron 1
Explanation of neuron 1 behavior: this neuron activates for vowels
""",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="""Activations: 
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


""",
        ),
        ChatMessage(
            role=Role.SYSTEM,
            content="Now, we're going predict the activation of a new neuron on a single token, following the same rules as the examples above. Activations still range from 0 to 10.",
        ),
        ChatMessage(
            role=Role.USER,
            content="""
Neuron 2
Explanation of neuron 2 behavior: this neuron activates for numbers and nothing else
Text:
ghi

Last token in the text:
i

Last token activation, considering the token in the context in which it appeared in the text:
""",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="""10

""",
        ),
        ChatMessage(
            role=Role.USER,
            content="""
Neuron 3
Explanation of neuron 3 behavior: this neuron activates for numbers and nothing else
Text:
01

Last token in the text:
1

Last token activation, considering the token in the context in which it appeared in the text:
""",
        ),
    ]

    prompt = ExplanationTokenByTokenSimulator(
        client=None,  # type: ignore
        explanation="EXPLANATION",
        few_shot_example_set=FewShotExampleSet.TEST,
        prompt_format=PromptFormat.CHAT_MESSAGES,
    ).make_single_token_simulation_prompt(
        tokens=[str(x) for x in range(3)],
        explanation="numbers and nothing else",
        token_index_to_score=1,
    )

    assert isinstance(prompt, list)
    assert isinstance(prompt[0], dict)  # Really a ChatMessage
    for actual_message, expected_message in zip(prompt, expected_prompt):
        assert actual_message["role"] == expected_message["role"]
        assert actual_message["content"] == expected_message["content"]
    assert prompt == expected_prompt
