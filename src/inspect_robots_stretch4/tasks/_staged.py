"""The shape every task here shares: one fixed scene, repeated, graded by the operator against a
staged 0-4 rubric.

The rubric prints at the operator's grading prompt and is saved in the log. The prompt only
takes y/n/partial/skip, so the grader types the stage (0-4) at the start of the notes.
"""

from __future__ import annotations

from inspect_robots import Scene, Task, operator_scorer

ANSWER_HINT = (
    'Answer y for 4, n for 0, partial for 1-3; start the notes with the stage, e.g. "3 - cap off".'
)


def staged_task(name: str, instruction: str, rubric: str, epochs: int, max_steps: int) -> Task:
    """One fixed scene, repeated `epochs` times, scored from the operator's verdict."""
    return Task(
        name=name,
        scenes=[Scene(id=name, instruction=instruction, metadata={"rubric": rubric})],
        scorer=operator_scorer(),
        max_steps=max_steps,
        epochs=epochs,
    )
