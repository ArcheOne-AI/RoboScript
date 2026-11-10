"""The person at the robot: they set the scene up and judge the outcome
(docs/how-to/add-a-real-robot.md). A real robot's reset is manual and its success is external, so
the server asks an `Operator` for both; how it asks (a console, a button box, a web page) is the
deployment's."""

from collections.abc import Iterable
from typing import Protocol

import anyio

from roboscript.contracts.ids import TaskName


class Operator(Protocol):
    async def confirm_reset(self, task: TaskName, prompt: str) -> None:
        """Show how to set the scene up for `task` and return once a person has done it."""
        ...

    async def judge(self, task: TaskName, instruction: str) -> bool:
        """Ask a person whether the episode achieved `instruction`."""
        ...


class ConsoleOperator:
    """Prompts on the server's terminal: Enter once the scene is set, y or n for the verdict."""

    async def confirm_reset(self, task: TaskName, prompt: str) -> None:
        await anyio.to_thread.run_sync(
            input, f"[reset for {task}] {prompt}\nPress Enter when the scene is ready. "
        )

    async def judge(self, task: TaskName, instruction: str) -> bool:
        while True:
            answer = await anyio.to_thread.run_sync(
                input, f"[{task}] did the robot '{instruction}'? [y/n] "
            )
            if answer.strip().lower() in ("y", "n"):
                return answer.strip().lower() == "y"


class ScriptedOperator:
    """Confirms every reset at once and judges from a script: the verdicts in order, the last one
    for every later episode. Records what it was asked, for tests and tutorials."""

    def __init__(self, verdicts: Iterable[bool] = (False,)) -> None:
        self._verdicts = list(verdicts)
        if not self._verdicts:
            raise ValueError("a scripted operator needs at least one verdict")
        self.resets: list[tuple[TaskName, str]] = []
        self.judged: list[TaskName] = []

    async def confirm_reset(self, task: TaskName, prompt: str) -> None:
        self.resets.append((task, prompt))

    async def judge(self, task: TaskName, instruction: str) -> bool:
        self.judged.append(task)
        index = min(len(self.judged), len(self._verdicts)) - 1
        return self._verdicts[index]
