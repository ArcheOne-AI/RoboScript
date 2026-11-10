"""The two arms of a P4 evaluation (`wicl` and the registered `wicl+rsi`): the same Agent
host phase, the same Actor prompt, and with an RSI document that has no readable entry the same
prompt byte for byte; with entries, the document's parts after it and nothing else changed."""

from pathlib import Path

from roboscript.agent.overlay import NO_OVERLAY, SEPARATOR, overlay_loader
from roboscript.contracts.artifact import ArtifactLayer, ArtifactRef, ArtifactStack
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MOCK
from roboscript.learning.agent_host import actor_prompt, host_phase
from roboscript.learning.method import load_method
from roboscript.memory.artifact import KIND as TASK_MEMORY
from roboscript.testing.memory import sample_memory
from roboscript.tools.skillsets import guides as skillset_guides
from roboscript_methods.rsi.doc import Evidence, RsiDoc, RsiEntry
from roboscript_methods.wicl.method import Wicl
from roboscript_methods.wicl_rsi import WiclRsi

RECIPE = Path("recipes/wicl")
TASK = TaskName("lift_red")
MEMORY = ArtifactRef(kind=TASK_MEMORY, name=TASK, version="v000", hash="1" * 64)


def _prompt(method: type, instruction: str = "lift the red block") -> str:
    runtime = method.runtime(RECIPE)
    phase = host_phase(method, runtime)
    guides = skillset_guides(MOCK.skillsets)
    prompt, _ = actor_prompt(runtime.prompts({}, guides), load_embodiment(MOCK_BIMANUAL), phase)
    return prompt(instruction, sample_memory(TASK))


def _entry(entry_id: str) -> RsiEntry:
    evidence = Evidence(block="block1", task=TASK, episode=0)
    return RsiEntry(
        id=entry_id,
        when="the red block rests on the table",
        do="grasp it from above",
        check="it rises",
        evidence=(evidence,),
        added=1,
    )


async def _wrapped(doc: RsiDoc) -> str:
    ref = ArtifactRef(kind="rsi", name="G1", version="v000", hash=doc.content_hash())

    async def load(layer_ref: ArtifactRef) -> RsiDoc:
        assert layer_ref == ref
        return doc

    stack = ArtifactStack(
        layers=(
            ArtifactLayer(ref=MEMORY, task=TASK),
            ArtifactLayer(ref=ref),
            ArtifactLayer(ref=ref, task=TASK),
        )
    )
    overlay = await overlay_loader(load, TASK_MEMORY)(TASK, stack.for_task(TASK))
    if not doc.entries():
        assert overlay == NO_OVERLAY
    return overlay.wrap(_prompt(WiclRsi))


def test_wicl_rsi_is_registered_and_hosts_wicls_eval_phase() -> None:
    method = load_method("wicl+rsi")
    assert method.name == "wicl+rsi"
    assert host_phase(method, method.runtime(RECIPE)) == "p3"
    assert host_phase(Wicl, Wicl.runtime(RECIPE)) == "p3"
    assert WiclRsi.runtime(RECIPE).mounts["p3"] == Wicl.runtime(RECIPE).mounts["p3"]


def test_the_arms_read_the_same_actor_prompt() -> None:
    assert _prompt(WiclRsi) == _prompt(Wicl)
    assert _prompt(WiclRsi, "stack the bowls") == _prompt(Wicl, "stack the bowls")


async def test_an_empty_document_leaves_the_prompt_byte_identical() -> None:
    assert (await _wrapped(RsiDoc())).encode() == _prompt(Wicl).encode()


async def test_entries_come_after_the_prompt_and_change_nothing_else() -> None:
    doc = RsiDoc(
        global_entries=(_entry("G-01"),),
        task_entries={TASK: (_entry("T-lift_red-01"),)},
    )
    wrapped = await _wrapped(doc)
    base = _prompt(Wicl)
    assert wrapped.startswith(base + SEPARATOR)
    rest = wrapped[len(base + SEPARATOR) :]
    assert rest == SEPARATOR.join((doc.context(None), doc.context(TASK)))
    assert "[G-01]" in rest and "[T-lift_red-01]" in rest
