# Quickstart: one episode on the Mock benchmark

Goal: see every part that takes part in an episode, in about 80 lines, without a model.

```bash
cd examples/tutorials
uv run python -m quickstart_mock.run /tmp/quickstart   # the records go there (default: a temp dir)
```

It prints how the episode ended, the event log counted by kind, and where its records are.
The code is `quickstart_mock/run.py`; the test is `tests/examples/tutorials/test_quickstart.py`.

## The parts

1. **The env** is `MockEnv`: colored boxes on a table, the `mock_bimanual` robot, three cameras,
   all on the CPU. It has the three planes every env has: `env.public` (what the env is),
   `env.control` (the runner's: open, judge, close, ground truth) and `env.agent` (what a robot
   has: observations, robot state, execute). Here one object serves all three, in process.
2. **A toolset** is a class of tools. `Sensing` reads the robot, `Hold` keeps the arms still for
   some counted actions. A `LocalProvider` mounts a toolset in this process.
3. **A phase** (`PhaseSpec`) is the method's declaration of who sees which tools, in what order, at
   which tier. Ours has one role, `actor`, at tier D (no ground truth), with three tools. The
   model sees them in the order listed.
4. **The Hub** takes the providers and the phases, and every tool call goes through its guard
   pipeline: visibility, validation, budget, action lock, the tool, rendering. It writes every
   call to the event log.
5. **The control plane opens the episode.** `env.open` returns an agent token; the Hub episode is
   opened with an `EnvBinding` carrying it. The brain never gets the control plane.
6. **The brain** is `ScriptedBrain`: each model call is answered with the next step of the
   script (`call("tool", **arguments)`). The real brain, `ResponsesBrain`, drives a model through
   the same session; everything else stays as it is.
7. **The verdict** comes from the env, not from the brain: `finish` ends the episode, and only
   `env.judge` says whether the task is done. Holding still does not lift the box, so
   `success` is `False`. (`docs/concepts/agent-runtime.md` has the pipeline and the loop.)

## Things to try

- Change `steps=3` to `steps=50`: the Mock env's step limit is 40, and the receipt says it held
  for 40 actions and why it stopped (`step_limit`).
- Add `"sensing.observe"` to the phase's visibility and `call("observe")` to the script: the
  episode records a frame and its camera images in the blob store (`blobs/` next to the log).
- Take `hold.hold` out of the visibility list: the call comes back as an error the model reads
  (`unknown_tool`), and the episode goes on.
- Read `events.jsonl` and `episode/turns.jsonl` next to it. Every request the brain sent can be
  rebuilt from the log (`roboscript.agent.rebuild_requests`).

## Next

`custom_toolset` adds a tool of your own to this episode.
