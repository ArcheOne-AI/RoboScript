"""P3 on RoboTwin (docs/concepts/learning.md): RoboTwin's own evaluation client drives the agent.

- `policy/`: the official policy package (`deploy_policy.py`, `deploy_policy.yml`), linked as
  `ROBOTWIN_ROOT/policy/RoboScript`; both official processes import it (simulator env).
- `relay`: what its `get_model` returns in the policy server, a thin client of the Agent host;
  `wire`: what the policy's `eval()` and the relay send each other.
- `install`: the reversible, manifest-tracked links into a RoboTwin checkout (agent side).
- `launch`: one official evaluation, RoboTwin's server and client around the Agent host (agent
  side), or `shards` pairs of them on the block's episodes.
- `shard_client`: one shard's client, the official client's `main` with a loop that checks every
  seed and evaluates the shard's episodes (simulator env).
"""
