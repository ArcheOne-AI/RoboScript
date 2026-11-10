"""RoboScript's policy package for RoboTwin's evaluation scripts, which load `get_model`, `eval`
and `reset_model` from it by name (as `policy/Your_Policy` does)."""

from .deploy_policy import eval, get_model, reset_model

__all__ = ["eval", "get_model", "reset_model"]
