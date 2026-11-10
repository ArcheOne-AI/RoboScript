import pytest

from roboscript.contracts.errors import (
    EpisodeEnded,
    InfraError,
    Problem,
    ProtocolError,
    RSError,
    ToolError,
    error_from_problem,
)

ERRORS: list[tuple[RSError, int]] = [
    (ProtocolError("invalid_request", "bad body"), 400),
    (ProtocolError("unsupported_version", "client 2.0"), 400),
    (ProtocolError("unauthorized"), 401),
    (ProtocolError("forbidden_plane", "agent token on env.control"), 403),
    (ProtocolError("not_found", "no episode"), 404),
    (EpisodeEnded("closed"), 409),
    (ToolError("obj_3 is not registered"), 422),
    (InfraError("busy"), 503),
    (InfraError("not_ready", "loading assets"), 503),
    (InfraError("timeout"), 503),
    (InfraError("sim_fault", "physx crashed"), 503),
    (RSError("internal", "see the server log"), 500),
]


@pytest.mark.parametrize(("error", "status"), ERRORS, ids=lambda x: getattr(x, "code", ""))
def test_problem_round_trip(error: RSError, status: int) -> None:
    problem = error.to_problem()
    assert (problem.status, error.status) == (status, status)
    assert problem.type == f"urn:roboscript:problem:{error.code}"
    assert problem.retryable is isinstance(error, InfraError)
    back = error_from_problem(Problem.model_validate_json(problem.model_dump_json()))
    assert (type(back), back.code, back.detail) == (type(error), error.code, error.detail)
    assert str(back) == str(error)


def test_unknown_codes_from_a_newer_server_keep_retryable() -> None:
    problem = Problem(type="x", title="New", status=418, code="teapot", detail="d")
    error = error_from_problem(problem)
    assert type(error) is RSError and error.code == "teapot" and error.detail == "d"
    assert error.status == 500 and not error.retryable
    retry = error_from_problem(problem.model_copy(update={"code": "overloaded", "retryable": True}))
    assert type(retry) is InfraError and retry.code == "overloaded" and retry.retryable


def test_problem_ignores_extension_members() -> None:
    text = '{"type": "t", "title": "T", "status": 409, "code": "busy", "instance": "/v1/x"}'
    assert Problem.model_validate_json(text).detail is None
