"""A tiny service for the stack tests: `make_app` served by `serve`, with switches to misbehave.

Run as `python stack_service.py --ready-file <path> [...]`, as the supervisor does.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from roboscript.rpc import make_app, serve, write_ready

ENV = (
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "RS_TEST",
    "RS_ROOT",
    "RS_BLOB_DIR",
    "RS_BLOB_ROOTS",
)
STUBBORN = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--exit", type=int, help="exit with this code instead of serving")
    parser.add_argument("--never-ready", action="store_true", help="/readyz answers 503")
    parser.add_argument("--sick-file", type=Path, help="/readyz answers 503 while this exists")
    parser.add_argument("--helper", action="store_true", help="start a child ignoring SIGTERM")
    parser.add_argument("--ignore-term", action="store_true", help="ignore SIGTERM itself")
    parser.add_argument("--two-planes", action="store_true", help="serve toy.b on a second port")
    parser.add_argument("--finally-file", type=Path, help="written once serving has returned")
    parser.add_argument("--port", type=int, default=0, help="the port to listen on (0: any)")
    args = parser.parse_args()
    if args.exit is not None:
        print("exiting early on purpose", flush=True)
        sys.exit(args.exit)
    if args.sick_file is not None:
        args.sick_file.unlink(missing_ok=True)  # a restart is healthy again
    helper = subprocess.Popen([sys.executable, "-c", STUBBORN]) if args.helper else None

    async def info(request: Request) -> Response:
        return JSONResponse(
            {
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "helper": helper and helper.pid,
                "env": {name: os.environ.get(name) for name in ENV},
            }
        )

    app = make_app(
        [Route("/info", info)],
        ready=lambda: not args.never_ready and not (args.sick_file and args.sick_file.exists()),
    )
    try:
        if args.ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            # uvicorn handles SIGTERM only on the main thread, so it stays ignored here.
            thread = threading.Thread(target=anyio.run, args=(_serve, app, args))
            thread.start()
            thread.join()
        else:
            anyio.run(_serve, app, args)
    finally:
        if args.finally_file is not None:
            args.finally_file.write_text("serve returned", encoding="utf-8")


async def _serve(app: Starlette, args: argparse.Namespace) -> None:
    if not args.two_planes:
        await serve(app, ready_file=args.ready_file, port=args.port)
        return
    async with anyio.create_task_group() as tg:
        primary = await tg.start(serve, app)
        second = await tg.start(serve, make_app())
        write_ready(args.ready_file, primary, {"toy.a": primary, "toy.b": second})


if __name__ == "__main__":
    main()
