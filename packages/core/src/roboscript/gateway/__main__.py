"""Serve the model gateway: `python -m roboscript.gateway [--config C] [--ready-file PATH]`.

`--config` is a YAML path (ending in .yaml or .yml) or a name under configs/ (default `gateway`).
Upstream URLs and keys come from `.env.api` (or the file `RS_ENV_API_FILE` names), the client
token from `RS_GATEWAY_TOKEN` in `.env`. Changes to either file take effect on restart.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import anyio

from roboscript.gateway.app import Gateway
from roboscript.gateway.proxy import upstream_client
from roboscript.rpc import serve
from roboscript.settings import config_hash

log = logging.getLogger("roboscript.gateway")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m roboscript.gateway", description=__doc__)
    parser.add_argument("--config", default="gateway", help="YAML path or name under configs/")
    parser.add_argument("--ready-file", type=Path, help="written with the URL once listening")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines name upstream URLs
    anyio.run(_serve, args.config, args.ready_file)


async def _serve(source: str, ready_file: Path | None) -> None:
    async with upstream_client() as http:
        gateway = Gateway.from_files(source, http=http)
        config = gateway.config
        host, port = config.address
        log.info(
            "%d endpoints, models %s, config %s",
            len(config.endpoints),
            sorted(config.models),
            config_hash(config)[:12],
        )
        try:
            await serve(gateway.app, host=host, port=port, ready_file=ready_file)
        finally:
            await gateway.aclose()


if __name__ == "__main__":
    main()
