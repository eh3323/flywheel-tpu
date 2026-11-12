#!/usr/bin/env python3
# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Relay a unix socket to the TPU VM's vbar control agent.

    python3 vbar_relay.py /run/flywheel-tpu-ci/vbar.sock

On a v6e VM, libtpu reaches Google's vbar_control_agent at [::]:8353 before
it opens the chip. The TPU test containers have no network, so they reach it
through this socket instead (VBAR_CONTROL_SERVICE_URL=unix:...), and through
nothing else. The agent's own unix socket lives in its container's mount
namespace, which a bind mount cannot reach. setup_tpu_runner.sh runs this as
a systemd service. Standard library only, for the VM's system Python.
"""

import asyncio
import os
import sys

AGENT_HOST = "127.0.0.1"
AGENT_PORT = 8353
CHUNK_BYTES = 1 << 16


async def pipe(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while data := await reader.read(CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def relay(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        agent_reader, agent_writer = await asyncio.open_connection(
            AGENT_HOST, AGENT_PORT
        )
    except OSError:
        writer.close()
        return
    await asyncio.gather(
        pipe(reader, agent_writer),
        pipe(agent_reader, writer),
        return_exceptions=True,
    )


async def serve(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)
    server = await asyncio.start_unix_server(relay, path)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve(sys.argv[1]))
