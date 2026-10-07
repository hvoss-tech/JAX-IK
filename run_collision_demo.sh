#!/usr/bin/env bash
# Panda arm with realtime collision avoidance (self-collision, floor, spheres).
# Keys in the window: m cycles collision mode (off / soft / hard), c toggles the proxies,
# g toggles the floor, 1 and 2 toggle the spheres.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python examples/panda_collision_demo.py "$@"
