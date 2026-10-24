#!/bin/sh
exec /usr/bin/python3 "$(dirname "$(readlink "$0" || echo "$0")")/mesh-launchd-agent.py" "$@"
