#!/usr/bin/python3
# A launchd job's definition NEW put in place as PLIST (default NEW without .new) and the job loaded in DOMAIN (default
# user/<uid>) where it is not loaded; a loaded job is left running, its new definition taking effect at its next load
# (README.md, launchd).   mesh-launchd-agent.sh NEW [DOMAIN [PLIST]]
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

status, said = meshlib.launchd(*sys.argv[1:4])
if said:
    print(said)
sys.exit(status)
