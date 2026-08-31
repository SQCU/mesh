#!/bin/sh
log=$1
shift
exec "$@" >"$log" 2>&1 </dev/null
