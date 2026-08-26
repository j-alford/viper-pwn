#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./tools/stop_mcmc.sh results/cta1_runNN"
    exit 1
fi
pid_file="$1/mcmc.pid"
if [ ! -f "$pid_file" ]; then
    echo "No mcmc.pid found in $1 -- is it running?"
    exit 1
fi
pid=$(cat "$pid_file")
echo "Stopping PID $pid..."
kill "$pid"
rm -f "$pid_file"
echo "Done."
