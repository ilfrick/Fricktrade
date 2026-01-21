#!/usr/bin/env bash
set -euo pipefail

START_TIME="15:30"
END_TIME="22:00"
TZ_LOCAL="Europe/Rome"

now_ts=$(TZ="$TZ_LOCAL" date +%s)
today=$(TZ="$TZ_LOCAL" date +%F)
start_ts=$(TZ="$TZ_LOCAL" date -d "$today $START_TIME" +%s)
end_ts=$(TZ="$TZ_LOCAL" date -d "$today $END_TIME" +%s)

if [ "$end_ts" -le "$start_ts" ]; then
  echo "End time must be after start time for $today" >&2
  exit 1
fi

sleep_until_start=$(( start_ts - now_ts ))
if [ "$sleep_until_start" -lt 0 ]; then
  sleep_until_start=0
fi

run_dir="/home/nicola/Fricktrade/data/monitoring/$today"
mkdir -p "$run_dir"

echo "Monitoring window: $today $START_TIME-$END_TIME ($TZ_LOCAL)" | tee "$run_dir/window.txt"
echo "Sleeping ${sleep_until_start}s until start." | tee -a "$run_dir/window.txt"

sleep "$sleep_until_start"

echo "Monitoring started at $(TZ="$TZ_LOCAL" date)" | tee -a "$run_dir/window.txt"

# Start decision trace tail
trace_src="/home/nicola/Fricktrade/data/reports/decision_trace/${today}.jsonl"
trace_out="$run_dir/decision_trace.jsonl"
( tail -F "$trace_src" >> "$trace_out" ) &
trace_pid=$!

# Start docker logs (retry if containers aren't up yet)
logs_out="$run_dir/docker_logs.txt"
(
  while true; do
    docker compose logs -f --tail=0 trader healthwatch api learner >> "$logs_out" 2>&1 || true
    sleep 5
  done
) &
logs_pid=$!

# Metrics sampling loop
metrics_out="$run_dir/metrics_samples.txt"
(
  while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "### $ts" >> "$metrics_out"
    curl -s http://localhost:8001/metrics >> "$metrics_out" || echo "metrics_unavailable" >> "$metrics_out"
    echo >> "$metrics_out"
    sleep 60
  done
) &
metrics_pid=$!

# Wait until end time
while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
  sleep 30
  true
  
  # Keep window in case of system time changes
  if [ $(TZ="$TZ_LOCAL" date +%s) -ge "$end_ts" ]; then
    break
  fi
  
  # If decision trace file isn't present yet, keep waiting (tail -F will handle it)
  if [ ! -f "$trace_src" ]; then
    echo "Waiting for decision trace file: $trace_src" >> "$run_dir/window.txt"
  fi
  
  # If logs process died, restart it
  if ! kill -0 "$logs_pid" 2>/dev/null; then
    (
      while true; do
        docker compose logs -f --tail=0 trader healthwatch api learner >> "$logs_out" 2>&1 || true
        sleep 5
      done
    ) &
    logs_pid=$!
  fi

done

# Stop background processes
kill "$trace_pid" "$logs_pid" "$metrics_pid" 2>/dev/null || true

echo "Monitoring ended at $(TZ="$TZ_LOCAL" date)" | tee -a "$run_dir/window.txt"
