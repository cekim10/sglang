# ft-detect: failure-detection latency kill test (SGLang, TP=4)

Measures, without fixing anything, how long it takes a stock SGLang replica to
notice that a TP rank has died (case A, `kill -9`) or hung (case B, `SIGSTOP`),
and the smallest watchdog timeout that never fires spuriously under load.

Dependencies beyond the SGLang environment: `httpx`, `psutil`, `pandas`.
No SGLang source is modified; `launch_wrapped.py` monkey-patches from outside.

## Files

| file | role |
|---|---|
| `common.py` | one monotonic clock (`time.monotonic_ns`) and JSONL helpers shared by every process |
| `launch_wrapped.py` | stock `launch_server()` plus per-rank logs/PIDs and per-step timestamps (see docstring) |
| `launch.sh` / `stop.py` | start a replica and wait for readiness / SIGCONT + kill the recorded tree |
| `recon.py` | version, commit, timeout defaults with file:line from the *installed* code; `--gpus`; `--tree` |
| `load.py` | open-loop asyncio+httpx generator, `mixed` and `harsh` profiles, `--closed-loop` for calibration |
| `probe.py` | 100 ms sidecar: /health, /health_generate, 1-token /generate, PID states, log-pattern tail |
| `inject.py` | records `t_inject` then SIGKILL / SIGSTOP one rank; `--resume` sends SIGCONT |
| `steady.py` | steady-state gate on the last two 60 s windows of TTFT |
| `analyze.py` | joins the logs of one or more runs -> CSV row + markdown |
| `run_case.sh` | Step 2 driver (N runs of a case, fresh server each) |
| `sweep_floor.sh` | Step 3 driver (timeout sweep without injection) |
| `calibrate.sh` | saturation throughput -> 70% rate |
| `report.py` | Step 4: `REPORT.md` |

Every run lives in its own directory (`runs/<name>/`): `logs/server.log`,
`logs/rank<N>.log`, `logs/steps_rank<N>.jsonl`, `pids.json`, `requests.jsonl`,
`probe.jsonl`, `inject.jsonl`, `recon.json`, `summary.{json,md}`.

## Order of operations on the GPU host

```bash
cd ft-detect
python recon.py --gpus                     # which GPUs are free; pick 4 that nobody else uses
export CUDA_VISIBLE_DEVICES=0,1,2,3        # ONLY your allocation
export MODEL=Qwen/Qwen3-14B                # or meta-llama/Llama-3.1-8B; must be in $HF_HOME already
export TP=4 FT_PORT=30000

python recon.py                            # Step 0 on the installed version -> run/recon.json (compare with the Step 0 report)
./calibrate.sh mixed                       # prints "70% -> --rate X"
export RATE=X

./run_case.sh A 1 5                        # Step 2, case A, kill -9 rank 1, 5 runs   -> results/caseA.{csv,md}
./run_case.sh B 1 5                        # Step 2, case B, SIGSTOP rank 1, 5 runs   -> results/caseB.{csv,md}

./sweep_floor.sh mixed                     # Step 3, 20 min x {300,60,30,10,5,2}, watchdog+dist together
./sweep_floor.sh harsh                     # same with 30% 32k prefill + 3x bursts every 5 min
SWEEP=watchdog ./sweep_floor.sh mixed 10 5 2   # only if 'together' differs from what you expect: sweep the knobs separately
SWEEP=dist     ./sweep_floor.sh mixed 10 5 2

python report.py                           # -> REPORT.md ; add free-text observations to results/notes.md first
```

Manual single run, if you want to watch it:

```bash
export FT_RUN_DIR=$PWD/runs/manual1
./launch.sh
python probe.py &                          # run/probe.jsonl
python load.py --rate $RATE --duration 1800 --profile mixed &
python steady.py --wait 600
python inject.py --case B --rank 1
# ... wait for the watchdog (300-450 s at defaults) or whatever fires first ...
python stop.py                             # SIGCONTs the stopped rank, kills the tree
python analyze.py $FT_RUN_DIR
```

## Definitions used by analyze.py

- `t_inject`: monotonic timestamp taken immediately before the `kill()` syscall.
- `t_first_symptom`: last streamed chunk received by any request that was in flight
  at `t_inject` and never completed (client-visible moment output stopped).
- `t_stall_engine`: last `run_batch` timestamp on a surviving rank (rank 0 preferred)
  before a gap of >= 5 s, from `steps_rank<N>.jsonl`.
- `t_detect_engine`: earliest of a detector string in any log (watchdog fire, NCCL
  timeout/error, scheduler exception, subprocess crash, SIGQUIT, abort) or the death
  of any process other than the injected rank.
- `t_detect_health`: first non-200 (503 / timeout / connection error) from `/health`
  or `/health_generate` after `t_inject`. `t_detect_gen1` is the same for the 1-token
  `/generate` probe (5 s client timeout), also reported as first 3-in-a-row.
- `n_inflight_hung`: requests sent before `t_inject` that ended in an error.
- `n_new_errored`: requests sent between `t_inject` and `t_detect_engine` that errored.
- `spurious_kill` (floor runs): any engine detector string or process death during a
  run with no injection; a failed launch at that timeout also counts.

## Things to know before trusting a number

- The scheduler watchdog polls every `timeout/2`, so at 300 s it fires between 300
  and 450 s after the last step, then sleeps 5 s before SIGQUIT.
- `/health` returns 200 on *any* engine output, and 503 only after 20 s of silence;
  under load with a hung engine expect ~20 s. `/ready` never fails during a hang.
- Case A is caught by `SubprocessWatchdog` (1 s poll) — expect ~1-2 s, by design.
- The load generator never times out requests on its own (`--req-timeout` unset), so
  hung requests are observed hanging until the tree is killed.
- `--dist-timeout` below a few seconds may make startup itself fail; the sweep records
  that as `launch_failed` and moves on.
- Prompts are random token ids over `/generate` (`input_ids`), not text, so the
  model's output is garbage. Latency is what is measured, not quality.
