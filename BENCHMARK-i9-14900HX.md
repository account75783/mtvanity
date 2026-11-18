# mtvanity V1 on the Intel Core i9-14900HX

Authored by Claude Fable 5.2 (Max, with Research), assisted by a human.
Recorded 2026-11-17 (Linux build under WSL, release candidate) and 2026-11-18 (native Windows build) on the human's laptop.

## Result

**516.4 million keys per second on 32 threads.** An exact 8-digit node ID (`!66666666`) was found in 6.85 seconds.

## Machine

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900HX: 8 P-cores with hyper-threading plus 16 E-cores, 32 threads |
| Vector support | AVX2 (no AVX-512) |
| Environment | Linux binary under WSL on Windows 11 (inferred from the `/mnt/c/...` path) |
| Threads used | 32, pinned one per logical CPU |
| Kernel chosen | AVX2 on all 32 threads, P-cores and E-cores alike |

Compiler version and power settings (plugged in, performance mode) were not recorded.

## Runs

| Command | Keys tried | Time | Rate | Outcome |
|---|---|---|---|---|
| `./mtvanity c0de` | 4.26e+05 | 0.03 s | not meaningful | `!c0de52f0` |
| `./mtvanity 66666666` | 3.54e+09 | 6.85 s | 516.4 M/s | `!66666666` |

The 4-digit run ends before the threads finish starting up, so its printed rate (16.1 M/s) measures start-up, not search speed.

The 8-digit run was slightly lucky: it needed 3.54 billion keys against an average of 4.29 billion.

## What to expect at this rate

| Digits fixed | Average keys needed | Average time |
|---|---|---|
| 4 | 65,536 | under 1 ms, plus start-up |
| 5 | 1.05 million | about 2 ms |
| 6 | 16.8 million | about 0.03 s |
| 7 | 268 million | about 0.5 s |
| 8 (full ID) | 4.29 billion | about 8.3 s |

Times are averages. A single search can finish much sooner or take several times longer.

## Compared with the development sandbox

| Machine | Threads | Rate | Per thread |
|---|---|---|---|
| Cloud sandbox (Xeon Sapphire Rapids, about 3 GHz) | 2 | 60.2 M/s | 30.1 M/s |
| i9-14900HX | 32 | 516.4 M/s | 16.1 M/s average |

The per-thread average is lower on the laptop because the 32 threads are a mix: hyper-threads share a P-core, and E-cores are slower than P-cores. Per-core-type rates were not measured separately.

The pre-run estimate for a full ID on this CPU was 5 to 8 seconds. The measured average of about 8.3 seconds lands just past the slow end of that range.

## Native Windows build (2026-11-18)

The static Windows `.exe` (MinGW-w64 gcc) passed the full self-test on the same laptop, then ran `--benchmark` at 5 seconds per line on 32 threads:

| Kernel | Keys per second | Average time for a full 8-digit ID |
|---|---|---|
| AVX2 | 495.9 M | about 8.7 s |
| Scalar | 275.7 M | about 15.6 s |
| Auto (chose AVX2 on all 32 threads) | 476.0 M | about 9.0 s |

Native Windows is within a few percent of the Linux build under WSL. The gap between the AVX2 and Auto lines is the same kernel measured twice, so it shows the run-to-run variation on this laptop (heat, background load) rather than a real difference.

## Caveats

- The WSL figure comes from one search run, not a `--benchmark` run. A sustained run on a laptop may be slower once it heats up.
- The first table is from the Linux build under WSL. Native Linux (without Windows underneath) was not measured.
- Private keys from these runs are deliberately left out of this file.
