# How the GPU backend searches five billion keys per second

mtvanity V1.2, source revision 6. Measured 2026-11-27 and 2026-11-28.

mtvanity looks for a Meshtastic key whose node ID or public key reads the way you want. Since revision 6 it can do the search on an NVIDIA GPU. This page shows how, what was measured, and what has not been checked.

| | |
|---|---|
| Keys per second, RTX 4080 Laptop GPU | **4.56 billion** averaged over two minutes at full power; **6.6 to 7.2 billion** for the first seconds from cool; **3.3 billion** with the GPU held to about 40 W |
| The same laptop's 32 CPU threads (AVX2) | 0.51 to 0.56 billion |
| Needed besides the program | **the NVIDIA driver, nothing else** |

The V1.2 release binaries have it. To build it yourself, add `-DMTV_CUDA` and keep `mtvanity_cuda.h` next to `mtvanity.c`. Without the option the program is the CPU program it was.

## No CUDA toolkit, at build time or at run time

- **The GPU kernel is text.** `mtvanity_cuda.h` contains a small emitter that writes the kernel in PTX, NVIDIA's assembly language, when the program starts. The driver compiles it. `./mtvanity --cuda-ptx` prints all 18,690 lines for review.
- **The driver is loaded at run time** (`libcuda.so.1` on Linux, `nvcuda.dll` from the Windows system directory). If it is missing, or there is no usable GPU, a line on stderr says why and the program searches on the CPU.
- **On Linux this means linking dynamically.** glibc's linker warns that a static program using `dlopen` needs the shared libraries of the same glibc version at run time. A static test program crashed when it tried (2026-11-27). So the V1.2 Linux binary is linked dynamically. It is built against Debian 11's glibc, so that it runs with glibc 2.29 or newer and needs nothing else. The Windows `.exe` stays static.
- **Conservative target:** PTX ISA 5.0 for `sm_50`. PTX ISA 5.0 came with CUDA 8.0, and NVIDIA documents CUDA 8.0 as needing driver 367.48 or newer on Linux and 369.30 or newer on Windows. `sm_50` covers GPUs of compute capability 5.0 and later: the Maxwell generation (GeForce GTX 750 Ti, GTX 900 series) and everything since. Driver 610.88 on an RTX 4080 Laptop GPU is the only combination that has been run.
- **The first start compiles the kernel:** 0.4 to 0.8 s on an idle machine, and 8.4 s in one run where the driver had to share the CPU with 32 search threads. The driver keeps the result in its own cache, and later starts skip it.

## What one key costs on a GPU

The walk is the CPU kernels' walk: windows of neighbouring keys around a centre point, both keys `M + iS` and `M - iS` from one shared inverse, and Montgomery's trick to share the inversion. On the GPU a window has 4,097 keys and every GPU thread runs its own walk, 29,696 of them on this card. Each key costs:

- **3 multiplications and 1 squaring** of 255-bit numbers (the CPU kernels need 2.5 and 1),
- **one CRC-32**, by four table look-ups per 32-bit word, skipped when no target reads the node ID,
- **the same filter as on the CPU**, target group by target group.

## Limbs

A GPU multiplies 32 x 32 bits and has cheap carry chains. So a field element is eight 32-bit limbs, kept below 2^256 and not reduced further until a key is published.

```
AVX2:    10 limbs of 25 or 26 bits      AVX-512 IFMA: 5 limbs of 51 bits
GPU:      8 limbs of 32 bits, with carries (mad.lo.cc / madc.hi.cc)
```

A multiplication is 164 PTX instructions and a squaring 123. A second form, ten limbs without carry chains as in the AVX2 kernel, was built and measured 5% slower with a quarter more registers.

## Memory decided the design

Montgomery's trick stores one running product per table point and reads them back in reverse. A CPU core keeps that in its own cache. A GPU has 29,696 walks and one shared cache.

| Prototype variant (2026-11-27, no CRC) | Keys per second |
|---|---:|
| Every product stored (goes to DRAM) | 4.5e9 |
| Same arithmetic, nothing stored (wrong keys, for comparison) | 6.7e9 |
| **Product stored every 16 points, each chunk recomputed into a small stack** | **5.5 to 5.7e9** |

So the kernel saves the product only every 16 table points. On the way back it recomputes each chunk of 16 into a stack of 16 entries per walk, 15 MB in all, which stays in the GPU's 48 MB L2 cache. That is the extra half multiplication per key, and it is cheaper than the memory traffic it replaces.

## How a key comes back

- **The GPU never sees a private scalar.** It gets the public centre points and the table of multiples.
- **Every walk has its own scalar from the operating system's random generator.** None is derived from another, so two printed keys are never a known offset apart. The 29,696 centres are computed on the host in one batch.
- **A key that passes the filter is written to a record:** which walk, which window, which position, the public key and the node ID.
- **The host forms the private key** from the walk's base and the position, and the unchanged reference code recomputes the public key and the CRC-32 before anything is printed. A walk that has given a key is retired.
- **Two launches are always queued** (whole windows, at most about 50 ms each), so the GPU does not wait for the host, and no launch comes near the two seconds after which Windows resets a GPU.

## Measured

RTX 4080 Laptop GPU (12 GB, driver 610.88), Intel i9-14900HX, Windows 11 with WSL2. Speeds are from `--benchmark`, which since this revision reports the speed after start-up. **The speed follows the power the laptop gives the GPU**, so every figure carries its power state.

| Measured | Build | GPU power state | `cuda` line |
|---|---|---|---:|
| 2026-11-27, nothing else running | Linux under WSL | Windows "Balanced", on mains, GPU power limit reported as 85 W; the GPU drew about 40 W, 66 to 73 °C | 3.22 billion keys/s over 10 s, 3.33 billion over the next 30 s (twice) |
| Same hour, back to back | Windows `.exe`, started from WSL | the same | 3.22 billion over 10 s, 3.35 billion over the next 30 s |
| 2026-11-27, earlier, nothing else running | Linux under WSL | limit reported as 175 W; 122 W drawn, 68 °C at the start, 86 °C after 10 s | 6.60 billion over the first 10 s, then 4.56 billion averaged over 120 s |
| By the human, native Windows from PowerShell, cool laptop (an earlier build of revision 6) | Windows `.exe` | not recorded | 7.20 billion (5 s line) |

- **WSL costs no GPU speed:** the Linux build under WSL and the Windows build ran within 1% of each other, back to back.
- **The 175 W state is not safe on this laptop for long.** The run after the 120 s line ended with the laptop overheating and switching off. The figures at about 40 W are from the state it came back in.
- **The CPU alone:** 557.8 million keys/s with AVX2 and 303.6 million scalar (native Windows, by the human, cool laptop); 512.9 million and 277.3 million under WSL in the build run.

The prototype figures above (4.5 to 6.7 billion) were taken in the design run with the GPU at its 80 W limit, a state in between.

## Starting: the CPU first, then the GPU

Under `auto` every CPU thread starts searching at once. The GPU is brought up beside them, checked, and takes over after its first launch. The CPU threads then stop, except in three cases:

- `--threads N` asked for them;
- the search is about to end;
- the GPU is not clearly faster than they are. Its first launch must reach 1.5 times the speed the CPU threads had, because a launch from cool ran up to 30% faster than the GPU settled at. A small GPU beside a big CPU then searches beside the CPU threads instead of making the search slower.

A search that ends first leaves without waiting for the driver.

Two more rules since 2026-11-28:

- **On Linux two CPUs are left free while the driver starts,** in a search expected to need 2^30 keys or more. At most one thread in eight waits, so a CPU with 8 to 15 threads leaves one free and a smaller one none; CPUs that the search leaves idle anyway count as free. With threads bound to CPUs it is the first two that stay free. The threads that waited start if the GPU does not take over. Shorter searches, and the Windows build, start every thread at once.
- **CPU threads that search on time their kernels again** when the GPU's start-up is over: beside the GPU, after a GPU that failed, or when no GPU could be used. The first timing runs beside the driver's start, and some threads then pick the scalar kernel.

If anything goes wrong with the GPU, the search goes on on the CPU:

- **No driver library:** the plan printed before the search says so, for example `no usable GPU: no NVIDIA driver library (libcuda.so.1) was found; searching on the CPU`.
- **No usable device, a kernel the driver refuses, or too little memory:** the same kind of line, with the CUDA error, appears once the driver has answered.
- **A GPU that fails its start-up check** is not used, with a warning.
- **`--backend cuda`** without a usable GPU says why and searches on the CPU instead of stopping.
- **A GPU that fails partway through a search,** for example in a driver reset, gives a warning that names the driver call and the CUDA error. The CPU threads that had stood down search again; under `--backend cuda` they start then. Keys already printed stand. A key the GPU had found but not yet handed over is lost, never printed wrong.

`--quiet` hides the lines about no driver and no usable device. The warnings and the `--backend cuda` line always show.

Time from starting the program to the first key (2026-11-27, WSL, laptop in the 40 W state, where everything was slower than usual):

| Target | CPU-only build | GPU build, `auto`, cache warm | GPU build, cache empty |
|---|---|---|---|
| 4-digit node ID | 0.25 to 0.26 s | 0.24 to 0.27 s | 0.25 to 0.27 s |
| 6-digit node ID | 0.25 to 0.41 s | 0.27 to 0.40 s | 0.37 to 0.47 s |
| `key:Mes` | 0.25 to 0.28 s | 0.24 to 0.27 s | 0.25 to 0.27 s |
| Full 8-digit node ID | 2.6, 13.9, 23.1 s | 2.8 to 4.0 s (six runs) | 0.9, 3.2, 3.9, 13.2 s |

`--backend cuda` (the GPU alone, nothing searches while it starts) took 1.2 to 6.3 s for the 8-digit ID in four runs.

What the GPU's start-up consists of, timed beside 32 searching CPU threads (`MTV_CUDA_TIMING=1` prints it), three runs:

| Phase | Seconds |
|---|---|
| Driver (`cuInit`) | 0.99 to 1.17 |
| Context | 0.55 to 0.67 |
| Kernel from the driver's cache (8.4 s once with the cache empty) | 0.03 to 0.05 |
| Buffers | 0.02 to 0.04 |
| Tables | 0.07 to 0.11 |
| Seeding 29,696 walks on all CPU threads | 0.11 to 0.13 |
| Start-up check | 0.40 to 0.61 |
| **First launch done, into the search** | **2.3 to 2.4** |

- **Most of it is the driver's own start**, which was 0.15 to 0.8 s on an idle machine in the build run. Seeding, once the longest part that is ours, is a tenth of a second on all threads.

**Free CPUs for the driver (2026-11-28).** Time until the GPU is ready under `auto`, old and new builds alternating, a rest between runs, driver cache warm unless noted:

| Build | CPUs left free | Runs | GPU ready after | Median |
|---|---|---|---|---|
| Linux under WSL2 | none (as before) | 13 | 1.39 to 3.40 s | 1.71 s |
| Linux under WSL2 | two | 13 | 1.13 to 3.33 s | 1.34 s |
| Linux under WSL2 | one | 5 | 1.33 to 1.83 s | 1.43 s |
| Linux under WSL2, driver cache empty | none | 3 | 9.23 to 10.61 s | 9.98 s |
| Linux under WSL2, driver cache empty | two | 3 | 6.30 to 6.35 s | 6.31 s |
| Windows `.exe` | none | 5 | 1.09 to 1.57 s | 1.12 s |
| Windows `.exe` | two | 10 | 1.00 to 1.70 s | 1.15 s |

- **Under WSL2 the GPU is ready about 0.4 s sooner** with two CPUs free. What gets shorter is everything after the driver's own start: context, kernel, tables, seeding and check together took 0.64 to 1.58 s (median 0.99 s, nine runs) with all threads searching, and 0.48 to 0.83 s (median 0.52 s) with two CPUs free. `cuInit` itself took 0.3 to 2.5 s either way.
- **With an empty driver cache** the kernel compiled in 3.7 to 3.8 s instead of 7.5 to 8.1 s. On an idle machine it takes 0.4 to 0.8 s, so more free CPUs would help that one start further. Not tried.
- **On Windows free CPUs gained nothing,** so the `.exe` keeps none. Its search threads already give way between windows.
- **Short searches are untouched:** they start every thread at once. Time to the first key, six runs each, CPU-only build / old GPU build / new GPU build: 4-digit ID 0.07 to 0.26 / 0.07 to 0.27 / 0.08 to 0.26 s; 6-digit ID 0.08 to 0.47 / 0.12 to 0.49 / 0.10 to 0.32 s; `key:Mes` 0.07 to 0.27 / 0.07 to 0.28 / 0.08 to 0.27 s.
- **Full 8-digit ID, five runs each:** 1.51 to 3.78 s with the old build, 1.30 to 4.18 s with the new one. Luck decides most of that. The time to try 2^32 keys, which luck does not touch, was 2.78 to 3.99 s old and 2.61 to 4.52 s new (four runs each), following the driver's start.
- **The kernel choice (2026-11-28).** In most runs 1 to 5 of the CPU threads picked the scalar kernel while the driver started, with or without free CPUs. After a GPU made to count as lost (`--cuda-fail-at 1`) the summary now reads `cuda x1, avx2 x32` under `auto` and under `--backend cuda`; before, the human had seen `avx2 x29, scalar x3`.
- **Leaving after the last key (2026-11-28):** 0.02 to 0.07 s for the CPU-only build; 0.06 to 0.15 s after a key found on the GPU; 0.01 to 0.8 s, and 1.6 s once, when a short search under `auto` ends while the driver is still starting. That last wait is not explained and not fixed.

## Speed that is still checked

- **Every hit is recomputed** by the slow reference code before it is printed, exactly as for the CPU kernels. A fault on the GPU can lose a key but cannot print a wrong one.
- **Every search starts with a check** of the GPU kernel: its field arithmetic against the scalar code, keys of real windows against the reference ladder, planted keys.
- **11 more self-test checks** (57 on this machine, 46 in a build without the option): tables, seeding and kernel text, which need no GPU; 4,096 multiplications, squarings, inversions, canonical forms and CRCs against the scalar code, including the extreme values; 8,306 keys of eight walks against the reference ladder and CRC-32, with every key of the first window of two walks; keys planted at window edges, at the centre and where chunks meet, for every kind of filter; a centre forced onto a table point; a filter that passes too much; and real searches for 10 and 4 targets at once.
- **The kernel was broken on purpose in 38 places,** one at a time: a product in the wrong column, a dropped carry, the wrong reduction constant, a missing doubling, a square on the wrong diagonal, a final addition or fold left out, the exact canonical step skipped, an inversion one squaring short, a table value read from the wrong place, the two keys of a step swapped, a checkpoint or stack slot holding the wrong value, a centre moved with the wrong sign or not written back, a zero product not reported, a CRC shifted wrongly, reused or skipped, a filter mask or bitmap bit ignored, a key reported under its neighbour's walk, position or window, and seven of these again in the search kernel only. The self-test caught all 38, and all 38 again when the campaign was repeated on the final kernel.
- **Checkers** (gcc 14, 2026-11-27, Linux under WSL):
  - **ASan with UBSan** reports nothing for the CPU build (self-test and a search) or for the GPU build without a driver. With the GPU the self-test passes, and the only report is memory the NVIDIA driver library keeps at exit.
  - **TSan** reports nothing for the CPU build (full self-test and a search), for the GPU build's host code, or for searches that start on CPU threads and hand over to the GPU. On this kernel it only runs with address randomisation off (`setarch x86_64 -R`).
  - **valgrind** reports nothing for a search with the CPU build or with the GPU build without a driver. With the GPU the search works, but valgrind reports uninitialised values: most inside the driver's libraries, and 48 attributed to the routine that seeds the walks.
  - **The cause of the valgrind reports is confirmed** (2026-11-28, `valgrind --track-origins=yes`, a search with `--backend cuda` on the real driver). Of 841 reports about uninitialised values, 835 say where the value was created, and every one is memory allocated inside the driver's libraries: 780 in `libnvdxgdmal.so.1` called from `libcuda.so.1.1`, 52 in `libdxcore.so`, 3 in `libcuda.so.1.1`. The other six are raised inside `libcuda.so.1.1`. None starts in mtvanity's code. The reports in mtvanity's own routines are values the driver handed back: the GPU's size, from which the number of walks is worked out, and the result buffers.
- **Independent re-derivation:** 3,367 keys found on the GPU (2026-11-27, Linux build and Windows `.exe`, every kind of target, 64 random full 8-digit node IDs among them) were recomputed by a separate checker in Python with only the standard library. It checked the public key by RFC 7748, the CRC-32 node ID and the pattern. No private key repeats or lies a small offset from another. All passed.
- **The V1.2 changes leave the kernel alone:** `--cuda-ptx` prints the same PTX, byte for byte, as revision 6 did before them.
- **Falling back to the CPU** (V1.2, 2026-11-28) was tested on a machine without any NVIDIA driver, and against a stand-in driver library that fails on purpose at each step. Every run found its keys on the CPU, and TSan, ASan and UBSan report nothing for these runs:
  - no device
  - a refused kernel
  - a failed start-up check
  - a wait or a launch that fails partway through a search, under `auto` and under `--backend cuda`
  - a GPU too slow to be worth stopping the CPU threads for
- **On the test laptop** (V1.2, 2026-11-28, by the human, before the fixes of that day): the Linux release binary needs `GLIBC_2.29` and libm, libdl, libpthread and libc only, and passes 57 of 57 checks on the GPU and 47 of 47 in a Debian 11 root filesystem without a driver, where a search said why and found its key on the CPU. The Windows `.exe` passes 57 of 57 and imports `bcrypt.dll`, `KERNEL32.dll` and `msvcrt.dll` only. 140 keys found with `--backend cuda` by the two binaries passed the separate checker.
- **A real GPU made to count as lost** (`--cuda-fail-at 1`, target `c0ffee42`, by the human): under `auto` the GPU was ready after 2.64 s, the CPU threads stopped, the GPU was lost, and the CPU found the key in 7.20 s; under `--backend cuda` the CPU threads started at the loss and found the key in 8.10 s.
- **After the fixes of 2026-11-28:** the same PTX (SHA-256 `aa78401b...5e7d74`); 57 checks on the GPU with the Debian 11 build and the `.exe`, 46 without the option, 47 with the driver hidden; a build without the option is byte for byte what it was; TSan reports nothing for a hand-over, a lost GPU under `auto` and under `--backend cuda`, a stand-in driver without a GPU, and a search that ends during the driver's start; 46 more keys passed the separate checker.
- **Tested again after the fixes** (2026-11-28, by Claude Opus 5.5, on 2 cloud Xeon cores with AVX-512 and no NVIDIA driver): the fix run's Debian 11 build passes 58 of 58 checks with the AVX-512 kernel, and 65 keys it found on the CPU passed the separate checker. Against the stand-in driver, now also one that takes 0.5 to 3 s to start, every search ended with all its CPU threads on the fastest kernel, except one with 16 threads on the 2 CPUs, where 3 of the 16 chose AVX2 when they timed their kernels again. Threads that waited started with `--threads 8` and `16`. TSan, ASan and UBSan report nothing for these runs.
- **The self-test caught an error of its own design** while it was written: with several targets of one shape, the first version reported far more keys than its record buffer holds and lost the real one. The GPU now compares such keys exactly.

## Not yet checked

- **One GPU only:** an RTX 4080 Laptop GPU, compute capability 8.9, driver 610.88. No other card, no desktop, no older driver, no machine with two GPUs (the first one is used).
- **Power states:** about 40 W, 80 W (prototype only) and a short stretch at 122 W. Nothing sustained above 80 W: the laptop overheated when that was tried.
- **The machine code the driver makes from the PTX (SASS) has not been read.** That needs NVIDIA's toolkit, which was not installed.
- **No run longer than two minutes.**
- **Tuning of the finished kernel** was cut short. Blocks per multiprocessor (1, 2, 3), a checkpoint every 8 points instead of 16, launches of 25, 50 and 75 ms and a filter kept in registers were tried in the throttled 40 W state, and none beat the noise. Window size, threads per block, register limits and the rest were not tried.
- **No real GPU has failed by itself in the middle of a search.** A stand-in driver library, and a real GPU made to count as lost after N launches (`--cuda-fail-at N` in a build with `-DMTV_CUDA_FAULTS`), were used instead.
- **Leaving two CPUs free for the driver was measured under WSL2 only,** on one 32-thread CPU. Linux without Windows underneath, and smaller CPUs, were not measured.
- **CPU threads can still pick the scalar kernel while the driver starts:** 1 to 5 of 30 did in most runs (2026-11-28). They stop when the GPU takes over, and they time their kernels again if they search on. The first timing itself was not made robust against a busy CPU.
- **A short search under `auto` can take long to exit after its key:** 0.01 to 0.8 s, and 1.6 s once, where the CPU-only build takes 0.02 to 0.07 s (2026-11-28, 4-digit, 6-digit and `key:Mes` targets, six runs each). The search ends while the driver is still starting. What the exit waits for was not established (`strace` is not installed on the test machine); the driver call still in flight is the suspect. Not fixed. After a key found on the GPU the exit took 0.06 to 0.15 s, under `auto` and under `--backend cuda`.
- **The rule that keeps the CPU threads beside a slow GPU** has been tested with a stand-in driver only.
- **Whether the Windows desktop stays smooth during a search** was not watched.
- **No key found on the GPU has been written to a radio.**
- **Older GPUs** (before compute capability 7.0) multiply differently and will be slower by an unknown amount.
