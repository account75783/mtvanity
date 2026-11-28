# mtvanity

Vanity node ID and public key generator for Meshtastic 2.8+ radios (Heltec V4 and others), in a single C file.

## ⚠️ CHECK THAT THE ID IS OPEN FIRST

**Do not use another person's node ID.** Two nodes with the same ID disrupt traffic on the mesh: messages, routing and other people's node lists can all get confused, for them and for you.

Before you write a key to your radio, check that nobody on your mesh already has that ID (replace `c0ffee42` with yours):

```
meshtastic --nodes | grep -i c0ffee42 || echo "not seen by this radio"
```

Your radio only knows the nodes it has heard, so also check your local mesh map or ask in your mesh's group chat.

Take someone else's ID on purpose and expect to get T-hunted: the people whose mesh you broke will track down your transmitter. You will probably get blocked by the mesh too, with other operators ignoring your node so that nothing you send gets through.

## Status

**As of 2026-11-26, mtvanity is the fastest Meshtastic vanity key generator we know of:** 6.66 billion keys per second on a 32-core AMD Zen 5 machine, and 516 to 558 million on a laptop (Intel i9-14900HX, AVX2). The only other tool for Meshtastic keys we found with a published speed reports under 1 million keys per second on a desktop CPU. That comparison comes from a web search on 2026-11-25, not from a survey, so tell us if you know a faster one.

**Release:** V1.2 (2026-11-28). Prebuilt binaries for Linux and Windows are on the [Releases page](https://github.com/account75783/mtvanity/releases); see [Verifying a release](#verifying-a-release) before running one.

**New in V1.2: search on an NVIDIA GPU.** Both prebuilt binaries have the CUDA backend from source revision 6. A search starts on the CPU at once and moves to the GPU when it is ready, if the GPU is clearly the faster. On an RTX 4080 Laptop GPU it did 4.56 billion keys per second averaged over two minutes at full power, 6.6 to 7.2 billion for the first seconds from cool, and 3.3 billion with the GPU held to about 40 W (2026-11-27). That laptop's CPU does 0.5 to 0.56 billion. It needs the NVIDIA driver and nothing else, and it has run on that one GPU only. Without a usable GPU the program says why and searches on the CPU. See [Searching on a GPU](#searching-on-a-gpu) and [CUDA.md](CUDA.md).

**New in V1.2: the AVX-512 kernel in the prebuilt binaries.** Source revision 5 added it. On a CPU with AVX-512 IFMA (Intel since Ice Lake, AMD since Zen 4) the search has a third kernel to choose from. It has been measured on two machines: about 4.1 times as fast as the AVX2 kernel on AMD Zen 5 (6.66 billion keys per second on 32 cores) and about 3.5 times on an Intel Xeon. See [Performance](#performance).

**New in V1.1: vanity public keys.** You can now choose how the public key starts or ends as well as the node ID, ask for both in one key, and search for several keys in one run.

**Tested on a real radio on 2026-11-17, with firmware 2.8** (the latest stable release that day; 2.8.1 was in alpha). A generated key was written to the radio and the node ID changed to the chosen value. The test was done with V1; V1.1 and V1.2 make the same kind of key.

## Authorship

**Authored by Claude Fable 5.2 (Max, with Research), assisted by a human.**

- Design, code, tests and this document were written by Claude Fable 5.2 (model ID `claude-fable-5-2`), Anthropic's AI model.
- 4+ hours of research and engineering so far.
- A human set the goal, chose the target hardware, corrected the firmware assumptions and directed the work.
- The CUDA backend (`mtvanity_cuda.h`, source revision 6) was designed, written, measured and tested by Claude Fable 5.2 (`claude-fable-5-2`) on the human's laptop on 2026-11-27, to rules and decisions set by the human.
- The changes for the V1.2 release were made by Claude Opus 5.5 (`claude-opus-5-5`) on 2026-11-28: the move to the CPU when there is no usable GPU or it fails during a search, the rule that the CPU threads stop only for a GPU that is clearly faster, the Linux build for older systems, and these documents.
- The V1.2 fixes of 2026-11-28 were made by Claude Fable 5.2 (`claude-fable-5-2`) on the human's laptop, to a list of problems the human had measured: CPU threads beside a GPU time their kernels again when the GPU's start-up is over, two CPUs stay free on Linux while the driver starts, and the valgrind reports were traced to the driver's libraries.
- Claude Opus 5.5 (`claude-opus-5-5`) reviewed those fixes and tested them again on a cloud machine the same day.

## What it does

From firmware 2.8, a Meshtastic node ID is the CRC-32 of the node's 32-byte X25519 public key, shown as `!xxxxxxxx`. `mtvanity` searches for a private key whose public key hashes to the ID you want.

**New in V1.1:** it can also choose the public key itself, the base64 text the Meshtastic apps show for a node. `./mtvanity key:Mesh` finds a key whose public key starts with `Mesh`, `./mtvanity id:c0de+key:Me` finds one key with both a chosen node ID and a chosen public key, and one run can search for several keys at once. The patterns are listed under [Use](#use).

## Requirements

- **64-bit only.** An x86_64 or 64-bit ARM CPU with a 64-bit operating system. The arithmetic uses 128-bit integers, so a 32-bit build is not possible.
- **Minimum:** any 64-bit x86 CPU. AVX-512, AVX2 and carry-less multiply are used when the CPU has them and are not required.
- **Recommended:** 4 cores of Intel 4th-gen Core (Haswell, 2013) or newer, or AMD Ryzen 1st gen (2017) or newer. That is predicted to find a full 8-digit ID in about a minute or less.
- **Fastest:** a CPU with AVX-512 IFMA. That is AMD Zen 4 and Zen 5 (Ryzen 7000 and 9000, EPYC 9004 and 9005) and Intel Ice Lake, Tiger Lake, Rocket Lake and the Xeons since Ice Lake. Intel's 12th- to 14th-generation Core CPUs have no AVX-512. `./mtvanity --benchmark` says whether yours can run the kernel.
- **Windows uses at most 64 logical CPUs.** A machine with more than 64 will only cost you money on Windows: the extra cores sit idle. Linux uses up to 1024.
- **Systems:** Debian 12/13, Windows 11, Android (Termux).
- **The prebuilt Linux binary needs glibc 2.29 or newer**, which covers Debian 11, Ubuntu 20.04, RHEL 9, Amazon Linux 2023 and everything newer. It is linked dynamically so that it can load the NVIDIA driver, and needs nothing but glibc. On an older system, or one without glibc such as Alpine, build from source. The Windows `.exe` needs nothing.
- **GPU (optional):** an NVIDIA GPU with its driver, on Linux (also WSL2) or Windows. No CUDA toolkit is needed, to build or to run. The GPU kernel is PTX ISA 5.0 for `sm_50`. PTX ISA 5.0 came with CUDA 8.0, and NVIDIA documents CUDA 8.0 as needing driver 367.48 or newer on Linux and 369.30 or newer on Windows. `sm_50` covers GPUs of compute capability 5.0 and later: the Maxwell generation (GeForce GTX 750 Ti, GTX 900 series) and everything since. **Tried on one GPU only:** an RTX 4080 Laptop GPU with driver 610.88. Without a driver or a usable GPU the program says why and searches on the CPU.

### How long a full 8-digit ID takes

A full ID needs 4.29 billion keys on average. Rule of thumb: about 90 CPU cycles per key with AVX2 and about 185 without (with AVX-512 about 20 on AMD Zen 5 and about 40 on an Intel Xeon, the two CPUs measured), so

```
seconds = 4.29e9 x cycles per key / (cores x clock speed in Hz)
```

Measured:

| CPU | Keys per second | Average time |
|---|---|---|
| Intel Core i9-14900HX (laptop, 32 threads), Linux build under WSL | 516 M | about 8 s |
| Same laptop, native Windows build | 496 M | about 9 s |
| Same laptop, native Windows build, revision 6 (by the human, cool laptop) | 558 M | about 7.7 s |
| 2 cloud Xeon cores (Sapphire Rapids) | 60 M | about 71 s |
| 2 cloud Xeon cores (Emerald Rapids), with AVX-512 | 171 M | about 25 s |
| AWS EC2 c8a, 32 cores (AMD EPYC, Zen 5), with AVX-512 | 6,663 M | about 0.65 s |

Predicted from core counts and multiply speeds, **not measured**; treat each as good to within a factor of two:

| CPU | Keys per second | Average time |
|---|---|---|
| 4-core Skylake desktop (i5-6500 class, 2015) | about 125 M | about 35 s |
| 4-core Haswell desktop (i5-4590 class, 2013) | about 105 M | about 40 s |
| Ryzen 5 1400 (4 cores, 2017) | about 65 M | about 65 s |
| Intel N100 mini PC (4 cores, 2023) | about 50 M | about 1.5 min |
| 2-core Haswell laptop (i5-4200U class) | about 37 M | about 2 min |

### CPUs that will feel slow

No 64-bit CPU is hopeless, but these are predicted to be slow enough to notice (same caveat: not measured):

| CPU | Keys per second | Average time for a full ID |
|---|---|---|
| Raspberry Pi 4 (64-bit OS) | about 18 M | about 4 min |
| Core 2 Duo (2006 to 2009) | about 14 M | about 5 min |
| Celeron N2840 and other 2-core Bay Trail Atoms | about 7 M | about 10 min |
| Raspberry Pi 3 or Zero 2 W (64-bit OS) | about 5 M | about 15 min |
| Single-core Athlon 64 or 64-bit Pentium 4 | 3 to 5 M | 15 to 25 min |
| First-generation Atom netbook or nettop (D510, N450) | about 2 M | about 35 min |

- **Averages hide bad luck.** One search in twenty takes more than three times the average, so the last row can run for two hours.
- **Shorter IDs are far easier.** Each digit you drop is 16 times faster; a 7-digit prefix takes about 2 minutes even on the slowest row.
- **Cannot run at all:** 32-bit CPUs and 32-bit operating systems. That covers the Pentium M, Core Duo, Atom N270, Raspberry Pi Zero and Pi 1, and 32-bit Raspberry Pi OS on any Pi.

### How long a vanity public key takes

Every character you fix at the start of a public key multiplies the work by 64. At 500 million keys per second, roughly the i9-14900HX figure above:

| Characters fixed at the start | Average keys needed | Average time |
|---|---|---|
| 3 | 262 thousand | instant |
| 4 | 16.8 million | 0.03 s |
| 5 | 1.07 billion | about 2 s |
| 6 | 68.7 billion | about 2.3 min |
| 7 | 4.40 trillion | about 2.4 hours |
| 8 | 281 trillion | about 6.5 days |

- **`-i` halves the work for every letter**, because either case is accepted.
- **A node ID and a public key multiply.** Four ID digits with three key characters is 17 billion keys, about 35 s. A full 8-digit ID with just two key characters is 17.6 trillion keys, about 10 hours.
- **`mtvanity` prints the work for each target before it starts**, and warns once the estimate passes a day.

## Build

Each block is one line: install the tools, clone, build and run the self-test.

Debian 12/13 (also WSL):

```
sudo apt install -y git build-essential && git clone https://github.com/account75783/mtvanity.git && cd mtvanity && gcc -O3 -pthread mtvanity.c -o mtvanity -lm && ./mtvanity --self-test
```

Windows 11 (MSYS2 UCRT64 shell):

```
pacman -S --needed --noconfirm git mingw-w64-ucrt-x86_64-gcc && git clone https://github.com/account75783/mtvanity.git && cd mtvanity && gcc -O3 -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity.exe -lbcrypt && ./mtvanity.exe --self-test
```

Android (Termux):

```
pkg install -y git clang && git clone https://github.com/account75783/mtvanity.git && cd mtvanity && clang -O3 -march=armv8-a+crc -pthread mtvanity.c -o mtvanity -lm && ./mtvanity --self-test
```

Already have the source? Skip the clone and run only the `gcc` or `clang` part. To measure speed, run `./mtvanity --benchmark`.

### With an NVIDIA GPU

Add `-DMTV_CUDA`, and keep `mtvanity_cuda.h` next to `mtvanity.c`. Nothing else is installed: the program loads the NVIDIA driver when it starts, and says why and searches on the CPU if there is none.

Debian 12/13 (also WSL):

```
gcc -O3 -pthread -DMTV_CUDA mtvanity.c -o mtvanity -lm -ldl && ./mtvanity --self-test
```

Windows x86_64, cross-compiled on Debian or WSL:

```
x86_64-w64-mingw32-gcc -O3 -s -static -Wa,-muse-unaligned-vector-move -DMTV_CUDA mtvanity.c -o mtvanity.exe -lbcrypt
```

- **Do not add `-static` on Linux with `-DMTV_CUDA`.** A static Linux program cannot load the driver: glibc's linker warns that a static program using `dlopen` needs the shared libraries of the same glibc version at run time, and a static test program crashed when it tried (2026-11-27).
- **Windows is unaffected:** the static `.exe` loads `nvcuda.dll` from the system directory if it is there.
- The MSYS2 line with `-DMTV_CUDA` has not been run.

### Release binaries

The binaries on the [Releases page](https://github.com/account75783/mtvanity/releases) are built with the lines below, on Debian 13 under WSL. Both have the GPU backend, and both search on the CPU where there is no usable GPU.

Linux x86_64. A static program cannot load the NVIDIA driver, so this one is linked dynamically. It is built inside a Debian 11 root filesystem, with Debian 11's own gcc (10.2) and glibc (2.31). So it runs on any Linux with glibc 2.29 or newer, and it needs nothing but glibc (libc, libm, libdl, libpthread). The first line makes the root filesystem, once; use `http://archive.debian.org/debian` instead once Debian 11 has been moved there. The second builds in it, from the folder with the source.

```
sudo apt install -y debootstrap && sudo debootstrap --variant=minbase --include=gcc,libc6-dev bullseye /opt/bullseye http://deb.debian.org/debian
sudo cp mtvanity.c mtvanity_cuda.h /opt/bullseye/tmp/ && sudo chroot /opt/bullseye sh -c 'cd /tmp && gcc -O3 -s -pthread -DMTV_CUDA mtvanity.c -o mtvanity-linux-x86_64-V1.2 -lm -ldl' && cp /opt/bullseye/tmp/mtvanity-linux-x86_64-V1.2 .
```

Check which glibc it needs: `objdump -T mtvanity-linux-x86_64-V1.2 | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n1` prints `GLIBC_2.29`. Do not try `--sysroot` with Debian 13's own gcc instead: it takes Debian 11's headers but still links Debian 13's glibc, and the binary then needs glibc 2.34.

Windows x86_64, cross-compiled on Debian or WSL (this is how the release `.exe` is built):

```
sudo apt install -y gcc-mingw-w64-x86-64 && x86_64-w64-mingw32-gcc -O3 -s -static -Wa,-muse-unaligned-vector-move -DMTV_CUDA mtvanity.c -o mtvanity-windows-x86_64-V1.2.exe -lbcrypt
```

Windows x86_64, built on Windows itself (MSYS2 UCRT64 shell; not run with `-DMTV_CUDA`):

```
gcc -O3 -s -static -Wa,-muse-unaligned-vector-move -DMTV_CUDA mtvanity.c -o mtvanity-windows-x86_64-V1.2.exe -lbcrypt
```

A CPU-only build needs none of this: the lines under [Build](#build) with `-s -static` added give a Linux binary that needs no shared libraries at all.

The `-Wa,-muse-unaligned-vector-move` flag is required with MinGW gcc (GCC bug 54412: on 64-bit Windows it aligns the stack for 16 bytes, but moves 32- and 64-byte vectors as if they were aligned).

### Verifying a release

Releases are signed with the maintainer's PGP key (RSA 4096). Fingerprint:

```
F7DB ECD8 5737 C463 7C7A  9EE1 564D EAE8 5007 BB49
```

Download the binary you want and `SHA256SUMS.asc` into one folder.

Linux:

```
curl -s https://github.com/account75783.gpg | gpg --import && gpg --verify SHA256SUMS.asc && gpg -d SHA256SUMS.asc | sed 's#  .*/#  #' | sha256sum -c --ignore-missing
```

You need to see both `Good signature`, with the fingerprint above, and `OK` next to the file name. Then make the download executable: `chmod +x mtvanity-linux-x86_64-V1.2`.

Windows (Command Prompt, with Gpg4win installed):

```
curl -s https://github.com/account75783.gpg | gpg --import && gpg --verify SHA256SUMS.asc && gpg -d SHA256SUMS.asc && certutil -hashfile mtvanity-windows-x86_64-V1.2.exe SHA256
```

You need to see `Good signature`, with the fingerprint above, and the hash `certutil` prints must equal the one listed for the `.exe`.

- `gpg --verify` runs first, so a bad signature stops the check before any hash is compared.
- The checksums sit inside the signed message, so the hashes being compared are the ones that were signed.
- The `sed` step drops the folder names the list was made with, because release downloads arrive without folders.
- Windows shows a SmartScreen warning for the `.exe` because it carries no paid code-signing certificate; the PGP signature is how you know it is genuine.

## Use

```
./mtvanity c0de                  # node ID starting with c0de
./mtvanity c0ffee42              # exact 8-digit node ID
./mtvanity 'id:*beef'            # node ID ending with beef
./mtvanity key:Mesh              # public key starting with Mesh
./mtvanity 'key:*Radio='         # public key ending with Radio=
./mtvanity id:c0de+key:Me        # one key with both
./mtvanity dead beef key:Mesh    # three keys in one run, one for each target
./mtvanity --help                # everything else: threads, JSON output, limits
```

What you can write:

| You write | It means |
|---|---|
| `dead` or `id:dead` | the node ID (8 hex digits) starts with `dead` |
| `key:Mesh` | the public key, in the base64 form the apps show, starts with `Mesh` |
| `keyhex:c0ffee` | the public key written in hex starts with `c0ffee` |
| `id:c0de+key:Me` | one key that fits both patterns |
| `*` | any run of characters: `'id:*beef'` ends with `beef`, `'key:Me*Q='` starts with `Me` and ends with `Q=` |
| `.` | any one character: `id:d..d.00d` |
| `?d` `?l` `?u` `?h` `?H` `?s` `?a` | hashcat-style mask placeholders: a digit, a lower-case letter, an upper-case letter, a hex digit (lower case, upper case), `+` or `/`, any character |

- **Public key patterns are case-sensitive.** Add `-i` to accept either case, which also makes the search faster.
- **Quote patterns that contain `*` or `?`** so the shell leaves them alone. The Windows command prompt needs no quotes.
- **Not every ending exists.** A public key always ends in one of `AEIMQUYcgkosw048` followed by `=`, and the character before that has limits too. `mtvanity` says so when a pattern is impossible and lists what is possible.
- **Every target gets its own key, and they are searched together.** Five full node IDs take about 2.3 times as long as one, not 5 times. `--hits N` asks for N keys per target.
- **Keys go to stdout and everything else to stderr**, so `./mtvanity dead beef > keys.txt` keeps the keys off the screen.

Each key is printed as its own block, and a run with several targets ends with a summary:

```
$ ./mtvanity dead key:Mesh id:c0+key:AB
mtvanity V1.2 (source revision 6)
searching for 3 targets on 2 threads
  target 1   node ID !dead*                 about 65.5 thousand keys per hit
  target 2   public key Mesh*               about 16.8 million keys per hit
  target 3   node ID !c0* + public key AB*  about 1.05 million keys per hit

(two blocks like the one below, for targets 1 and 3)

==== key 3 of 3 | target 2 | public key Mesh* ====
  node ID:              !54db2e3a
  public key (base64):  MesheafarbpQesugyf8Z2aT4sWZ9C3vQCPxXuYX8/2o=
  public key (hex):     31eb2179a7daadba507acba0c9ff19d9a4f8b1667d0b7bd008fc57b985fcff6a
  private key (base64): ...
  private key (hex):    ...
  install with:         meshtastic --set security.private_key "base64:..."

3 of 3 keys found; 16.9 million keys tried in 0.18 s (92.1 million keys/s; kernels: avx512 x2)
  target 1   node ID !dead*                 !deadc9e8
  target 2   public key Mesh*               !54db2e3a
  target 3   node ID !c0* + public key AB*  !c071b756
```

Install the result on the radio (back up the old key first):

```
meshtastic --set security.private_key "base64:..."
```

### Searching on a GPU

This needs a build with `-DMTV_CUDA`, which both release binaries are, and an NVIDIA driver that finds a usable GPU.

```
./mtvanity c0ffee42                  # auto: CPU threads at once, the GPU as soon as it is ready
./mtvanity c0ffee42 --threads 8      # 8 CPU threads, and they keep searching beside the GPU
./mtvanity c0ffee42 --backend cuda   # the GPU alone
./mtvanity c0ffee42 --backend avx2   # the CPU only
./mtvanity --benchmark               # a cuda line, or the reason there is none
./mtvanity --cuda-ptx                # print the GPU kernel (PTX text) for review
```

- **No search waits for the GPU.** Under `auto` every CPU thread starts searching at once, as in a build without the option, while the GPU is brought up and checked beside them. A short search is over before the GPU is ready and never touches it.
- **The GPU takes over when it has finished its first launch.** The CPU threads then stop, unless the search is about to end anyway or you asked for them with `--threads N`. A line on stderr says when that happened.
- **The CPU threads stop only for a GPU that is clearly faster.** Its first launch must reach 1.5 times the speed the CPU threads had. A small GPU beside a big CPU, such as an old GeForce next to an AVX-512 CPU, then searches beside the CPU threads instead of making the search slower. The line on stderr gives both speeds.
- **The first line of a search says what runs:** `30 threads until the GPU is ready`, `8 threads and the GPU when it is ready`, `on the GPU (name)` or plain `32 threads`.
- **On Linux a long search leaves two CPUs free while the driver starts,** hence 30 threads on a 32-thread CPU. At most one thread in eight waits: a CPU with 8 to 15 threads leaves one free, and a smaller one none. The GPU is ready sooner that way (see [CUDA.md](CUDA.md)). A search expected to need fewer than 2^30 keys starts every thread at once, and so does the Windows build, where free CPUs gained nothing. If the GPU does not take over, the threads that waited start as soon as that is known.
- **CPU threads that search on time their kernels again** once the GPU's start-up is over: beside the GPU, after a GPU that failed, or when no GPU could be used. Their first timing runs beside the driver's start and can pick the slow kernel.
- **Every pattern works the same** on the GPU: node ID, `key:`, `keyhex:`, combined targets, masks, `-i`, several targets, `--hits`.
- **Without a driver or a usable GPU, a line on stderr says why** and the search runs on the CPU, for example `no usable GPU: no NVIDIA driver library (libcuda.so.1) was found; searching on the CPU`. `--quiet` hides that line. `--backend cuda` without a usable GPU also says why and searches on the CPU instead of stopping.
- **If the GPU fails during a search,** for example in a driver reset, a warning names the call and the CUDA error, and the CPU threads carry the search on. Under `--backend cuda` they start at that moment. Keys already printed stand. A key the GPU had found but not yet handed over is lost, never printed wrong.
- **Laptops get hot.** A long search runs the GPU at full power. On the test laptop the GPU reached its power limit within seconds, and in one run the machine switched itself off from heat (2026-11-27). Give it good cooling, or a lower GPU power limit, and keep an eye on it.

### What changed in V1.2

New:

- **Search on an NVIDIA GPU** (source revision 6), in both prebuilt binaries. See [Searching on a GPU](#searching-on-a-gpu).
- **The AVX-512 kernel** (source revision 5) in the prebuilt binaries.

Changed:

- **The prebuilt Linux binary is linked dynamically**, so that it can load the NVIDIA driver, and needs glibc 2.29 or newer. The V1.1 one was static.
- **Speeds are written out**, for example `3.33 billion keys/s` where V1.1 wrote `3330.0 M/s`, in the benchmark, the status line and the summary. `--json` is unchanged.
- **`--benchmark` gives the speed after start-up**, measured between two moments at which keys came in, so a slow start no longer pulls the figure down.
- **A failed check before a search is reported on stderr**, so that stdout keeps only keys.

### What changed from V1 to V1.1

New:

- **Vanity public keys.** `key:` chooses how the base64 public key starts or ends, and `keyhex:` does the same for its hex form.
- **A node ID and a public key in one key**, such as `id:c0de+key:Me`.
- **Several keys in one run.** Up to 64 targets are searched together, each gets its own key, and the run ends with a summary.
- **Mask patterns.** `*` for any run of characters, `.` for any one, and the hashcat-style placeholders `?d ?l ?u ?h ?H ?s ?a`.
- **`-i`** accepts either case in a public key pattern.
- **An AVX-512 kernel**, since source revision 5, for CPUs with AVX-512 IFMA: eight walks per thread where the AVX2 kernel runs four, on multiplications that are twice as wide. A thread uses it where it is the fastest; `--backend avx512` forces it and `--backend avx2` keeps it out.

Changed:

- **`*` now means "any run of characters".** In V1 it stood for a single digit; write `.` for that now. V1's `x` is refused, and `?` now starts a mask placeholder such as `?d`, so if you reuse an old pattern, check the line `mtvanity` prints before it starts.
- **A list finds a key for every entry.** In V1, `dead,beef` stopped at the first match of either. Now both are found, and `--hits` counts per target.
- **The printed block has a header line**, and JSON output gained `target` and `pattern`.
- **Every search starts by naming its build**, for example `mtvanity V1.1 (source revision 5)`.
- **The status line fits the width of the terminal.** When stderr is a file or a pipe, a plain progress line is written now and then instead.
- **`kill` ends a search the way Ctrl-C does**, with the summary.

Fixed:

- **The Windows console keeps up.** Every message is written in one piece, and each search thread lets anything that is waiting for its CPU run first, a few thousand times a second. Before, with every CPU searching, the status line landed on a new row at each update.
- **A key that cannot be written to stdout stops the search** with an error and exit status 1, instead of being lost without a word.
- **Secrets are handled more tightly.** Two leftover temporaries are wiped, and a private key no longer passes through a C library buffer.
- **Smaller things:** a CRC-32 that was computed for nothing with some combined targets, the thread count on Windows when the CPU mask is unavailable, and leftovers of `^C` on the summary line.

The scalar and AVX2 kernels are the ones from V1, and as fast as they were. The AVX-512 kernel is new.

## Performance

| Machine | Kernel | Keys per second | Average time for a full 8-digit ID |
|---|---|---|---|
| Intel Core i9-14900HX, 32 threads, Linux build under WSL | AVX2 | 516.4 M | about 8.3 s |
| Same laptop, native Windows build | AVX2 | 495.9 M | about 8.7 s |
| Same laptop, native Windows build | Scalar | 275.7 M | about 15.6 s |
| 2 shared cloud cores (Xeon Sapphire Rapids, about 3 GHz, gcc 13.3) | AVX2 | 60.2 M | about 71 s |
| Same 2 cloud cores | Scalar | 33.2 M | about 2.2 min |
| 2 shared cloud cores (Xeon Emerald Rapids, about 3.3 GHz, gcc 13.3), source revision 5 | **AVX-512** | **170.8 M** | **about 25 s** |
| Same 2 cloud cores, same minute | AVX2 | 50.4 M | about 85 s |
| Same 2 cloud cores, same minute | Scalar | 29.2 M | about 2.5 min |
| AWS EC2 c8a, 32 cores (AMD EPYC, Zen 5), 32 threads, source revision 5 | **AVX-512** | **6,662.5 M** | **about 0.65 s** |
| Same c8a instance, same run | AVX2 | 1,608.9 M | about 2.7 s |
| Same c8a instance, same run | Scalar | 856.6 M | about 5.0 s |

Details of the laptop run are in [BENCHMARK-i9-14900HX.md](BENCHMARK-i9-14900HX.md). How the AVX-512 kernel works is explained in [AVX512.md](AVX512.md).

**The AVX-512 kernel** was measured on 2026-11-25 with source revision 5, on the Emerald Rapids machine only. Its three rows are the middle one of three runs of `./mtvanity --benchmark` taken back to back: AVX-512 did 169 to 187 M keys per second, AVX2 44 to 53 M and the scalar kernel 23 to 31 M (it is a shared machine, and its speed wanders). On one thread it was 92 to 95 M against 25 to 27 M. That makes the AVX-512 kernel about 3.5 times as fast as the AVX2 one there.

**On AMD Zen 5** it was measured on 2026-11-25 by the human, with source revision 5 built with gcc on Debian 13 on a 32-core AWS EC2 c8a instance: one run of `./mtvanity --benchmark`. AVX-512 did 6,662.5 M keys per second, AVX2 1,608.9 M and the scalar kernel 856.6 M, so the AVX-512 kernel is about 4.1 times as fast as the AVX2 one there, and every one of the 32 threads chose it by itself. That is about 208 M keys per second per core, and about 11 times the i9-14900HX laptop. That is the benchmark's figure; in a long search on the same instance (Debian 13) the human saw about 5,700 M keys per second, some 15% less, and why has not been looked into. Zen 4 has not been measured.

The other rows were measured with V1. The scalar and AVX2 kernels have not changed since, and V1.1 runs at the same speed with them. Run back to back on the two cloud cores (on a day the machine was a little faster than when the table was made):

- V1 and V1.1 both did 67 to 70 M keys per second for one node ID.
- V1.1 did 72 M for a public key alone, and 63 M with 64 node IDs at once.
- The slow case is a pattern with no fixed character at all, such as one made only of `?l`: about 42 M, because every key has to be looked at in full.

On the laptop under Windows, V1.1 ran at 515.6 M keys per second with `--threads 24`, one thread per core. That matches the best figure in the table, which used all 32 threads.

**The CUDA backend** (source revision 6), on the i9-14900HX laptop's RTX 4080 Laptop GPU with driver 610.88. Speeds are from `--benchmark`, which since this revision reports the speed after start-up. **The speed follows the power the laptop gives the GPU**, so every figure carries its power state.

| Measured | Build | GPU power state | `cuda` line |
|---|---|---|---:|
| 2026-11-27, nothing else running | Linux under WSL | Windows "Balanced", on mains, GPU power limit reported as 85 W; the GPU drew about 40 W, 66 to 73 °C | 3.22 billion keys/s over 10 s, 3.33 billion over the next 30 s (twice) |
| Same hour, back to back | Windows `.exe`, started from WSL | the same | 3.22 billion over 10 s, 3.35 billion over the next 30 s |
| 2026-11-27, earlier, nothing else running | Linux under WSL | limit reported as 175 W; 122 W drawn, 68 °C at the start, 86 °C after 10 s | 6.60 billion over the first 10 s, then 4.56 billion averaged over 120 s |
| By the human, native Windows from PowerShell, cool laptop (an earlier build of revision 6) | Windows `.exe` | not recorded | 7.20 billion (5 s line) |

- **WSL costs no GPU speed:** the Linux build under WSL and the Windows build ran within 1% of each other, back to back.
- **The 175 W state is not safe on this laptop for long.** The run after the 120 s line ended with the laptop overheating and switching off. The figures at about 40 W are from the state it came back in.
- **The CPU alone:** 557.8 million keys/s with AVX2 and 303.6 million scalar (native Windows, by the human, cool laptop); 512.9 million and 277.3 million under WSL in the build run.

Time from starting the program to the first key (2026-11-27, WSL, the laptop in the 40 W state, in which everything was slower than usual):

| Target | CPU-only build | GPU build, `auto` |
|---|---|---|
| 4-digit node ID | 0.25 to 0.26 s | 0.24 to 0.27 s |
| 6-digit node ID | 0.25 to 0.41 s | 0.27 to 0.40 s |
| `key:Mes` | 0.25 to 0.28 s | 0.24 to 0.27 s |
| Full 8-digit node ID | 2.6, 13.9 and 23.1 s in three runs | 2.8 to 4.0 s in six runs |

With an empty driver cache the short searches took the same, and the 8-digit ID 0.9 to 13.2 s in four runs: once the driver needed 8.4 s to compile the kernel beside the 32 searching CPU threads. More in [CUDA.md](CUDA.md).

How it gets there:

- Affine Montgomery walk with a shared denominator and batch inversion: 2.5 multiplications and 1 squaring per key.
- AVX2 kernel running four walks per thread.
- AVX-512 kernel running eight walks per thread, where the CPU has AVX-512 IFMA: its 52-bit multiplications get a field element into five limbs where AVX2 needs ten.
- CRC-32 by five carry-less multiplications, no tables.
- Each thread times every kernel its CPU can run, on its own core, and keeps the fastest.
- One comparison per key: each target is boiled down to the bits every match must share, and a key is looked at closely only if those bits fit. When only public keys are asked for, the CRC-32 is skipped.

## Verification status

Checked:

- 57 self-test checks on a CPU with AVX-512 and 46 on one without, including the RFC 7748 vectors, a published Meshtastic key, a full window of every kernel compared key by key against an independent X25519 ladder, keys planted at chosen places in a window, and the pattern matcher compared against a separate reading of the pattern language.
- Every hit is re-derived by the independent reference code before it is printed.
- 2,511 keys for node ID, public key and combined targets were re-derived outside the program by separate Python code (its own X25519, zlib's CRC-32, Python's base64 and a regex reading of each pattern). All matched. The same 162-key list was run with each kernel forced in turn, the AVX-512 one included.
- The AVX-512 kernel was broken on purpose in 47 places, one at a time: a product in the wrong column, a missing doubling, a limit off by one, parts of the table joined wrongly, a key reported under its neighbour's number. The self-test caught all 47, and so did the short check every search starts with. Three of them got past the first version of the tests, which were extended until they did not; the same new test now also covers two cases in the older kernels that nothing checked before (a match on the centre key of a window, and a second match from a base that has already given a key).
- Clean under ASan, UBSan and TSan (gcc) with each kernel, and under strict warnings with gcc and clang. For the AVX-512 code one ASan feature, the use-after-return check, has to be off: gcc 13 misplaces 64-byte variables when it is on.
- Under valgrind, which behaves like a CPU without AVX-512 and stops at any instruction it does not know, the self-test passes and a search finds its keys. So the new kernel stays out of the way where it cannot run.
- An exact 8-digit ID was found end to end, in the same run as a vanity public key. With revision 5 and the AVX-512 kernel that took 25 s on the two cloud cores, at 166 M keys per second, which is the average for a full ID there.
- Confirmed by the human on a real radio running firmware 2.8 (2026-11-17), with a V1 key: after a generated key is written, the node ID re-derives to the chosen value.
- The static Linux x86_64 release build of V1.1 passed the self-test and ran at the same speed as the normal build.
- The native Windows x86_64 build (static, MinGW-w64 gcc) passes the full self-test on Windows 11 on the i9-14900HX (V1.1 source revision 4 on 2026-11-21; V1 also ran the benchmark there on 2026-11-18).
- On that laptop under Windows, with a search thread on each of its 32 CPUs, the status line stays in place. Before the search threads gave way to other programs it did not: it landed on a new row at each update unless `--threads` was set below the number of CPUs.

Checked for the CUDA backend (source revision 6, 2026-11-27, RTX 4080 Laptop GPU):

- 11 more self-test checks, 57 in all on that laptop: tables, batched seeding and the kernel text (needs no GPU, so a GPU build on a machine without one has 47); the GPU's field arithmetic, inversion, canonical forms and CRC-32 against the scalar code on 4,096 values each; 8,306 keys of eight GPU walks against the reference ladder and CRC-32; planted keys; a centre forced onto a table point; an overfull record buffer; real searches for 10 and for 4 targets at once.
- The Linux build under WSL and the Windows build (cross-compiled with MinGW-w64, started from WSL) both pass all 57. By the human, natively from PowerShell: an earlier build of revision 6 passed its 56.
- Under `auto` the GPU is checked while the CPU threads already search: field arithmetic, keys of four walks against the reference ladder, and a planted key that must come back.
- The GPU kernel was broken on purpose in 38 places, one at a time: the self-test caught all 38 in the build run, and all 38 again on the final kernel. The list is in [CUDA.md](CUDA.md).
- 310 keys found on the GPU in the build run (node ID, public key, combined and `-i` targets; 60 of them with the Windows build) were re-derived outside the program by separate Python code. All matched.
- 3,367 more keys found on the GPU on 2026-11-27 were re-derived by a separate checker, written apart from mtvanity, in Python with only the standard library. They came from the Linux build and the Windows `.exe`, and covered node ID, public key and combined targets, 64 random full 8-digit node IDs among them.
  - The checker recomputed each public key from its private key by RFC 7748, the CRC-32 node ID and the pattern.
  - It also checked that no private key repeats or lies a small offset from another.
  - All passed. A look across 3,067 of them for stuck bits and for zeroed or shared 64-bit chunks of the private keys found none.
  - These keys came from revision 6 before the V1.2 changes below.
- Checkers on revision 6 (gcc 14, 2026-11-27, Linux under WSL):
  - **ASan with UBSan** reports nothing for the CPU build (self-test and a search) or for the GPU build without a driver. With the GPU the self-test passes, and the only report is memory the NVIDIA driver library keeps at exit.
  - **TSan** reports nothing for the CPU build (full self-test and a search), for the GPU build's host code, or for searches that start on CPU threads and hand over to the GPU. On this kernel it only runs with address randomisation off (`setarch x86_64 -R`).
  - **valgrind** reports nothing for a search with the CPU build or with the GPU build without a driver. With the GPU the search works, but valgrind reports uninitialised values: most inside the driver's libraries, and 48 attributed to the routine that seeds the walks. That routine is clean under valgrind when it runs without the driver.
  - **The cause of the valgrind reports is confirmed** (2026-11-28, `valgrind --track-origins=yes`, a search with `--backend cuda` on the real driver). Of 841 reports about uninitialised values, 835 say where the value was created, and every one is memory allocated inside the driver's libraries: 780 in `libnvdxgdmal.so.1` called from `libcuda.so.1.1`, 52 in `libdxcore.so`, 3 in `libcuda.so.1.1`. The other six are raised inside `libcuda.so.1.1`. None starts in mtvanity's code. The reports in mtvanity's own routines are values the driver handed back: the GPU's size, from which the number of walks is worked out, and the result buffers.
- Built without `-DMTV_CUDA`, the self-test has its 46 checks on that laptop and passes, with gcc, clang, the static Linux build and the MinGW-w64 build. With a driver library that does not exist, the GPU build searches on the CPU.
- No warnings under `-Wall -Wextra` with gcc 14, clang 19 and MinGW-w64 gcc 14, with and without `-DMTV_CUDA`.

Checked for the V1.2 changes (2026-11-28, on a cloud machine with an AVX-512 CPU and no GPU, gcc 13 and clang 18):

- **The GPU kernel is untouched.** The PTX that `--cuda-ptx` prints is the same, byte for byte, as before the changes (18,690 lines), so the checks of the kernel above still stand. The changes are in the host code around it.
- **Without any NVIDIA driver,** the GPU build passes its self-test (58 checks there) and finds its keys on the CPU, with the line that says why, under `auto`, under `--backend cuda` and with `--quiet`.
- **Against a stand-in driver library that fails on purpose,** every failure prints its line on stderr and the search finds its keys on the CPU. The failures were:
  - no device
  - a refused kernel
  - a failed start-up check
  - a wait or a launch that fails partway through a search, under `auto`, with `--threads` and under `--backend cuda`
  - a GPU too slow to be worth stopping the CPU threads for

  The keys from these runs were re-derived by the separate checker, and all passed. TSan, and ASan with UBSan, report nothing for these runs.
- **Number arguments are read without `strtoll` now.** 108 inputs to `--threads`, `--hits`, `--max-keys` and `--benchmark` give the same results as before. They range from 0 and leading zeros to signs, spaces and overflow.
- **A dynamically linked build needs glibc 2.34 at most when built on glibc 2.39,** where it needed 2.38 before because of `strtoll`.
- No warnings under `-Wall -Wextra` with gcc 13 and clang 18, with and without `-DMTV_CUDA` and the fault option. MinGW-w64 was not available there.

Checked on the test laptop for V1.2 (2026-11-28, RTX 4080 Laptop GPU, driver 610.88, Windows 11 with WSL2).

By the human, with the V1.2 files before the fixes below:

- **The prebuilt Linux binary** needs `GLIBC_2.29` and loads libm.so.6, libdl.so.2, libpthread.so.0 and libc.so.6. Its self-test passes 57 of 57 checks on the GPU, and 47 of 47 inside the Debian 11 root filesystem, where there is no driver. A search there printed the "no usable GPU" line and found its key on the CPU.
- **The Windows `.exe`** passes 57 of 57 on the GPU. It imports `bcrypt.dll`, `KERNEL32.dll` and `msvcrt.dll` only.
- **140 keys** found with `--backend cuda` by the two release binaries passed the separate checker.
- **A real GPU made to count as lost** (`--cuda-fail-at 1`, target `c0ffee42`, fault build). Under `auto` the GPU was ready after 2.64 s and the CPU threads stopped; the GPU was then lost, the CPU took over and found the key in 7.20 s (kernels: cuda x1, avx2 x29, scalar x3). Under `--backend cuda` the CPU threads started when the GPU was lost and found the key in 8.10 s (avx2 x32).

By Claude Fable 5.2, the same day, for the fixes (the laptop ran hot and throttled for part of it, so CPU speeds were as low as 250 million keys/s):

- **Every CPU thread that searches on ends on its fastest kernel.** The same `--cuda-fail-at 1` runs now end with `cuda x1, avx2 x32` under `auto` (three runs) and under `--backend cuda` (three runs). A stand-in driver that takes a second to report no GPU ends with `avx2 x32`.
- **Two CPUs left free while the driver starts (Linux under WSL2).** Time until the GPU is ready, 13 runs each, alternating: 1.39 to 3.40 s, median 1.71 s, with all 32 threads searching; 1.13 to 3.33 s, median 1.34 s, with two CPUs free. With an empty driver cache, three runs each: 9.2 to 10.6 s against 6.3 s. The driver's own start (`cuInit`) varied from 0.3 to 2.5 s either way. On native Windows free CPUs gained nothing (medians 1.12 s and 1.15 s, five runs each), so the `.exe` leaves none. The tables are in [CUDA.md](CUDA.md).
- **No short search got slower:** 4-digit, 6-digit and `key:Mes` targets start every thread at once, as before, and took the same time to the first key in the old and the new build (six runs each).
- **The GPU kernel is untouched:** `--cuda-ptx` gives SHA-256 `aa78401b...5e7d74` as before, from the Linux build and, with the carriage returns removed, from the `.exe`.
- **A build without `-DMTV_CUDA` is byte for byte the binary it was** before the fixes (gcc 14, same command line).
- **Self-tests after the fixes:** 57 checks on the GPU with the Linux build made by Debian 11's gcc 10.2 and with the Windows `.exe`; 46 with the CPU-only builds; 47 with the driver library hidden. The Debian 11 build still needs glibc 2.29.
- **TSan** reports nothing for five kinds of run: `auto` with a hand-over, `auto` with the GPU lost after two launches, `--backend cuda` with the GPU lost, a stand-in driver without a GPU, and a search that ends while the driver is starting. The last one first showed a report in the summary code, which is fixed.
- **46 keys** found after the fixes (Linux build and `.exe`, 23 each: every kind of target, with `--backend cuda` and under `auto`) passed the separate checker.
- No warnings under `-Wall -Wextra` with gcc 14, clang 19, MinGW-w64 gcc 14 and Debian 11's gcc 10.2, with and without `-DMTV_CUDA` and the fault option.

By Claude Opus 5.5, after the fixes (2026-11-28, 2 cloud Xeon cores, Emerald Rapids, Ubuntu 24.04 with glibc 2.39, no NVIDIA driver):

- **The fix run's Debian 11 build** (gcc 10.2, the release command line, the final sources) passes 58 of 58 checks there, the AVX-512 kernel included. 65 keys it found, with each CPU kernel and every kind of target, passed the separate checker.
- **Its CPU kernels are as fast as a gcc 13 build** of the same sources, within the noise of three alternating `--benchmark 2` runs: AVX-512 170 to 180 against 158 to 179 million keys/s.
- **A build without `-DMTV_CUDA` is byte for byte the V1.2 binary** with gcc 13 and clang 18 as well, and `--cuda-ptx` still gives `aa78401b...5e7d74`.
- **Against the stand-in driver,** now also one that takes 0.5 to 3 s to start, with each failure listed above, a GPU too slow to stop for, and `--threads 4`, `8` and `16` beside it:
  - Every search ended with all its CPU threads on the AVX-512 kernel, the fastest there, except one with 16 threads on the 2 CPUs: when they timed their kernels again, 3 of the 16 chose AVX2.
  - With `--threads 8` and `16` the threads that waited started, and the summary counts all of them.
  - TSan, and ASan with UBSan, report nothing for seven of these runs, among them a lost GPU, a GPU without a device that takes a second to say so, and a search that ends while the driver is starting. TSan does show the summary-code race in the V1.2 code from before the fixes.
  - 27 keys from these runs passed the separate checker.
- No warnings under `-Wall -Wextra` with gcc 13 and clang 18, with and without `-DMTV_CUDA` and the fault option.

Not yet checked:

- **The CUDA backend has run on one GPU and one driver only.** The machine code the driver makes from the kernel has not been read (that needs NVIDIA's toolkit). The tuning of the finished kernel was cut short by an overheated laptop: blocks per multiprocessor, the checkpoint interval, the launch length and a filter kept in registers were tried in the throttled 40 W state, and none beat the noise. The rest was not tried. No key found on the GPU has been written to a radio. Details are in [CUDA.md](CUDA.md).
- **No real GPU has failed by itself in the middle of a search.** A stand-in driver library, and a real GPU made to count as lost after N launches (`--cuda-fail-at N` in a build with `-DMTV_CUDA_FAULTS`), were used instead.
- **Leaving two CPUs free for the driver was measured under WSL2 only,** on one 32-thread CPU. Linux without Windows underneath, and smaller CPUs, were not measured.
- **CPU threads can still pick the scalar kernel while the driver starts:** 1 to 5 of 30 did in most runs (2026-11-28). They stop when the GPU takes over, and they time their kernels again if they search on. The first timing itself was not made robust against a busy CPU.
- **A short search under `auto` can take long to exit after its key:** 0.01 to 0.8 s, and 1.6 s once, where the CPU-only build takes 0.02 to 0.07 s (2026-11-28, 4-digit, 6-digit and `key:Mes` targets, six runs each). The search ends while the driver is still starting. What the exit waits for was not established (`strace` is not installed on the test machine); the driver call still in flight is the suspect. Not fixed. After a key found on the GPU the exit took 0.06 to 0.15 s, under `auto` and under `--backend cuda`.
- **The rule that keeps the CPU threads beside a slow GPU** has been tested with a stand-in driver only, because no small GPU was at hand.
- **The prebuilt Linux binary has been run on Debian only.** A build made the same way from the same sources passed its self-test on Ubuntu 24.04 and was as fast there as a gcc 13 build, on 2 cloud cores. On the 32-thread laptop its speed has not been compared with that of a gcc 14 build. The GPU kernel is the same either way, because the program writes it at run time.
- **The AVX-512 kernel has run on two CPUs only**, an Intel Xeon (Emerald Rapids) and an AMD EPYC (Zen 5), both in the cloud under Linux. On the Zen 5 only the benchmark was reported; the output of its full self-test has not been seen. AMD Zen 4 and Intel's desktop and laptop CPUs with AVX-512 have not been tried.
- **The AVX-512 kernel has never run under Windows.** The test laptop has no AVX-512, so it only ever shows the other two kernels.
- No key chosen by its public key has been written to a radio yet.
- The Windows crash-dump suppression has never been exercised by a real crash.
- The MSYS2 build line for Windows has not been run; the tested `.exe` was cross-compiled.
- No measurements on a Ryzen 9 or on real ARM hardware.

## Security notes

- Keys come from the OS random generator only (`getrandom`, `arc4random_buf`, `BCryptGenRandom`).
- With the GPU backend, private scalars stay on the host. Each of the GPU's walks has its own scalar from the OS random generator; the GPU sees public points only, and every hit is recomputed on the host before it is printed.
- Private keys are written to stdout only, secrets are wiped after use, and core dumps (Linux) and crash dumps (Windows, best effort) are disabled.
- The search is not constant-time. Run it on a machine you trust. On a rented cloud machine, copy the key off and terminate the instance when you are done, so its disk goes with it.
- A node ID is a 32-bit label, not proof of identity: anyone can generate a key with the same ID. The public key is what identifies a node, and only the whole of it: anyone can also generate a key that starts or ends like yours.
- A vanity key is as strong as any other. The search fixes a few public characters and leaves the private key random.
- IDs `!00000000` to `!00000003` and `!ffffffff` are reserved and refused.

## TODO

Planned, with nothing written yet:

1. **AMD GPU hardware acceleration.**

It is meant to come the way the Nvidia one did (source revision 6, `mtvanity_cuda.h`): as an optional `.h` header next to `mtvanity.c`, so that the CPU build stays a single C file with no extra dependencies.

## License

MIT. See [LICENSE](LICENSE).

## Support

If mtvanity gave you a fun node ID, tips are welcome in Monero (XMR):

    87MusZ465nXZUb96BfWjgMjDfAqEffuahNjsyefTVhwuF9QQJmw9qDM7auju8vFdpAVJfj8bZZ9SKj5DdQaKKqD1LBnu3pD

This address is signed with the same key as the releases. Verify it with `gpg --verify DONATE.txt.asc`.
