# mtvanity

Vanity node ID generator for Meshtastic 2.8+ radios (Heltec V4 and others), in a single C file.

**Release:** V1-RC1 (release candidate, 2026-11-17)

**Tested and working on Meshtastic firmware 2.8, the latest release, on 2026-11-17.** A generated key was written to a real radio and the node ID changed to the chosen value. (2.8.1 alpha was the current firmware release on that date.)

## Authorship

**Authored by Claude Fable 5.2 (Max, with Research), assisted by a human.**

- Design, code, tests and this document were written by Claude Fable 5.2 (model ID `claude-fable-5-2`), Anthropic's AI model.
- 2+ hours of research and engineering so far.
- A human set the goal, chose the target hardware, corrected the firmware assumptions and directed the work.

## What it does

From firmware 2.8, a Meshtastic node ID is the CRC-32 of the node's 32-byte X25519 public key, shown as `!xxxxxxxx`. `mtvanity` searches for a private key whose public key hashes to the ID you want.

## Requirements

- **64-bit only.** An x86_64 or 64-bit ARM CPU with a 64-bit operating system. The arithmetic uses 128-bit integers, so a 32-bit build is not possible.
- **Minimum:** any 64-bit x86 CPU. AVX2 and carry-less multiply are used when the CPU has them and are not required.
- **Recommended:** 4 cores of Intel 4th-gen Core (Haswell, 2013) or newer, or AMD Ryzen 1st gen (2017) or newer. That is predicted to find a full 8-digit ID in about a minute or less.
- **Windows uses at most 64 logical CPUs.** A machine with more than 64 will only cost you money on Windows: the extra cores sit idle. Linux uses up to 1024.
- **Systems:** Debian 12/13, Windows 11, Android (Termux).

### How long a full 8-digit ID takes

A full ID needs 4.29 billion keys on average. Rule of thumb: about 90 CPU cycles per key with AVX2 and about 185 without, so

```
seconds = 4.29e9 x cycles per key / (cores x clock speed in Hz)
```

Measured:

| CPU | Keys per second | Average time |
|---|---|---|
| Intel Core i9-14900HX (laptop, 32 threads), Linux build under WSL | 516 M | about 8 s |
| Same laptop, native Windows build | 496 M | about 9 s |
| 2 cloud Xeon cores (Sapphire Rapids) | 62 M | about 70 s |

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
| Single-core Athlon 64 or 64-bit Pentium 4 | 3 to 5 M | 15 to 20 min |
| First-generation Atom netbook or nettop (D510, N450) | about 2 M | about 35 min |

- **Averages hide bad luck.** One search in twenty takes more than three times the average, so the last row can run for two hours.
- **Shorter IDs are far easier.** Each digit you drop is 16 times faster; a 7-digit prefix takes about 2 minutes even on the slowest row.
- **Cannot run at all:** 32-bit CPUs and 32-bit operating systems. That covers the Pentium M, Core Duo, Atom N270, Raspberry Pi Zero and Pi 1, and 32-bit Raspberry Pi OS on any Pi.

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

### Release binaries

Release builds add `-s -static`, so the program needs no shared libraries.

Linux x86_64:

```
gcc -O3 -s -static -pthread mtvanity.c -o mtvanity-linux-x86_64 -lm
```

Windows x86_64, cross-compiled on Debian or WSL (this is how the release `.exe` is built):

```
sudo apt install -y gcc-mingw-w64-x86-64 && x86_64-w64-mingw32-gcc -O3 -s -static -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity-windows-x86_64.exe -lbcrypt
```

Windows x86_64, built on Windows itself (MSYS2 UCRT64 shell):

```
gcc -O3 -s -static -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity-windows-x86_64.exe -lbcrypt
```

### Verifying a release

Releases are signed with the maintainer's PGP key. Fingerprint: `PASTE-KEY-FINGERPRINT-HERE`

```
curl -s https://github.com/account75783.gpg | gpg --import && gpg --verify SHA256SUMS.asc SHA256SUMS && sha256sum -c --ignore-missing SHA256SUMS
```

Check that the fingerprint `gpg` prints matches the one above. Windows shows a SmartScreen warning for the `.exe` because it carries no paid code-signing certificate; the PGP signature is how you know it is genuine.

The `-Wa,-muse-unaligned-vector-move` flag is required with MinGW gcc (GCC bug 54412: 32-byte stack alignment on 64-bit Windows).

## Use

```
./mtvanity c0de          # ID starting with c0de
./mtvanity c0ffee42      # exact 8-digit ID
./mtvanity --help        # wildcards, lists, threads, JSON output
```

Install the result on the radio (back up the old key first):

```
meshtastic --set security.private_key "base64:..."
```

## Performance

| Machine | Kernel | Keys per second | Average time for a full 8-digit ID |
|---|---|---|---|
| Intel Core i9-14900HX, 32 threads, Linux build under WSL | AVX2 | 516.4 M | about 8.3 s |
| Same laptop, native Windows build | AVX2 | 495.9 M | about 8.7 s |
| Same laptop, native Windows build | Scalar | 275.7 M | about 15.6 s |
| 2 shared cloud cores (Xeon Sapphire Rapids, about 3 GHz, gcc 13.3) | AVX2 | 60.2 M | about 71 s |
| Same 2 cloud cores | Scalar | 33.2 M | about 2.2 min |

Details of the laptop run are in `BENCHMARK-i9-14900HX.md`.

How it gets there:

- Affine Montgomery walk with a shared denominator and batch inversion: 2.5 multiplications and 1 squaring per key.
- AVX2 kernel running four walks per thread.
- CRC-32 by five carry-less multiplications, no tables.
- Each thread times both kernels on its own core and keeps the faster one.

## Verification status

Checked:

- 32 self-test checks, including the RFC 7748 vectors, a published Meshtastic key, and a full window compared key by key against an independent X25519 ladder.
- Every hit is re-derived by the independent reference code before it is printed.
- Clean under ASan, UBSan and TSan (gcc), and under strict warnings with gcc and clang.
- An exact 8-digit ID was found end to end.
- Confirmed by the human on a real radio running the latest 2.8 firmware (2026-11-17): after a generated key is written, the node ID re-derives to the chosen value.
- The static Linux x86_64 release build passes the self-test and runs at the same speed as the normal build.
- The native Windows x86_64 build (static, MinGW-w64 gcc) passes the full self-test and the benchmark on Windows 11 on the i9-14900HX (2026-11-18).

Not yet checked (why this is a release candidate):

- The Windows crash-dump suppression has never been exercised by a real crash.
- The MSYS2 build line for Windows has not been run; the tested `.exe` was cross-compiled.
- No measurements on a Ryzen 9 or on real ARM hardware.

## Security notes

- Keys come from the OS random generator only (`getrandom`, `arc4random_buf`, `BCryptGenRandom`).
- Private keys are written to stdout only, secrets are wiped after use, and core dumps (Linux) and crash dumps (Windows, best effort) are disabled.
- The search is not constant-time. Run it on a machine you trust.
- A node ID is a 32-bit label, not proof of identity: anyone can generate a key with the same ID. The public key is what identifies a node.
- IDs `!00000000` to `!00000003` and `!ffffffff` are reserved and refused.

## License

MIT.

---

## ⚠️ CHECK THAT THE ID IS OPEN FIRST

**Do not use another person's node ID.** Two nodes with the same ID disrupt traffic on the mesh: messages, routing and other people's node lists can all get confused, for them and for you.

Before you write a key to your radio, check that nobody on your mesh already has that ID:

```
meshtastic --nodes | grep -i NODEHASHHERE || echo "not seen by this radio"
```

Your radio only knows the nodes it has heard, so also check your local mesh map or ask in your mesh's group chat.

Take someone else's ID on purpose and expect to get T-hunted: the people whose mesh you broke will track down your transmitter.
