# mtvanity

Vanity node ID generator for Meshtastic 2.8+ radios (Heltec V4 and others), in a single C file.

**Release:** V1-RC1 (release candidate, 2026-11-17)

## Authorship

**Authored by Claude Fable 5.2 (Max, with Research), assisted by a human.**

- Design, code, tests and this document were written by Claude Fable 5.2 (model ID `claude-fable-5-2`), Anthropic's AI model.
- 2+ hours of research and engineering so far.
- A human set the goal, chose the target hardware, corrected the firmware assumptions and directed the work.

## What it does

From firmware 2.8, a Meshtastic node ID is the CRC-32 of the node's 32-byte X25519 public key, shown as `!xxxxxxxx`. `mtvanity` searches for a private key whose public key hashes to the ID you want.

## Build

Debian 12/13:

```
gcc -O3 -pthread mtvanity.c -o mtvanity -lm && ./mtvanity --self-test && ./mtvanity --benchmark
```

Windows 11 (MSYS2 UCRT64):

```
gcc -O3 -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity.exe -lbcrypt && ./mtvanity.exe --self-test
```

Android (Termux):

```
clang -O3 -march=armv8-a+crc -pthread mtvanity.c -o mtvanity -lm && ./mtvanity --self-test
```

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

Measured on 2 shared cloud cores (Xeon Sapphire Rapids, about 3 GHz, gcc 13.3):

| Kernel | Keys per second | Average time for a full 8-digit ID |
|---|---|---|
| Scalar | 33.2 M | about 2.2 min |
| AVX2 | 60.2 M | about 71 s |

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

Not yet checked (why this is a release candidate):

- No key from this tool has been loaded onto a real radio.
- The Windows build has only been syntax-checked.
- No measurements on the target CPUs (i9-14900HX, Ryzen 9) or on real ARM hardware.

## Security notes

- Keys come from the OS random generator only (`getrandom`, `arc4random_buf`, `BCryptGenRandom`).
- Private keys are written to stdout only, secrets are wiped after use, and core dumps are disabled.
- The search is not constant-time. Run it on a machine you trust.
- A node ID is a 32-bit label, not proof of identity: anyone can generate a key with the same ID. The public key is what identifies a node.
- IDs `!00000000` to `!00000003` and `!ffffffff` are reserved and refused.

## License

MIT.
