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

**Release:** V1.1 (2026-11-21). Prebuilt binaries for Linux and Windows are on the [Releases page](https://github.com/account75783/mtvanity/releases); see [Verifying a release](#verifying-a-release) before running one.

**New in V1.1: vanity public keys.** You can now choose how the public key starts or ends as well as the node ID, ask for both in one key, and search for several keys in one run.

**Tested on a real radio on 2026-11-17, with firmware 2.8** (the latest stable release that day; 2.8.1 was in alpha). A generated key was written to the radio and the node ID changed to the chosen value. The test was done with V1; V1.1 makes the same kind of key.

## Authorship

**Authored by Claude Fable 5.2 (Max, with Research), assisted by a human.**

- Design, code, tests and this document were written by Claude Fable 5.2 (model ID `claude-fable-5-2`), Anthropic's AI model.
- 4+ hours of research and engineering so far.
- A human set the goal, chose the target hardware, corrected the firmware assumptions and directed the work.

## What it does

From firmware 2.8, a Meshtastic node ID is the CRC-32 of the node's 32-byte X25519 public key, shown as `!xxxxxxxx`. `mtvanity` searches for a private key whose public key hashes to the ID you want.

**New in V1.1:** it can also choose the public key itself, the base64 text the Meshtastic apps show for a node. `./mtvanity key:Mesh` finds a key whose public key starts with `Mesh`, `./mtvanity id:c0de+key:Me` finds one key with both a chosen node ID and a chosen public key, and one run can search for several keys at once. The patterns are listed under [Use](#use).

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
| 2 cloud Xeon cores (Sapphire Rapids) | 60 M | about 71 s |

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

### Release binaries

The binaries on the [Releases page](https://github.com/account75783/mtvanity/releases) are built with the lines below. Release builds add `-s -static`, so the program needs no shared libraries.

Linux x86_64:

```
gcc -O3 -s -static -pthread mtvanity.c -o mtvanity-linux-x86_64-V1.1 -lm
```

Windows x86_64, cross-compiled on Debian or WSL (this is how the release `.exe` is built):

```
sudo apt install -y gcc-mingw-w64-x86-64 && x86_64-w64-mingw32-gcc -O3 -s -static -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity-windows-x86_64-V1.1.exe -lbcrypt
```

Windows x86_64, built on Windows itself (MSYS2 UCRT64 shell):

```
gcc -O3 -s -static -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity-windows-x86_64-V1.1.exe -lbcrypt
```

The `-Wa,-muse-unaligned-vector-move` flag is required with MinGW gcc (GCC bug 54412: 32-byte stack alignment on 64-bit Windows).

### Verifying a release

Releases are signed with the maintainer's PGP key (RSA 4096). Fingerprint:

```
F7DB ECD8 5737 C463 7C7A  9EE1 564D EAE8 5007 BB49
```

Download the binary you want and `SHA256SUMS.asc` into one folder.

Linux:

```
curl -s https://github.com/account75783.gpg | gpg --import && gpg -d SHA256SUMS.asc | sed 's#  .*/#  #' | sha256sum -c --ignore-missing
```

You need to see both `Good signature`, with the fingerprint above, and `OK` next to the file name. Then make the download executable: `chmod +x mtvanity-linux-x86_64-V1.1`.

Windows (with Gpg4win installed):

```
curl -s https://github.com/account75783.gpg | gpg --import & gpg -d SHA256SUMS.asc & certutil -hashfile mtvanity-windows-x86_64-V1.1.exe SHA256
```

You need to see `Good signature`, with the fingerprint above, and the hash `certutil` prints must equal the one listed for the `.exe`.

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
mtvanity V1.1 (source revision 4)
searching for 3 targets on 2 threads
  target 1   node ID !dead*                 about 65.5 thousand keys per hit
  target 2   public key Mesh*               about 16.8 million keys per hit
  target 3   node ID !c0* + public key AB*  about 1.05 million keys per hit

(two blocks like the one below, for targets 1 and 3)

==== key 3 of 3 | target 2 | public key Mesh* ====
  node ID:              !91ea5415
  public key (base64):  Mesh/etcr9gYaoGaOuAvvCk414cwAyfRye+Wn0Q601Q=
  public key (hex):     31eb21fdeb5cafd8186a819a3ae02fbc2938d787300327d1c9ef969f443ad354
  private key (base64): ...
  private key (hex):    ...
  install with:         meshtastic --set security.private_key "base64:..."

3 of 3 keys found; 3.92 million keys tried in 0.12 s (31.8 M/s; kernels: avx2 x2)
  target 1   node ID !dead*                 !dead656f
  target 2   public key Mesh*               !91ea5415
  target 3   node ID !c0* + public key AB*  !c0a3c2eb
```

Install the result on the radio (back up the old key first):

```
meshtastic --set security.private_key "base64:..."
```

### What changed since V1

New:

- **Vanity public keys.** `key:` chooses how the base64 public key starts or ends, and `keyhex:` does the same for its hex form.
- **A node ID and a public key in one key**, such as `id:c0de+key:Me`.
- **Several keys in one run.** Up to 64 targets are searched together, each gets its own key, and the run ends with a summary.
- **Mask patterns.** `*` for any run of characters, `.` for any one, and the hashcat-style placeholders `?d ?l ?u ?h ?H ?s ?a`.
- **`-i`** accepts either case in a public key pattern.

Changed:

- **`*` now means "any run of characters".** In V1 it stood for a single digit; write `.` for that now. V1's `x` is refused, and `?` now starts a mask placeholder such as `?d`, so if you reuse an old pattern, check the line `mtvanity` prints before it starts.
- **A list finds a key for every entry.** In V1, `dead,beef` stopped at the first match of either. Now both are found, and `--hits` counts per target.
- **The printed block has a header line**, and JSON output gained `target` and `pattern`.
- **Every search starts by naming its build**, for example `mtvanity V1.1 (source revision 4)`.
- **The status line fits the width of the terminal.** When stderr is a file or a pipe, a plain progress line is written now and then instead.
- **`kill` ends a search the way Ctrl-C does**, with the summary.

Fixed:

- **The Windows console keeps up.** Every message is written in one piece, and each search thread lets anything that is waiting for its CPU run first, a few thousand times a second. Before, with every CPU searching, the status line landed on a new row at each update.
- **A key that cannot be written to stdout stops the search** with an error and exit status 1, instead of being lost without a word.
- **Secrets are handled more tightly.** Two leftover temporaries are wiped, and a private key no longer passes through a C library buffer.
- **Smaller things:** a CRC-32 that was computed for nothing with some combined targets, the thread count on Windows when the CPU mask is unavailable, and leftovers of `^C` on the summary line.

The search kernels are the ones from V1, and the speed is the same.

## Performance

| Machine | Kernel | Keys per second | Average time for a full 8-digit ID |
|---|---|---|---|
| Intel Core i9-14900HX, 32 threads, Linux build under WSL | AVX2 | 516.4 M | about 8.3 s |
| Same laptop, native Windows build | AVX2 | 495.9 M | about 8.7 s |
| Same laptop, native Windows build | Scalar | 275.7 M | about 15.6 s |
| 2 shared cloud cores (Xeon Sapphire Rapids, about 3 GHz, gcc 13.3) | AVX2 | 60.2 M | about 71 s |
| Same 2 cloud cores | Scalar | 33.2 M | about 2.2 min |

Details of the laptop run are in [BENCHMARK-i9-14900HX.md](BENCHMARK-i9-14900HX.md).

These figures were measured with V1. V1.1 uses the same search code and runs at the same speed. Run back to back on the two cloud cores (on a day the machine was a little faster than when the table was made):

- V1 and V1.1 both did 67 to 70 M keys per second for one node ID.
- V1.1 did 72 M for a public key alone, and 63 M with 64 node IDs at once.
- The slow case is a pattern with no fixed character at all, such as one made only of `?l`: about 42 M, because every key has to be looked at in full.

On the laptop under Windows, V1.1 ran at 515.6 M keys per second with `--threads 24`, one thread per core. That matches the best figure in the table, which used all 32 threads.

How it gets there:

- Affine Montgomery walk with a shared denominator and batch inversion: 2.5 multiplications and 1 squaring per key.
- AVX2 kernel running four walks per thread.
- CRC-32 by five carry-less multiplications, no tables.
- Each thread times both kernels on its own core and keeps the faster one.
- One comparison per key: each target is boiled down to the bits every match must share, and a key is looked at closely only if those bits fit. When only public keys are asked for, the CRC-32 is skipped.

## Verification status

Checked:

- 44 self-test checks, including the RFC 7748 vectors, a published Meshtastic key, a full window compared key by key against an independent X25519 ladder, and the pattern matcher compared against a separate reading of the pattern language.
- Every hit is re-derived by the independent reference code before it is printed.
- 1,800 keys for node ID, public key and combined targets were re-derived outside the program by separate Python code (its own X25519, zlib's CRC-32, Python's base64 and a regex reading of each pattern). All matched.
- Clean under ASan, UBSan and TSan (gcc), and under strict warnings with gcc and clang.
- An exact 8-digit ID was found end to end, in the same run as a vanity public key.
- Confirmed by the human on a real radio running firmware 2.8 (2026-11-17), with a V1 key: after a generated key is written, the node ID re-derives to the chosen value.
- The static Linux x86_64 release build passes the self-test and runs at the same speed as the normal build.
- The native Windows x86_64 build (static, MinGW-w64 gcc) passes the full self-test on Windows 11 on the i9-14900HX (V1.1 on 2026-11-21; V1 also ran the benchmark there on 2026-11-18).
- On that laptop under Windows, with a search thread on each of its 32 CPUs, the status line stays in place. Before the search threads gave way to other programs it did not: it landed on a new row at each update unless `--threads` was set below the number of CPUs.

Not yet checked:

- No key chosen by its public key has been written to a radio yet.
- The Windows crash-dump suppression has never been exercised by a real crash.
- The MSYS2 build line for Windows has not been run; the tested `.exe` was cross-compiled.
- No measurements on a Ryzen 9 or on real ARM hardware.

## Security notes

- Keys come from the OS random generator only (`getrandom`, `arc4random_buf`, `BCryptGenRandom`).
- Private keys are written to stdout only, secrets are wiped after use, and core dumps (Linux) and crash dumps (Windows, best effort) are disabled.
- The search is not constant-time. Run it on a machine you trust. On a rented cloud machine, copy the key off and terminate the instance when you are done, so its disk goes with it.
- A node ID is a 32-bit label, not proof of identity: anyone can generate a key with the same ID. The public key is what identifies a node, and only the whole of it: anyone can also generate a key that starts or ends like yours.
- A vanity key is as strong as any other. The search fixes a few public characters and leaves the private key random.
- IDs `!00000000` to `!00000003` and `!ffffffff` are reserved and refused.

## TODO

Planned, with nothing written yet:

1. **Nvidia GPU hardware acceleration.**
2. **AMD GPU hardware acceleration**, after the Nvidia one.

Both are meant to come as optional `.h` header libraries next to `mtvanity.c`, so that the CPU build stays a single C file with no extra dependencies.

## License

MIT. See [LICENSE](LICENSE).

## Support

If mtvanity gave you a fun node ID, tips are welcome in Monero (XMR):

    87MusZ465nXZUb96BfWjgMjDfAqEffuahNjsyefTVhwuF9QQJmw9qDM7auju8vFdpAVJfj8bZZ9SKj5DdQaKKqD1LBnu3pD

This address is signed with the same key as the releases. Verify it with `gpg --verify DONATE.txt.asc`.
