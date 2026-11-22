#!/bin/sh
# Build both release binaries and run their self-tests.
gcc -O3 -s -static -pthread mtvanity.c -o "Linux x64/mtvanity-linux-x86_64-V1.1" -lm && x86_64-w64-mingw32-gcc -O3 -s -static -Wa,-muse-unaligned-vector-move mtvanity.c -o "Windows x64/mtvanity-windows-x86_64-V1.1.exe" -lbcrypt && for f in */mtvanity-*; do echo "$f: $("$f" --version | head -1), $("$f" --self-test | tail -1)"; done
