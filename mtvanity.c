/*
 * mtvanity - vanity node-ID generator for Meshtastic firmware 2.8 and later
 *
 * Authored by Claude, assisted by a human.
 *
 * Since firmware 2.8 a Meshtastic node number is the CRC-32 of the node's
 * 32-byte X25519 public key, displayed as "!%08x".  This program searches
 * for a private key whose node ID matches a pattern you choose.  (Older
 * firmware takes the number from the MAC address; a key changes nothing
 * there.)
 *
 * BUILD
 *   Debian:   gcc -O3 -pthread mtvanity.c -o mtvanity -lm
 *   Windows:  gcc -O3 -Wa,-muse-unaligned-vector-move mtvanity.c -o mtvanity.exe -lbcrypt
 *             (MSYS2 UCRT64.  The -Wa option works around GCC bug 54412:
 *             on 64-bit Windows gcc aligns the stack to 16 bytes yet moves
 *             32-byte vectors as if they were aligned.  clang needs no
 *             such option.)
 *   Android:  clang -O3 -march=armv8-a+crc -pthread mtvanity.c -o mtvanity -lm
 *   The x86 kernels are selected at run time, so one binary serves every
 *   x86-64 CPU and -march=native is not needed.
 *
 * USE
 *   ./mtvanity --self-test       check every component against a reference
 *   ./mtvanity --benchmark       keys per second of each kernel on this machine
 *   ./mtvanity c0de              a key whose node ID begins !c0de
 *   ./mtvanity --help            patterns and options
 *
 * HOW IT IS FAST
 *   1. Scalar walk.  Candidates are the clamped scalars s, s+8, s+16, ...
 *      so consecutive public keys differ by the fixed point S = 8B and
 *      nothing is ever multiplied from scratch.
 *   2. Shared denominators.  In affine Montgomery coordinates the two keys
 *      M + iS and M - iS around a window centre M share the denominator
 *      (u_i - u_M), so one inverse yields both public keys:
 *          u(M +/- iS) = ((v_i -/+ v_M) / (u_i - u_M))^2 - A - u_M - u_i
 *      and the result is already the affine u-coordinate X25519 publishes.
 *   3. Batched inversion.  Montgomery's trick turns the W inverses of a
 *      window into three multiplications each plus one real inversion.
 *      Net cost: 2.5 multiplications and 1 squaring per key.
 *   4. No separate reduction.  The final addition and the canonical
 *      encoding ride on the last squaring's carry chain; the rare value
 *      that needs more is detected and finished exactly.
 *   5. CRC-32 without tables.  Five carry-less multiplications reduce a
 *      32-byte message directly (PCLMULQDQ, two messages at a time with
 *      VPCLMULQDQ); ARMv8 uses its CRC32 instructions.
 *   6. AVX2.  Four independent walks run in lockstep, one per 64-bit lane,
 *      on radix-2^25.5 field elements.
 *   7. Per-core choice.  Each thread times the kernels on the core it is
 *      bound to and keeps the faster one, which matters on CPUs that mix
 *      performance and efficiency cores.
 *
 * SECURITY
 *   - Keys come only from the operating system's CSPRNG.
 *   - Every hit is recomputed by an independent implementation (a plain
 *     RFC 7748 ladder on 16-bit limbs and a bitwise CRC) before it is
 *     shown.  A fault in the fast path can therefore lose a key but can
 *     never print a wrong one.  The same reference checks the fast path
 *     every time the program starts.
 *   - A base scalar is retired after its first hit, so two keys this
 *     program prints are never a small known offset apart.
 *   - Private keys are written to stdout only, never to disk or logs, and
 *     core dumps are disabled.
 *   - The search is deliberately NOT constant-time: it is an offline tool.
 *     Do not run it on a machine you share with people you do not trust.
 *   - A vanity key is an ordinary random key with one chosen public
 *     property.  Pick an ID nobody on your mesh already uses.
 *
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 the mtvanity authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
 * THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#  include <bcrypt.h>
#  include <malloc.h>
#else
#  include <pthread.h>
#  include <sys/resource.h>
#  include <time.h>
#  include <unistd.h>
#  ifdef __linux__
#    include <sched.h>
#    ifndef __ANDROID__
#      include <sys/random.h>
#    endif
#  endif
#endif

#ifndef __SIZEOF_INT128__
#  error "mtvanity needs a 64-bit gcc or clang (it uses unsigned __int128)"
#endif

#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
#  pragma message("mtvanity: gcc on Windows needs -Wa,-muse-unaligned-vector-move (see BUILD)")
#endif

/* -DMTV_PORTABLE_ONLY builds without any x86-specific code, for comparison. */
#if defined(__x86_64__) && !defined(MTV_PORTABLE_ONLY)
#  include <cpuid.h>
#  include <immintrin.h>
#  define MTV_X86 1
#endif
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#  include <arm_acle.h>
#  define MTV_ARM_CRC 1
#endif

#define MTV_VERSION     "2.0.0"
#define MTV_ATTRIBUTION "Authored by Claude, assisted by a human."

/* A window is the 2W+1 keys centred on one point; W is also the table size. */
#define HALF_WINDOW 512
#define WINDOW      (2 * HALF_WINDOW + 1)

#define MAX_THREADS 1024
#define MAX_TARGETS 64
#define MAX_LANES   4

#define LIKELY(x)     __builtin_expect(!!(x), 1)
#define UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define NOINLINE      __attribute__((noinline))
#define ARRAY_LEN(a)  (sizeof(a) / sizeof((a)[0]))

__extension__ typedef unsigned __int128 u128;

/* ======================================================================== */
/* 1. Small utilities                                                        */
/* ======================================================================== */

static void die(const char *msg)
{
    fprintf(stderr, "mtvanity: %s\n", msg);
    exit(2);
}

/* Wipe memory in a way the optimiser may not remove. */
static void secure_zero(void *p, size_t n)
{
    volatile unsigned char *v = (volatile unsigned char *)p;
    while (n--)
        *v++ = 0;
}

static uint64_t load64_le(const uint8_t *p)
{
    uint64_t x = 0;
    for (int i = 7; i >= 0; i--)
        x = (x << 8) | p[i];
    return x;
}

static void store64_le(uint8_t *p, uint64_t x)
{
    for (int i = 0; i < 8; i++) {
        p[i] = (uint8_t)x;
        x >>= 8;
    }
}

static void words_to_bytes(uint8_t out[32], const uint64_t w[4])
{
    for (int k = 0; k < 4; k++)
        store64_le(out + 8 * k, w[k]);
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Decode exactly n bytes of hex; only used on literals inside this file. */
static void hex_decode(uint8_t *out, const char *hex, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (uint8_t)((hex_digit(hex[2 * i]) << 4) | hex_digit(hex[2 * i + 1]));
}

static void hex_encode(char *out, const uint8_t *in, size_t n)
{
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = digits[in[i] >> 4];
        out[2 * i + 1] = digits[in[i] & 15];
    }
    out[2 * n] = '\0';
}

/* Standard base64 with padding; out needs 4 * ceil(n / 3) + 1 bytes. */
static void base64_encode(char *out, const uint8_t *in, size_t n)
{
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t b0 = in[i];
        const uint32_t b1 = i + 1 < n ? in[i + 1] : 0;
        const uint32_t b2 = i + 2 < n ? in[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = i + 1 < n ? tab[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < n ? tab[v & 63] : '=';
    }
    out[o] = '\0';
}

/* ======================================================================== */
/* 2. Operating system layer                                                 */
/* ======================================================================== */

/* Fill buf from the OS CSPRNG; there is no fallback, failure is fatal. */
static void os_random(void *buf, size_t len)
{
#if defined(_WIN32)
    if (BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        die("the system random number generator failed");
#elif defined(__linux__) && !defined(__ANDROID__)
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        const ssize_t r = getrandom(p, len, 0);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            die("getrandom() failed");
        }
        p += r;
        len -= (size_t)r;
    }
#else
    arc4random_buf(buf, len);
#endif
}

static double now_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void sleep_ms(unsigned ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000);
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

/* Logical CPUs this process may run on. */
static int os_cpu_count(void)
{
#if defined(_WIN32)
    DWORD_PTR process_mask, system_mask;
    int n = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask))
        for (; process_mask != 0; process_mask >>= 1)
            n += (int)(process_mask & 1);
    return n > 0 ? n : 1;
#else
#  if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0)
        return CPU_COUNT(&set);
#  endif
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/*
 * Bind the calling thread to the slot-th CPU the process may use, so that a
 * thread which has measured its core keeps running on it.  Best effort: on
 * other systems, or on failure, the thread simply stays unbound.
 */
static void os_pin_thread(int slot)
{
#if defined(_WIN32)
    DWORD_PTR process_mask, system_mask;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask))
        return;
    for (unsigned bit = 0; bit < 8 * sizeof process_mask; bit++) {
        if (((process_mask >> bit) & 1) && slot-- == 0) {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << bit);
            return;
        }
    }
#elif defined(__linux__)
    cpu_set_t allowed, one;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0)
        return;
    for (size_t cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &allowed) && slot-- == 0) {
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            (void)sched_setaffinity(0, sizeof one, &one);
            return;
        }
    }
#else
    (void)slot;
#endif
}

/* A crash must not leave private keys in a core file. */
static void os_disable_core_dumps(void)
{
#ifndef _WIN32
    const struct rlimit none = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &none);
#endif
}

static void *xalloc_aligned(size_t size)
{
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(size, 64);
#else
    if (posix_memalign(&p, 64, size) != 0)
        p = NULL;
#endif
    if (!p)
        die("out of memory");
    return p;
}

static void xfree_aligned(void *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ---- threads and one mutex ---- */

typedef struct {
    void (*fn)(void *);
    void *arg;
#ifdef _WIN32
    HANDLE handle;
#else
    pthread_t handle;
#endif
} thread_t;

#ifdef _WIN32
static DWORD WINAPI thread_entry(LPVOID p)
{
    thread_t *t = (thread_t *)p;
    t->fn(t->arg);
    return 0;
}
static int thread_start(thread_t *t)
{
    t->handle = CreateThread(NULL, 0, thread_entry, t, 0, NULL);
    return t->handle ? 0 : -1;
}
static void thread_join(thread_t *t)
{
    WaitForSingleObject(t->handle, INFINITE);
    CloseHandle(t->handle);
}
static CRITICAL_SECTION g_output_lock;
static void output_lock_init(void) { InitializeCriticalSection(&g_output_lock); }
static void output_lock(void) { EnterCriticalSection(&g_output_lock); }
static void output_unlock(void) { LeaveCriticalSection(&g_output_lock); }
#else
static void *thread_entry(void *p)
{
    thread_t *t = (thread_t *)p;
    t->fn(t->arg);
    return NULL;
}
static int thread_start(thread_t *t)
{
    return pthread_create(&t->handle, NULL, thread_entry, t) == 0 ? 0 : -1;
}
static void thread_join(thread_t *t)
{
    pthread_join(t->handle, NULL);
}
static pthread_mutex_t g_output_lock = PTHREAD_MUTEX_INITIALIZER;
static void output_lock_init(void) {}
static void output_lock(void) { pthread_mutex_lock(&g_output_lock); }
static void output_unlock(void) { pthread_mutex_unlock(&g_output_lock); }
#endif

/* ---- CPU features (x86-64 only; everything else uses the portable path) ---- */

#ifdef MTV_X86
static struct {
    int pclmul;     /* PCLMULQDQ (and SSE4.1)          */
    int avx2;       /* AVX2, with OS support for YMM   */
    int vpclmul;    /* VPCLMULQDQ on YMM               */
} g_cpu;

static void cpu_detect(void)
{
    unsigned a, b, c, d;
    int os_ymm = 0, avx = 0;

    if (__get_cpuid(1, &a, &b, &c, &d)) {
        g_cpu.pclmul = ((c >> 1) & 1) && ((c >> 19) & 1);     /* the code also uses SSE4.1 */
        avx = (c >> 28) & 1;
        if ((c >> 27) & 1) {                        /* OSXSAVE: XGETBV is usable */
            unsigned lo, hi;
            __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            (void)hi;
            os_ymm = (lo & 6) == 6;                 /* XMM and YMM state enabled */
        }
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        g_cpu.avx2 = os_ymm && avx && ((b >> 5) & 1);
        g_cpu.vpclmul = g_cpu.avx2 && g_cpu.pclmul && ((c >> 10) & 1);
    }
}
#endif /* MTV_X86 */

/* ======================================================================== */
/* 3. Reference implementation                                               */
/*                                                                           */
/* Deliberately simple and deliberately separate: it shares no arithmetic    */
/* with the search path.  Every key this program prints has passed through   */
/* it, and the self-tests measure the fast code against it.  The ladder      */
/* follows the public-domain TweetNaCl design (sixteen 16-bit limbs).        */
/* ======================================================================== */

typedef int64_t gf[16];

_Static_assert((-1 >> 1) == -1, "the reference ladder needs arithmetic right shifts");

static const gf GF_121665 = {0xDB41, 1};

static void gf_carry(gf o)
{
    for (int i = 0; i < 16; i++) {
        o[i] += (int64_t)1 << 16;
        const int64_t c = o[i] >> 16;
        o[(i + 1) * (i < 15)] += c - 1 + 37 * (c - 1) * (i == 15);
        o[i] -= c * 65536;
    }
}

static void gf_cswap(gf p, gf q, int b)
{
    const int64_t mask = ~((int64_t)b - 1);
    for (int i = 0; i < 16; i++) {
        const int64_t t = mask & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

static void gf_pack(uint8_t o[32], const gf n)
{
    gf m, t;
    for (int i = 0; i < 16; i++)
        t[i] = n[i];
    gf_carry(t);
    gf_carry(t);
    gf_carry(t);
    for (int j = 0; j < 2; j++) {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; i++) {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        const int borrow = (int)((m[15] >> 16) & 1);
        m[14] &= 0xffff;
        gf_cswap(t, m, 1 - borrow);
    }
    for (int i = 0; i < 16; i++) {
        o[2 * i] = (uint8_t)(t[i] & 0xff);
        o[2 * i + 1] = (uint8_t)(t[i] >> 8);
    }
}

static void gf_unpack(gf o, const uint8_t n[32])
{
    for (int i = 0; i < 16; i++)
        o[i] = n[2 * i] + ((int64_t)n[2 * i + 1] << 8);
    o[15] &= 0x7fff;
}

static void gf_add(gf o, const gf a, const gf b)
{
    for (int i = 0; i < 16; i++)
        o[i] = a[i] + b[i];
}

static void gf_sub(gf o, const gf a, const gf b)
{
    for (int i = 0; i < 16; i++)
        o[i] = a[i] - b[i];
}

static void gf_mul(gf o, const gf a, const gf b)
{
    int64_t t[31];
    for (int i = 0; i < 31; i++)
        t[i] = 0;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            t[i + j] += a[i] * b[j];
    for (int i = 0; i < 15; i++)
        t[i] += 38 * t[i + 16];
    for (int i = 0; i < 16; i++)
        o[i] = t[i];
    gf_carry(o);
    gf_carry(o);
}

static void gf_invert(gf o, const gf in)
{
    gf c;
    for (int i = 0; i < 16; i++)
        c[i] = in[i];
    for (int a = 253; a >= 0; a--) {
        gf_mul(c, c, c);
        if (a != 2 && a != 4)
            gf_mul(c, c, in);
    }
    for (int i = 0; i < 16; i++)
        o[i] = c[i];
}

/* q = X25519(n, p) exactly as RFC 7748 section 5 defines it. */
static void x25519_ref(uint8_t q[32], const uint8_t n[32], const uint8_t p[32])
{
    uint8_t z[32];
    gf x, a, b, c, d, e, f;

    memcpy(z, n, 32);
    z[0] &= 248;
    z[31] = (uint8_t)((z[31] & 127) | 64);
    gf_unpack(x, p);
    for (int i = 0; i < 16; i++) {
        b[i] = x[i];
        d[i] = a[i] = c[i] = 0;
    }
    a[0] = d[0] = 1;
    for (int i = 254; i >= 0; i--) {
        const int r = (z[i >> 3] >> (i & 7)) & 1;
        gf_cswap(a, b, r);
        gf_cswap(c, d, r);
        gf_add(e, a, c);
        gf_sub(a, a, c);
        gf_add(c, b, d);
        gf_sub(b, b, d);
        gf_mul(d, e, e);
        gf_mul(f, a, a);
        gf_mul(a, c, a);
        gf_mul(c, b, e);
        gf_add(e, a, c);
        gf_sub(a, a, c);
        gf_mul(b, a, a);
        gf_sub(c, d, f);
        gf_mul(a, c, GF_121665);
        gf_add(a, a, d);
        gf_mul(c, c, a);
        gf_mul(a, d, f);
        gf_mul(d, b, x);
        gf_mul(b, e, e);
        gf_cswap(a, b, r);
        gf_cswap(c, d, r);
    }
    gf_invert(c, c);
    gf_mul(a, a, c);
    gf_pack(q, a);
    secure_zero(z, sizeof z);
    secure_zero(a, sizeof(gf));
    secure_zero(b, sizeof(gf));
    secure_zero(c, sizeof(gf));
    secure_zero(d, sizeof(gf));
}

/* CRC-32 as used by Ethernet and zlib: reflected, init and final XOR all ones. */
static uint32_t crc32_ref(const uint8_t *p, size_t n)
{
    uint32_t c = 0xffffffffu;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return ~c;
}

/* The X25519 base point u = 9. */
static const uint8_t BASE_U[32] = {9};

/* ======================================================================== */
/* 4. Fast CRC-32 of one 32-byte message                                     */
/*                                                                           */
/* A message is passed as four 64-bit words, w[0] holding bytes 0..7 in      */
/* little-endian order, which is how the field code produces a public key.   */
/* ======================================================================== */

typedef void (*crc_fn)(const uint64_t (*msg)[4], uint32_t *out, size_t n);

/* ---- portable: slicing-by-8 ---- */

static uint32_t CRC_TAB[8][256];

static void crc_table_init(void)
{
    for (uint32_t b = 0; b < 256; b++) {
        uint32_t c = b;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
        CRC_TAB[0][b] = c;
    }
    for (int t = 1; t < 8; t++)
        for (uint32_t b = 0; b < 256; b++)
            CRC_TAB[t][b] = (CRC_TAB[t - 1][b] >> 8) ^ CRC_TAB[0][CRC_TAB[t - 1][b] & 0xff];
}

static void crc_msgs_table(const uint64_t (*msg)[4], uint32_t *out, size_t n)
{
    for (size_t m = 0; m < n; m++) {
        uint32_t c = 0xffffffffu;
        for (int k = 0; k < 4; k++) {
            const uint64_t x = msg[m][k] ^ c;
            c = CRC_TAB[7][x & 0xff] ^ CRC_TAB[6][(x >> 8) & 0xff] ^
                CRC_TAB[5][(x >> 16) & 0xff] ^ CRC_TAB[4][(x >> 24) & 0xff] ^
                CRC_TAB[3][(x >> 32) & 0xff] ^ CRC_TAB[2][(x >> 40) & 0xff] ^
                CRC_TAB[1][(x >> 48) & 0xff] ^ CRC_TAB[0][x >> 56];
        }
        out[m] = ~c;
    }
}

#ifdef MTV_ARM_CRC
static void crc_msgs_armv8(const uint64_t (*msg)[4], uint32_t *out, size_t n)
{
    for (size_t m = 0; m < n; m++) {
        uint32_t c = 0xffffffffu;
        c = __crc32d(c, msg[m][0]);
        c = __crc32d(c, msg[m][1]);
        c = __crc32d(c, msg[m][2]);
        c = __crc32d(c, msg[m][3]);
        out[m] = ~c;
    }
}
#endif

/*
 * ---- carry-less multiplication ----
 *
 * Write the message as a polynomial M(x) over GF(2), first bit = highest
 * degree, and split it into four 64-bit words W0..W3.  Leaving the all-ones
 * init and final XOR aside (they only add the constant CRC_ZERO32, because a
 * CRC is affine), the CRC is
 *
 *     M * x^32 mod P  =  W0*K0 + W1*K1 + W2*K2 + W3*x^32  (mod P),
 *     Kj = x^(224 - 64j) mod P.
 *
 * The right-hand side T has degree < 96.  One Barrett step reduces it: with
 * T = A*x^32 + B and mu = floor(x^96 / P) = x^64 + mu_lo,
 *
 *     q = floor(T / P) = A + floor(A * mu_lo / x^64),    CRC = B + (q*P mod x^32).
 *
 * That is five carry-less multiplications.  In the bit-reflected register
 * layout a 64x64 product lands one bit low, which is why the constants are
 * stored shifted left by one.  All constants are derived at start-up from
 * the polynomial and the self-test compares every engine with crc32_ref.
 */
static uint64_t CRC_K0, CRC_K1, CRC_K2, CRC_MU, CRC_P1;
static uint32_t CRC_ZERO32;                 /* CRC-32 of thirty-two zero bytes */

#define CRC_POLY 0x104c11db7ULL             /* x^32 + ... + 1, bit i = x^i     */

static uint64_t gf2_xpow_mod_p(unsigned e)  /* x^e mod P */
{
    uint64_t r = 1;
    while (e--) {
        r <<= 1;
        if (r >> 32)
            r ^= CRC_POLY;
    }
    return r;
}

static uint64_t gf2_x96_div_p(void)         /* low 64 bits of floor(x^96 / P) */
{
    uint64_t q = 0, r = 0;
    for (int bit = 96; bit >= 0; bit--) {
        r = (r << 1) | (bit == 96);
        q <<= 1;                            /* the quotient's x^64 term falls off */
        if (r >> 32) {
            r ^= CRC_POLY;
            q |= 1;
        }
    }
    return q;
}

static uint64_t bit_reverse(uint64_t x, int width)
{
    uint64_t r = 0;
    for (int i = 0; i < width; i++)
        r |= ((x >> i) & 1) << (width - 1 - i);
    return r;
}

static void crc_clmul_init(void)
{
    static const uint8_t zero[32] = {0};
    CRC_K0 = bit_reverse(gf2_xpow_mod_p(224), 32) << 1;
    CRC_K1 = bit_reverse(gf2_xpow_mod_p(160), 32) << 1;
    CRC_K2 = bit_reverse(gf2_xpow_mod_p(96), 32) << 1;
    CRC_MU = bit_reverse(gf2_x96_div_p(), 64);
    CRC_P1 = (uint64_t)0xedb88320u << 1;
    CRC_ZERO32 = crc32_ref(zero, 32);
}

#ifdef MTV_X86
/*
 * The reduction above on one 128-bit register (or on both halves of a
 * 256-bit one): X01 = [W0, W1], X23 = [W2, W3], KA = [K0, K1],
 * KB = [K2, mu_lo], KC = [P, 0].  Afterwards 32-bit element 2 of T is the
 * CRC, still without the constant CRC_ZERO32.
 */
#define CRC_REDUCE(T, X01, X23, KA, KB, KC, CLMUL, XOR, SHR8, SHL1)                 \
    do {                                                                            \
        T = XOR(CLMUL(X01, KA, 0x00), CLMUL(X01, KA, 0x11));    /* W0*K0 + W1*K1 */ \
        T = XOR(T, CLMUL(X23, KB, 0x00));                       /* + W2*K2       */ \
        T = XOR(T, SHR8(X23));                                  /* + W3*x^32     */ \
        /* q = A + floor(A * mu_lo / x^64), then T + q*P */                         \
        T = XOR(T, CLMUL(XOR(T, SHL1(CLMUL(T, KB, 0x10))), KC, 0x00));              \
    } while (0)

#define CRC_X_SHR8(x) _mm_srli_si128((x), 8)
#define CRC_X_SHL1(x) _mm_slli_epi64((x), 1)
#define CRC_REDUCE_X(T, X01, X23, KA, KB, KC)                                       \
    CRC_REDUCE(T, X01, X23, KA, KB, KC, _mm_clmulepi64_si128, _mm_xor_si128,        \
               CRC_X_SHR8, CRC_X_SHL1)

__attribute__((target("sse4.1,pclmul")))
static void crc_msgs_pclmul(const uint64_t (*msg)[4], uint32_t *out, size_t n)
{
    const __m128i ka = _mm_set_epi64x((long long)CRC_K1, (long long)CRC_K0);
    const __m128i kb = _mm_set_epi64x((long long)CRC_MU, (long long)CRC_K2);
    const __m128i kc = _mm_set_epi64x(0, (long long)CRC_P1);

    for (size_t m = 0; m < n; m++) {
        const __m128i x01 = _mm_loadu_si128((const __m128i *)(const void *)&msg[m][0]);
        const __m128i x23 = _mm_loadu_si128((const __m128i *)(const void *)&msg[m][2]);
        __m128i t;
        CRC_REDUCE_X(t, x01, x23, ka, kb, kc);
        out[m] = (uint32_t)_mm_extract_epi32(t, 2) ^ CRC_ZERO32;
    }
}
#endif

/* The engine the scalar search kernel uses; chosen once at start-up. */
static crc_fn crc_msgs = crc_msgs_table;

static void crc_init(void)
{
    crc_table_init();
    crc_clmul_init();
#if defined(MTV_ARM_CRC)
    crc_msgs = crc_msgs_armv8;
#elif defined(MTV_X86)
    if (g_cpu.pclmul)
        crc_msgs = crc_msgs_pclmul;
#endif
}

/* ======================================================================== */
/* 5. Field arithmetic modulo p = 2^255 - 19, five 51-bit limbs              */
/*                                                                           */
/* Bounds, written against each function:                                    */
/*   canonical  the unique representative: value < p, every limb < 2^51      */
/*   tight      every limb < 2^51 + 2^16                                     */
/*   loose      every limb < 2^54                                            */
/* Multiplication and squaring accept loose inputs and return tight ones.    */
/* ======================================================================== */

typedef struct {
    uint64_t v[5];
} fe;

#define MASK51 ((uint64_t)0x7ffffffffffff)

static const fe FE_ONE = {{1, 0, 0, 0, 0}};

/* loose -> tight */
static inline void fe_carry(fe *r)
{
    uint64_t c;
    c = r->v[0] >> 51; r->v[0] &= MASK51; r->v[1] += c;
    c = r->v[1] >> 51; r->v[1] &= MASK51; r->v[2] += c;
    c = r->v[2] >> 51; r->v[2] &= MASK51; r->v[3] += c;
    c = r->v[3] >> 51; r->v[3] &= MASK51; r->v[4] += c;
    c = r->v[4] >> 51; r->v[4] &= MASK51; r->v[0] += 19 * c;
}

/* tight + tight -> loose */
static inline void fe_add_lazy(fe *r, const fe *a, const fe *b)
{
    for (int i = 0; i < 5; i++)
        r->v[i] = a->v[i] + b->v[i];
}

/* tight - tight -> loose: computes a + 2p - b, whose limbs cannot go negative */
static inline void fe_sub_lazy(fe *r, const fe *a, const fe *b)
{
    r->v[0] = a->v[0] + 0xfffffffffffdaULL - b->v[0];
    r->v[1] = a->v[1] + 0xffffffffffffeULL - b->v[1];
    r->v[2] = a->v[2] + 0xffffffffffffeULL - b->v[2];
    r->v[3] = a->v[3] + 0xffffffffffffeULL - b->v[3];
    r->v[4] = a->v[4] + 0xffffffffffffeULL - b->v[4];
}

/* tight (+,-) tight -> tight */
static inline void fe_add(fe *r, const fe *a, const fe *b)
{
    fe_add_lazy(r, a, b);
    fe_carry(r);
}

static inline void fe_sub(fe *r, const fe *a, const fe *b)
{
    fe_sub_lazy(r, a, b);
    fe_carry(r);
}

/* Shared tail of fe_mul and fe_sq: fold five 128-bit columns into limbs. */
static ALWAYS_INLINE void fe_fold(fe *r, u128 t0, u128 t1, u128 t2, u128 t3, u128 t4)
{
    uint64_t r0, r1, r2, r3, r4, c;
    t1 += (uint64_t)(t0 >> 51); r0 = (uint64_t)t0 & MASK51;
    t2 += (uint64_t)(t1 >> 51); r1 = (uint64_t)t1 & MASK51;
    t3 += (uint64_t)(t2 >> 51); r2 = (uint64_t)t2 & MASK51;
    t4 += (uint64_t)(t3 >> 51); r3 = (uint64_t)t3 & MASK51;
    c = (uint64_t)(t4 >> 51);   r4 = (uint64_t)t4 & MASK51;
    r0 += 19 * c;               /* c < 2^60 for loose inputs, so this fits */
    r1 += r0 >> 51;
    r0 &= MASK51;
    r->v[0] = r0; r->v[1] = r1; r->v[2] = r2; r->v[3] = r3; r->v[4] = r4;
}

static ALWAYS_INLINE void fe_mul(fe *r, const fe *a, const fe *b)
{
    const uint64_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4];
    const uint64_t b0 = b->v[0], b1 = b->v[1], b2 = b->v[2], b3 = b->v[3], b4 = b->v[4];
    const uint64_t b1_19 = 19 * b1, b2_19 = 19 * b2, b3_19 = 19 * b3, b4_19 = 19 * b4;

#define P(x, y) ((u128)(x) * (y))
    const u128 t0 = P(a0, b0) + P(a1, b4_19) + P(a2, b3_19) + P(a3, b2_19) + P(a4, b1_19);
    const u128 t1 = P(a0, b1) + P(a1, b0) + P(a2, b4_19) + P(a3, b3_19) + P(a4, b2_19);
    const u128 t2 = P(a0, b2) + P(a1, b1) + P(a2, b0) + P(a3, b4_19) + P(a4, b3_19);
    const u128 t3 = P(a0, b3) + P(a1, b2) + P(a2, b1) + P(a3, b0) + P(a4, b4_19);
    const u128 t4 = P(a0, b4) + P(a1, b3) + P(a2, b2) + P(a3, b1) + P(a4, b0);
#undef P
    fe_fold(r, t0, t1, t2, t3, t4);
}

static ALWAYS_INLINE void fe_sq(fe *r, const fe *a)
{
    const uint64_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4];
    const uint64_t a0_2 = 2 * a0, a1_2 = 2 * a1;
    const uint64_t a1_38 = 38 * a1, a2_38 = 38 * a2, a3_38 = 38 * a3;
    const uint64_t a3_19 = 19 * a3, a4_19 = 19 * a4;

    const u128 t0 = (u128)a0 * a0 + (u128)a1_38 * a4 + (u128)a2_38 * a3;
    const u128 t1 = (u128)a0_2 * a1 + (u128)a2_38 * a4 + (u128)a3_19 * a3;
    const u128 t2 = (u128)a0_2 * a2 + (u128)a1 * a1 + (u128)a3_38 * a4;
    const u128 t3 = (u128)a0_2 * a3 + (u128)a1_2 * a2 + (u128)a4_19 * a4;
    const u128 t4 = (u128)a0_2 * a4 + (u128)a1_2 * a3 + (u128)a2 * a2;
    fe_fold(r, t0, t1, t2, t3, t4);
}

static void fe_sq_times(fe *r, const fe *a, int n)
{
    *r = *a;
    while (n-- > 0)
        fe_sq(r, r);
}

/* r = z^(p-2) = 1/z (and 0 for z = 0): 254 squarings and 11 multiplications */
static NOINLINE void fe_invert(fe *r, const fe *z)
{
    fe z2, z9, z11, z_5_0, z_10_0, z_20_0, z_50_0, z_100_0, t;

    fe_sq(&z2, z);                                          /* 2 */
    fe_sq_times(&t, &z2, 2);                                /* 8 */
    fe_mul(&z9, &t, z);                                     /* 9 */
    fe_mul(&z11, &z9, &z2);                                 /* 11 */
    fe_sq(&t, &z11);                                        /* 22 */
    fe_mul(&z_5_0, &t, &z9);                                /* 2^5 - 1 */
    fe_sq_times(&t, &z_5_0, 5);   fe_mul(&z_10_0, &t, &z_5_0);     /* 2^10 - 1 */
    fe_sq_times(&t, &z_10_0, 10); fe_mul(&z_20_0, &t, &z_10_0);    /* 2^20 - 1 */
    fe_sq_times(&t, &z_20_0, 20); fe_mul(&t, &t, &z_20_0);         /* 2^40 - 1 */
    fe_sq_times(&t, &t, 10);      fe_mul(&z_50_0, &t, &z_10_0);    /* 2^50 - 1 */
    fe_sq_times(&t, &z_50_0, 50); fe_mul(&z_100_0, &t, &z_50_0);   /* 2^100 - 1 */
    fe_sq_times(&t, &z_100_0, 100); fe_mul(&t, &t, &z_100_0);      /* 2^200 - 1 */
    fe_sq_times(&t, &t, 50);      fe_mul(&t, &t, &z_50_0);         /* 2^250 - 1 */
    fe_sq_times(&t, &t, 5);       fe_mul(r, &t, &z11);             /* 2^255 - 21 */
}

/* Canonical value of a (any limbs < 2^60) as four little-endian words. */
static ALWAYS_INLINE void fe_to_words(uint64_t w[4], const fe *a)
{
    fe t = *a;
    uint64_t q;

    fe_carry(&t);
    fe_carry(&t);                       /* t < 2^255 + 19, limbs 1..4 < 2^51 */
    q = (t.v[0] + 19) >> 51;
    q = (t.v[1] + q) >> 51;
    q = (t.v[2] + q) >> 51;
    q = (t.v[3] + q) >> 51;
    q = (t.v[4] + q) >> 51;             /* q = 1 exactly when t >= p */
    t.v[0] += 19 * q;
    t.v[1] += t.v[0] >> 51; t.v[0] &= MASK51;
    t.v[2] += t.v[1] >> 51; t.v[1] &= MASK51;
    t.v[3] += t.v[2] >> 51; t.v[2] &= MASK51;
    t.v[4] += t.v[3] >> 51; t.v[3] &= MASK51;
    t.v[4] &= MASK51;                   /* discards q * 2^255 */

    w[0] = t.v[0] | (t.v[1] << 51);
    w[1] = (t.v[1] >> 13) | (t.v[2] << 38);
    w[2] = (t.v[2] >> 26) | (t.v[3] << 25);
    w[3] = (t.v[3] >> 39) | (t.v[4] << 12);
}

/*
 * w = canonical words of a^2 + e, the last step of every key: a tight,
 * e loose.
 *
 * Folding the addition into the square's own carry chain leaves limbs 1, 3
 * and 4 exact, limb 0 below 2^51 and limb 2 at most 2^51.  Such a value is
 * already canonical unless limb 2 overflowed or limb 0 is within 19 of 2^51;
 * both are rare enough (about 2^-46) to hand to the general routine.
 */
static ALWAYS_INLINE void fe_sq_add_to_words(uint64_t w[4], const fe *a, const fe *e)
{
    const uint64_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4];
    const uint64_t a0_2 = 2 * a0, a1_2 = 2 * a1;
    const uint64_t a1_38 = 38 * a1, a2_38 = 38 * a2, a3_38 = 38 * a3;
    const uint64_t a3_19 = 19 * a3, a4_19 = 19 * a4;
    u128 t0 = (u128)a0 * a0 + (u128)a1_38 * a4 + (u128)a2_38 * a3 + e->v[0];
    u128 t1 = (u128)a0_2 * a1 + (u128)a2_38 * a4 + (u128)a3_19 * a3 + e->v[1];
    u128 t2 = (u128)a0_2 * a2 + (u128)a1 * a1 + (u128)a3_38 * a4 + e->v[2];
    u128 t3 = (u128)a0_2 * a3 + (u128)a1_2 * a2 + (u128)a4_19 * a4 + e->v[3];
    u128 t4 = (u128)a0_2 * a4 + (u128)a1_2 * a3 + (u128)a2 * a2 + e->v[4];
    uint64_t r0, r1, r2, r3, r4, c;

    t1 += (uint64_t)(t0 >> 51); r0 = (uint64_t)t0 & MASK51;
    t2 += (uint64_t)(t1 >> 51); r1 = (uint64_t)t1 & MASK51;
    t3 += (uint64_t)(t2 >> 51); r2 = (uint64_t)t2 & MASK51;
    t4 += (uint64_t)(t3 >> 51); r3 = (uint64_t)t3 & MASK51;
    c = (uint64_t)(t4 >> 51);   r4 = (uint64_t)t4 & MASK51;
    r0 += 19 * c;
    r1 += r0 >> 51; r0 &= MASK51;
    r2 += r1 >> 51; r1 &= MASK51;

    if (UNLIKELY((r2 >> 51) != 0 || r0 >= MASK51 - 18)) {
        const fe t = {{r0, r1, r2, r3, r4}};
        fe_to_words(w, &t);
        return;
    }
    w[0] = r0 | (r1 << 51);
    w[1] = (r1 >> 13) | (r2 << 38);
    w[2] = (r2 >> 26) | (r3 << 25);
    w[3] = (r3 >> 39) | (r4 << 12);
}

/* Bit 255 of the input is ignored, as RFC 7748 requires. */
static void fe_from_words(fe *r, const uint64_t w[4])
{
    r->v[0] = w[0] & MASK51;
    r->v[1] = ((w[0] >> 51) | (w[1] << 13)) & MASK51;
    r->v[2] = ((w[1] >> 38) | (w[2] << 26)) & MASK51;
    r->v[3] = ((w[2] >> 25) | (w[3] << 39)) & MASK51;
    r->v[4] = (w[3] >> 12) & MASK51;
}

static void fe_from_bytes(fe *r, const uint8_t s[32])
{
    const uint64_t w[4] = {load64_le(s), load64_le(s + 8), load64_le(s + 16), load64_le(s + 24)};
    fe_from_words(r, w);
}

static void fe_to_bytes(uint8_t s[32], const fe *a)
{
    uint64_t w[4];
    fe_to_words(w, a);
    words_to_bytes(s, w);
}

/* Replace a by its canonical representative. */
static void fe_canonical(fe *a)
{
    uint64_t w[4];
    fe_to_words(w, a);
    fe_from_words(a, w);
}

static int fe_is_zero(const fe *a)
{
    uint64_t w[4];
    fe_to_words(w, a);
    return (w[0] | w[1] | w[2] | w[3]) == 0;
}

static int fe_equal(const fe *a, const fe *b)
{
    uint64_t x[4], y[4];
    fe_to_words(x, a);
    fe_to_words(y, b);
    return ((x[0] ^ y[0]) | (x[1] ^ y[1]) | (x[2] ^ y[2]) | (x[3] ^ y[3])) == 0;
}

/* ======================================================================== */
/* 6. Curve25519 in affine Montgomery form:  v^2 = u^3 + A u^2 + u           */
/*                                                                           */
/* Set-up code only: one field inversion per operation is acceptable here.   */
/* None of it is constant-time.                                              */
/* ======================================================================== */

#define CURVE_A 486662

typedef struct {
    fe u, v;            /* never the point at infinity; limbs tight */
} mpoint;

static const fe FE_A = {{CURVE_A, 0, 0, 0, 0}};
static const fe FE_2A = {{2 * CURVE_A, 0, 0, 0, 0}};

static mpoint BASE;                         /* B = (9, v) from RFC 7748 section 4.1 */
static mpoint TAB[HALF_WINDOW + 1];         /* TAB[i] = i * 8B for i = 1..W, canonical */
static mpoint TABQ;                         /* (2W+1) * 8B: one window to the next    */

static int mont_on_curve(const mpoint *p)
{
    fe lhs, rhs, t;
    fe_sq(&lhs, &p->v);
    fe_add(&t, &p->u, &FE_A);
    fe_mul(&t, &t, &p->u);
    fe_add(&t, &t, &FE_ONE);
    fe_mul(&rhs, &t, &p->u);                /* u * ((u + A) * u + 1) */
    return fe_equal(&lhs, &rhs);
}

/* r = p + q.  Fails (-1) when u_p == u_q, i.e. q = +/-p. */
static int mont_add(mpoint *r, const mpoint *p, const mpoint *q)
{
    fe du, dv, lam, t, u3, v3;

    fe_sub(&du, &q->u, &p->u);
    if (fe_is_zero(&du))
        return -1;
    fe_sub(&dv, &q->v, &p->v);
    fe_invert(&t, &du);
    fe_mul(&lam, &dv, &t);                  /* lambda = (v_q - v_p) / (u_q - u_p) */
    fe_sq(&u3, &lam);
    fe_sub(&u3, &u3, &FE_A);
    fe_sub(&u3, &u3, &p->u);
    fe_sub(&u3, &u3, &q->u);                /* lambda^2 - A - u_p - u_q */
    fe_sub(&t, &p->u, &u3);
    fe_mul(&v3, &lam, &t);
    fe_sub(&v3, &v3, &p->v);                /* lambda * (u_p - u_3) - v_p */
    r->u = u3;
    r->v = v3;
    return 0;
}

/* r = 2p.  Fails (-1) for the point of order two. */
static int mont_double(mpoint *r, const mpoint *p)
{
    fe num, den, lam, t, u3, v3;

    if (fe_is_zero(&p->v))
        return -1;
    fe_sq(&num, &p->u);
    fe_add(&t, &num, &num);
    fe_add(&num, &num, &t);                 /* 3u^2 */
    fe_mul(&t, &p->u, &FE_2A);
    fe_add(&num, &num, &t);
    fe_add(&num, &num, &FE_ONE);            /* 3u^2 + 2Au + 1 */
    fe_add(&den, &p->v, &p->v);
    fe_invert(&den, &den);
    fe_mul(&lam, &num, &den);
    fe_sq(&u3, &lam);
    fe_sub(&u3, &u3, &FE_A);
    fe_sub(&u3, &u3, &p->u);
    fe_sub(&u3, &u3, &p->u);                /* lambda^2 - A - 2u */
    fe_sub(&t, &p->u, &u3);
    fe_mul(&v3, &lam, &t);
    fe_sub(&v3, &v3, &p->v);
    r->u = u3;
    r->v = v3;
    return 0;
}

/*
 * r = k * B for a clamped scalar k (bit 254 set, bit 255 clear).
 * Returns -1 if the chain meets an exceptional point, which for a random
 * scalar has probability about 2^-250; the caller then draws another one.
 */
static int mont_scalarmult_base(mpoint *r, const uint8_t k[32])
{
    mpoint acc = BASE;
    int rc = 0;

    for (int i = 253; i >= 0 && rc == 0; i--) {
        rc = mont_double(&acc, &acc);
        if (rc == 0 && ((k[i >> 3] >> (i & 7)) & 1))
            rc = mont_add(&acc, &acc, &BASE);
    }
    *r = acc;
    secure_zero(&acc, sizeof acc);
    return rc;
}

static void mpoint_canonical(mpoint *p)
{
    fe_canonical(&p->u);
    fe_canonical(&p->v);
}

/* Build B, the table of i * 8B and the window step.  Returns 0 on success. */
static int curve_init(void)
{
    static const char base_v_hex[] =        /* little-endian V(P), RFC 7748 section 4.1 */
        "d9d3ce7ea2c5e929b2617c6d7e4d3d924cd148772cdd1ee0b486a0b8a119ae20";
    uint8_t bytes[32];
    mpoint s, t;

    fe_from_bytes(&BASE.u, BASE_U);
    hex_decode(bytes, base_v_hex, 32);
    fe_from_bytes(&BASE.v, bytes);
    if (!mont_on_curve(&BASE))
        return -1;

    /* S = 8B */
    if (mont_double(&s, &BASE) || mont_double(&s, &s) || mont_double(&s, &s))
        return -1;

    TAB[1] = s;
    if (mont_double(&TAB[2], &s))
        return -1;
    for (int i = 3; i <= HALF_WINDOW; i++)
        if (mont_add(&TAB[i], &TAB[i - 1], &s))
            return -1;

    /* Q = (2W + 1) S = 2 * (W S) + S */
    if (mont_double(&t, &TAB[HALF_WINDOW]) || mont_add(&TABQ, &t, &s))
        return -1;

    for (int i = 1; i <= HALF_WINDOW; i++) {
        mpoint_canonical(&TAB[i]);
        if (!mont_on_curve(&TAB[i]))
            return -1;
    }
    mpoint_canonical(&TABQ);
    return mont_on_curve(&TABQ) ? 0 : -1;
}

/* ======================================================================== */
/* 7. Target patterns                                                        */
/* ======================================================================== */

typedef struct {
    uint32_t mask;                  /* 0xF in every fixed hex digit            */
    uint32_t value[MAX_TARGETS];
    int      count;
    int      fixed_digits;
    uint64_t bitmap[1024];          /* 65536-bit prefilter for target lists    */
} pattern_t;

static pattern_t g_pat;

/* The firmware reserves node numbers below 4 and the broadcast address. */
static inline int node_id_reserved(uint32_t id)
{
    return id < 4u || id == 0xffffffffu;
}

static inline uint32_t pattern_slot(uint32_t masked)
{
    return (masked * 0x9e3779b1u) >> 16;
}

static inline int pattern_match(uint32_t id)
{
    const uint32_t m = id & g_pat.mask;

    if (g_pat.count == 1) {
        if (LIKELY(m != g_pat.value[0]))
            return 0;
    } else {
        const uint32_t s = pattern_slot(m);
        int found = 0;
        if (LIKELY(!((g_pat.bitmap[s >> 6] >> (s & 63)) & 1)))
            return 0;
        for (int k = 0; k < g_pat.count; k++)
            found |= (m == g_pat.value[k]);
        if (!found)
            return 0;
    }
    return !node_id_reserved(id);
}

/* A pattern that can never match: used by the benchmark and the self-tests. */
static void pattern_set_none(void)
{
    memset(&g_pat, 0, sizeof g_pat);
    g_pat.mask = 0xffffffffu;
    g_pat.value[0] = 0xffffffffu;   /* matches only the reserved broadcast id */
    g_pat.count = 1;
    g_pat.fixed_digits = 8;
}

/*
 * Parse "dead", "!c0de", "dc80*a", "dead,beef,cafe" into g_pat.
 * Returns NULL on success or a message describing what is wrong.
 */
static const char *pattern_parse(const char *text)
{
    const char *p = text;
    int have_mask = 0;

    memset(&g_pat, 0, sizeof g_pat);
    for (;;) {
        uint32_t mask = 0, value = 0;
        int digits = 0, duplicate = 0;

        if (*p == '!')
            p++;
        for (; *p != '\0' && *p != ','; p++, digits++) {
            const int shift = 28 - 4 * digits;
            const int d = hex_digit(*p);
            if (digits >= 8)
                return "a target has at most 8 hex digits";
            if (d >= 0) {
                mask |= (uint32_t)0xf << shift;
                value |= (uint32_t)d << shift;
            } else if (strchr("*?.xX", *p) == NULL) {
                return "use hex digits 0-9 a-f, and * ? . or x for \"any digit\"";
            }
        }
        if (digits == 0)
            return "empty target";
        if (mask == 0)
            return "a target needs at least one fixed digit";
        if (have_mask && mask != g_pat.mask)
            return "every target in a list must fix the same digit positions";
        if (mask == 0xffffffffu && node_id_reserved(value))
            return "that node ID is reserved by the firmware (!00000000-!00000003, !ffffffff)";
        g_pat.mask = mask;
        have_mask = 1;

        for (int k = 0; k < g_pat.count; k++)
            duplicate |= (g_pat.value[k] == value);
        if (!duplicate) {
            if (g_pat.count == MAX_TARGETS)
                return "too many targets (the limit is 64)";
            g_pat.value[g_pat.count++] = value;
        }
        if (*p == '\0')
            break;
        p++;                        /* skip the comma */
    }

    for (uint32_t m = g_pat.mask; m != 0; m >>= 4)
        g_pat.fixed_digits += (m & 15) != 0;
    for (int k = 0; k < g_pat.count; k++) {
        const uint32_t s = pattern_slot(g_pat.value[k]);
        g_pat.bitmap[s >> 6] |= (uint64_t)1 << (s & 63);
    }
    return NULL;
}

/* Average number of keys that must be tried per hit. */
static double pattern_expected_keys(void)
{
    return pow(16.0, g_pat.fixed_digits) / g_pat.count;
}

/* ======================================================================== */
/* 8. Search engine: lanes, hit verification, output                         */
/* ======================================================================== */

/*
 * A lane is one independent walk.  Candidate j (0 <= j < WINDOW) of window
 * number win is the scalar  base + 8 * (win * WINDOW + j);  the window's
 * centre is candidate j = HALF_WINDOW.
 */
typedef struct {
    uint8_t  base[32];              /* secret, clamped                         */
    uint64_t win;                   /* windows completed since base was drawn  */
} lane_t;

#define LANE_MAX_WINDOWS ((uint64_t)1 << 48)

typedef struct {
    uint8_t  secret[32];
    uint8_t  public_key[32];
    uint32_t node_id;
} hit_t;

typedef enum { OUT_TEXT, OUT_JSON, OUT_CAPTURE } out_mode_t;

static struct {
    out_mode_t  mode;
    uint64_t    hits_wanted;
    uint64_t    hits;               /* guarded by the output lock              */
    hit_t      *capture;            /* OUT_CAPTURE: array of hits_wanted slots */
    int         progress_shown;     /* a progress line is on the terminal      */
    atomic_int  stop;
    atomic_ullong faults;           /* fast-path results the reference refused */
} g_run;

static volatile sig_atomic_t g_interrupted;

static void on_interrupt(int sig)
{
    (void)sig;
    g_interrupted = 1;
}

/* out = base + 8 * index, as a 256-bit little-endian integer. */
static void scalar_offset(uint8_t out[32], const uint8_t base[32], uint64_t index)
{
    uint64_t add = index << 3;      /* index < 2^59 by LANE_MAX_WINDOWS */
    unsigned carry = 0;

    for (int i = 0; i < 32; i++) {
        const unsigned s = (unsigned)base[i] + (unsigned)(add & 0xff) + carry;
        out[i] = (uint8_t)s;
        carry = s >> 8;
        add >>= 8;
    }
}

/*
 * Draw a fresh base scalar for a lane and compute its first window centre
 * (canonical).  The base is uniform over the clamped scalars except for the
 * top 2^72 of them, so that the walk can never carry out of the clamped
 * range.
 */
static void lane_seed(lane_t *lane, mpoint *centre)
{
    uint8_t k[32];

    for (;;) {
        uint8_t top = 0xff;

        os_random(lane->base, 32);
        lane->base[0] &= 248;
        lane->base[31] &= 127;
        lane->base[31] |= 64;
        for (int i = 9; i < 31; i++)
            top &= lane->base[i];
        if (top == 0xff && lane->base[31] == 0x7f)
            continue;               /* within 2^72 of 2^255: probability 2^-183 */

        scalar_offset(k, lane->base, HALF_WINDOW);
        if (mont_scalarmult_base(centre, k) == 0)
            break;
    }
    mpoint_canonical(centre);
    lane->win = 0;
    secure_zero(k, sizeof k);
}

static void print_hit(uint64_t number, const hit_t *h)
{
    char sk_hex[65], pk_hex[65], sk_b64[45], pk_b64[45];

    hex_encode(sk_hex, h->secret, 32);
    hex_encode(pk_hex, h->public_key, 32);
    base64_encode(sk_b64, h->secret, 32);
    base64_encode(pk_b64, h->public_key, 32);

    if (g_run.progress_shown)
        fprintf(stderr, "\r%78s\r", "");
    if (g_run.mode == OUT_JSON) {
        printf("{\"node_id\":\"!%08" PRIx32 "\",\"node_num\":%" PRIu32 ","
               "\"private_key_hex\":\"%s\",\"private_key_base64\":\"%s\","
               "\"public_key_hex\":\"%s\",\"public_key_base64\":\"%s\","
               "\"command\":\"meshtastic --set security.private_key \\\"base64:%s\\\"\"}\n",
               h->node_id, h->node_id, sk_hex, sk_b64, pk_hex, pk_b64, sk_b64);
    } else {
        printf("hit %" PRIu64 ": node id !%08" PRIx32 "\n"
               "  private key (hex):    %s\n"
               "  private key (base64): %s\n"
               "  public key (hex):     %s\n"
               "  public key (base64):  %s\n"
               "  install with:         meshtastic --set security.private_key \"base64:%s\"\n",
               number, h->node_id, sk_hex, sk_b64, pk_hex, pk_b64, sk_b64);
    }
    fflush(stdout);
    secure_zero(sk_hex, sizeof sk_hex);
    secure_zero(sk_b64, sizeof sk_b64);
}

/*
 * The fast path says candidate j of this lane has node ID fast_id and that it
 * matches.  Recompute everything independently; only a confirmed key is
 * reported.  Always returns 1: the lane's base is finished either way.
 */
static NOINLINE int engine_on_match(const lane_t *lane, unsigned j, uint32_t fast_id)
{
    hit_t h;

    scalar_offset(h.secret, lane->base, lane->win * WINDOW + j);
    x25519_ref(h.public_key, h.secret, BASE_U);
    h.node_id = crc32_ref(h.public_key, 32);

    if (h.node_id != fast_id || !pattern_match(h.node_id)) {
        atomic_fetch_add(&g_run.faults, 1);
    } else {
        output_lock();
        if (g_run.hits < g_run.hits_wanted) {
            if (g_run.mode == OUT_CAPTURE)
                g_run.capture[g_run.hits] = h;
            else
                print_hit(g_run.hits + 1, &h);
            if (++g_run.hits == g_run.hits_wanted)
                atomic_store(&g_run.stop, 1);
        }
        output_unlock();
    }
    secure_zero(&h, sizeof h);
    return 1;
}

/* What a kernel records for the self-test instead of (as well as) matching. */
typedef struct {
    uint64_t (*words)[4];           /* [lane * WINDOW + j]: canonical public key */
    uint32_t *id;                   /* [lane * WINDOW + j]: its node ID          */
} walk_dump;

/*
 * A backend runs `lanes` walks side by side.  Its state begins with the
 * lane_t table, so generic code can read which scalars a window covers.
 */
typedef struct {
    const char *name;
    unsigned    lanes;
    size_t      state_size;
    int       (*available)(void);
    void      (*seed)(void *state);                         /* fresh bases for every lane  */
    void      (*window)(void *state, const walk_dump *d);   /* one window on every lane    */
    /* place a lane's next window at centre c (canonical); seeding and tests use it */
    void      (*set_centre)(void *state, unsigned lane, const mpoint *c);
} backend_t;

/* ======================================================================== */
/* 9. Scalar kernel (one lane, 5x51-bit limbs)                               */
/* ======================================================================== */

typedef struct {
    lane_t lane[1];
    fe     um, vm;                          /* centre of the next window, tight */
    fe     prefix[HALF_WINDOW + 1];
} walk1_t;

static void walk1_set_centre(void *state, unsigned lane, const mpoint *c)
{
    walk1_t *w = (walk1_t *)state;
    (void)lane;
    w->um = c->u;
    w->vm = c->v;
}

static void walk1_seed(void *state)
{
    walk1_t *w = (walk1_t *)state;
    mpoint centre;

    lane_seed(&w->lane[0], &centre);
    walk1_set_centre(w, 0, &centre);
}

static int walk1_available(void)
{
    return 1;
}

static void walk1_window(void *state, const walk_dump *dump)
{
    walk1_t *w = (walk1_t *)state;
    lane_t *lane = &w->lane[0];
    fe *prefix = w->prefix;
    fe e0, e, d, acc, inv, t, lam;
    uint64_t msg[2][4];
    uint32_t id[2];
    int retired = 0;

    /*
     * e0 = 4p - A - u_M, so that  u(M +/- P_i) = lambda^2 + (e0 - u_i)
     * with every limb non-negative.  e0 and e0 - u_i are below 2^53.
     */
    e0.v[0] = 0x1fffffffffffb4ULL - CURVE_A - w->um.v[0];
    for (int k = 1; k < 5; k++)
        e0.v[k] = 0x1ffffffffffffcULL - w->um.v[k];

    /* Forward pass: prefix[i] = (u_Q - u_M) * prod_{k<=i} (u_k - u_M). */
    fe_sub_lazy(&prefix[0], &TABQ.u, &w->um);
    for (int i = 1; i <= HALF_WINDOW; i++) {
        fe_sub_lazy(&d, &TAB[i].u, &w->um);
        fe_mul(&prefix[i], &prefix[i - 1], &d);
    }
    if (UNLIKELY(fe_is_zero(&prefix[HALF_WINDOW]))) {
        walk1_seed(w);                      /* the centre hit a table point */
        return;
    }
    fe_invert(&acc, &prefix[HALF_WINDOW]);

    /* Backward pass: peel off one inverse per step and finish two keys. */
    for (int i = HALF_WINDOW; i >= 1; i--) {
        fe_mul(&inv, &acc, &prefix[i - 1]);         /* 1 / (u_i - u_M) */
        fe_sub_lazy(&d, &TAB[i].u, &w->um);
        fe_mul(&acc, &acc, &d);

        for (int k = 0; k < 5; k++)
            e.v[k] = e0.v[k] - TAB[i].u.v[k];

        fe_sub_lazy(&t, &TAB[i].v, &w->vm);
        fe_mul(&lam, &t, &inv);
        fe_sq_add_to_words(msg[0], &lam, &e);       /* M + P_i */

        fe_add_lazy(&t, &TAB[i].v, &w->vm);
        fe_mul(&lam, &t, &inv);
        fe_sq_add_to_words(msg[1], &lam, &e);       /* M - P_i */

        crc_msgs((const uint64_t (*)[4])msg, id, 2);

        if (UNLIKELY(dump != NULL)) {
            memcpy(dump->words[HALF_WINDOW + i], msg[0], 32);
            memcpy(dump->words[HALF_WINDOW - i], msg[1], 32);
            dump->id[HALF_WINDOW + i] = id[0];
            dump->id[HALF_WINDOW - i] = id[1];
        }
        if (UNLIKELY(pattern_match(id[0])) && !retired)
            retired = engine_on_match(lane, (unsigned)(HALF_WINDOW + i), id[0]);
        if (UNLIKELY(pattern_match(id[1])) && !retired)
            retired = engine_on_match(lane, (unsigned)(HALF_WINDOW - i), id[1]);
    }

    /* The centre itself. */
    fe_to_words(msg[0], &w->um);
    crc_msgs((const uint64_t (*)[4])msg, id, 1);
    if (UNLIKELY(dump != NULL)) {
        memcpy(dump->words[HALF_WINDOW], msg[0], 32);
        dump->id[HALF_WINDOW] = id[0];
    }
    if (UNLIKELY(pattern_match(id[0])) && !retired)
        retired = engine_on_match(lane, HALF_WINDOW, id[0]);

    if (retired || lane->win + 1 >= LANE_MAX_WINDOWS) {
        walk1_seed(w);
        return;
    }

    /* Move the centre: M' = M + Q.  Here acc = 1 / (u_Q - u_M). */
    fe_sub_lazy(&t, &TABQ.v, &w->vm);
    fe_mul(&lam, &t, &acc);
    fe_sq(&t, &lam);
    for (int k = 0; k < 5; k++)
        t.v[k] += e0.v[k] - TABQ.u.v[k];            /* lambda^2 - A - u_M - u_Q */
    fe_carry(&t);
    fe_sub_lazy(&d, &w->um, &t);
    fe_mul(&d, &lam, &d);
    fe_sub(&w->vm, &d, &w->vm);                     /* lambda * (u_M - u') - v_M */
    w->um = t;
    lane->win++;
}

static const backend_t BACKEND_SCALAR = {
    "scalar", 1, sizeof(walk1_t), walk1_available, walk1_seed, walk1_window, walk1_set_centre,
};

/* ======================================================================== */
/* 10. AVX2 kernel (four lanes, radix 2^25.5)                                */
/*                                                                           */
/* A field element is ten limbs of alternately 26 and 25 bits; an fe4 holds  */
/* four of them, limb k of lane l in 64-bit element l of vector k.           */
/*                                                                           */
/* Bounds:                                                                   */
/*   reduced  even limbs < 2^26, odd limbs < 2^25 + 2^12: what a product     */
/*            or FE4_CARRY leaves behind                                     */
/*   3x       limbs up to three times that: a reduced value plus the limbs   */
/*            of 2p, as every lazy subtraction here produces                 */
/* fe4_mul_pre accepts 3x operands (19 * 3 * 2^26 < 2^32, and ten products   */
/* sum to less than 2^63); fe4_sq needs a reduced operand.                   */
/* ======================================================================== */

#ifdef MTV_X86

#define AVX2_FN     __attribute__((target("avx2")))
#define AVX2_INLINE static inline __attribute__((always_inline, target("avx2")))

#define V4_ADD(a, b)  _mm256_add_epi64((a), (b))
#define V4_SUB(a, b)  _mm256_sub_epi64((a), (b))
#define V4_MUL(a, b)  _mm256_mul_epu32((a), (b))       /* low 32 bits x low 32 bits */
#define V4_AND(a, b)  _mm256_and_si256((a), (b))
#define V4_OR(a, b)   _mm256_or_si256((a), (b))
#define V4_SHR(a, n)  _mm256_srli_epi64((a), (n))
#define V4_SHL(a, n)  _mm256_slli_epi64((a), (n))
#define V4_SET1(x)    _mm256_set1_epi64x((long long)(x))

/*
 * The two operations a schoolbook product is made of: `dst = x * mem` and
 * `acc += x * mem`, where x is a register and mem a limb read from memory.
 *
 * They are spelled out as instructions because neither gcc nor clang
 * otherwise keeps the ten column accumulators in registers: left to choose,
 * both hoist the twenty operand limbs into registers instead and spill every
 * partial product, which doubles the cost of a multiplication.  Building with
 * -DMTV_PLAIN_INTRINSICS selects the equivalent portable definitions.
 */
#if defined(__GNUC__) && !defined(MTV_PLAIN_INTRINSICS)
#define V4_MULM(dst, x, mem)                                                \
    __asm__("vpmuludq {%[m], %[r], %[d]|%[d], %[r], %[m]}"                  \
            : [d] "=x"(dst) : [r] "x"(x), [m] "m"(mem))
#define V4_MULADDM(acc, x, mem)                                             \
    do {                                                                    \
        __m256i prod_;                                                      \
        __asm__("vpmuludq {%[m], %[r], %[t]|%[t], %[r], %[m]}\n\t"          \
                "vpaddq {%[t], %[h], %[h]|%[h], %[h], %[t]}"                \
                : [h] "+x"(acc), [t] "=&x"(prod_) : [r] "x"(x), [m] "m"(mem)); \
    } while (0)
#else
#define V4_MULM(dst, x, mem)    ((dst) = V4_MUL((x), (mem)))
#define V4_MULADDM(acc, x, mem) ((acc) = V4_ADD((acc), V4_MUL((x), (mem))))
#endif

typedef struct {
    __m256i l[10];
} fe4;

/* limb-wise 2p and 4p: adding one keeps a subtraction's limbs non-negative */
static const uint64_t TWO_P_25[10] = {
    0x7ffffda, 0x3fffffe, 0x7fffffe, 0x3fffffe, 0x7fffffe,
    0x3fffffe, 0x7fffffe, 0x3fffffe, 0x7fffffe, 0x3fffffe,
};
static const uint64_t FOUR_P_25[10] = {
    0xfffffb4, 0x7fffffc, 0xffffffc, 0x7fffffc, 0xffffffc,
    0x7fffffc, 0xffffffc, 0x7fffffc, 0xffffffc, 0x7fffffc,
};

/* canonical fe -> ten limbs, and any ten small limbs -> loose fe */
static void fe_to_limbs25(uint64_t h[10], const fe *a)
{
    for (int k = 0; k < 5; k++) {
        h[2 * k] = a->v[k] & 0x3ffffff;
        h[2 * k + 1] = a->v[k] >> 26;
    }
}

static void fe_from_limbs25(fe *r, const uint64_t h[10])
{
    for (int k = 0; k < 5; k++)
        r->v[k] = h[2 * k] + (h[2 * k + 1] << 26);
}

/* src and dst are uint64_t[10][4] arrays: [limb][lane] */
AVX2_INLINE void fe4_load(fe4 *r, const uint64_t *src)
{
    for (int k = 0; k < 10; k++)
        r->l[k] = _mm256_loadu_si256((const __m256i *)(const void *)(src + 4 * k));
}

AVX2_INLINE void fe4_store(uint64_t *dst, const fe4 *a)
{
    for (int k = 0; k < 10; k++)
        _mm256_storeu_si256((__m256i *)(void *)(dst + 4 * k), a->l[k]);
}

/* the same field element in all four lanes */
AVX2_INLINE void fe4_broadcast(fe4 *r, const uint64_t limbs[10])
{
    for (int k = 0; k < 10; k++)
        r->l[k] = V4_SET1(limbs[k]);
}

/*
 * Carry ten 64-bit columns (each < 2^63) down to a reduced element.
 * Two interleaved chains keep the dependency path short.
 */
#define FE4_CARRY_STEP(lo, hi, bits, mask)                                  \
    do {                                                                    \
        hi = V4_ADD(hi, V4_SHR(lo, bits));                                  \
        lo = V4_AND(lo, mask);                                              \
    } while (0)

#define FE4_CARRY(out, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)              \
    do {                                                                    \
        const __m256i m26_ = V4_SET1(0x3ffffff), m25_ = V4_SET1(0x1ffffff); \
        __m256i c_;                                                         \
        FE4_CARRY_STEP(h0, h1, 26, m26_); FE4_CARRY_STEP(h4, h5, 26, m26_); \
        FE4_CARRY_STEP(h1, h2, 25, m25_); FE4_CARRY_STEP(h5, h6, 25, m25_); \
        FE4_CARRY_STEP(h2, h3, 26, m26_); FE4_CARRY_STEP(h6, h7, 26, m26_); \
        FE4_CARRY_STEP(h3, h4, 25, m25_); FE4_CARRY_STEP(h7, h8, 25, m25_); \
        FE4_CARRY_STEP(h4, h5, 26, m26_); FE4_CARRY_STEP(h8, h9, 26, m26_); \
        c_ = V4_SHR(h9, 25);                                                \
        h9 = V4_AND(h9, m25_);                                              \
        h0 = V4_ADD(h0, V4_ADD(V4_ADD(V4_SHL(c_, 4), V4_SHL(c_, 1)), c_));  /* + 19c */ \
        FE4_CARRY_STEP(h0, h1, 26, m26_);                                   \
        (out)->l[0] = h0; (out)->l[1] = h1; (out)->l[2] = h2; (out)->l[3] = h3; (out)->l[4] = h4; \
        (out)->l[5] = h5; (out)->l[6] = h6; (out)->l[7] = h7; (out)->l[8] = h8; (out)->l[9] = h9; \
    } while (0)

/* g19[j] = 19 * g[j] for j = 1..9: the factor that folds 2^255 back to 19 */
AVX2_INLINE void fe4_mul19(fe4 *g19, const fe4 *g)
{
    const __m256i v19 = V4_SET1(19);
    for (int k = 1; k < 10; k++)
        V4_MULM(g19->l[k], v19, g->l[k]);
}

/*
 * out = f * g with g19 = fe4_mul19(g).  out may alias f or g.
 *
 * Schoolbook, one row per limb of f.  The ten column sums h0..h9 stay in
 * registers and the limbs of g are read straight from memory, so nothing is
 * spilled.  Column k collects f_i * g_j with i + j = k (mod 10); a product
 * that wraps past 2^255 uses 19 * g_j, and a product of two odd limbs is
 * doubled (a2 = 2 * f_i), because odd limbs sit half a bit low.
 */
AVX2_INLINE void fe4_mul_pre(fe4 *out, const fe4 *f, const fe4 *g, const fe4 *g19)
{
    __m256i h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, a, a2;

#define G(j)        g->l[j]
#define G19(j)      g19->l[j]
#define M0(k, x, y) V4_MULM(h##k, x, y)
#define MA(k, x, y) V4_MULADDM(h##k, x, y)
    a = f->l[0];
    M0(0, a, G(0)); M0(1, a, G(1)); M0(2, a, G(2)); M0(3, a, G(3)); M0(4, a, G(4));
    M0(5, a, G(5)); M0(6, a, G(6)); M0(7, a, G(7)); M0(8, a, G(8)); M0(9, a, G(9));
    a = f->l[1]; a2 = V4_ADD(a, a);
    MA(1, a, G(0)); MA(2, a2, G(1)); MA(3, a, G(2)); MA(4, a2, G(3)); MA(5, a, G(4));
    MA(6, a2, G(5)); MA(7, a, G(6)); MA(8, a2, G(7)); MA(9, a, G(8)); MA(0, a2, G19(9));
    a = f->l[2];
    MA(2, a, G(0)); MA(3, a, G(1)); MA(4, a, G(2)); MA(5, a, G(3)); MA(6, a, G(4));
    MA(7, a, G(5)); MA(8, a, G(6)); MA(9, a, G(7)); MA(0, a, G19(8)); MA(1, a, G19(9));
    a = f->l[3]; a2 = V4_ADD(a, a);
    MA(3, a, G(0)); MA(4, a2, G(1)); MA(5, a, G(2)); MA(6, a2, G(3)); MA(7, a, G(4));
    MA(8, a2, G(5)); MA(9, a, G(6)); MA(0, a2, G19(7)); MA(1, a, G19(8)); MA(2, a2, G19(9));
    a = f->l[4];
    MA(4, a, G(0)); MA(5, a, G(1)); MA(6, a, G(2)); MA(7, a, G(3)); MA(8, a, G(4));
    MA(9, a, G(5)); MA(0, a, G19(6)); MA(1, a, G19(7)); MA(2, a, G19(8)); MA(3, a, G19(9));
    a = f->l[5]; a2 = V4_ADD(a, a);
    MA(5, a, G(0)); MA(6, a2, G(1)); MA(7, a, G(2)); MA(8, a2, G(3)); MA(9, a, G(4));
    MA(0, a2, G19(5)); MA(1, a, G19(6)); MA(2, a2, G19(7)); MA(3, a, G19(8)); MA(4, a2, G19(9));
    a = f->l[6];
    MA(6, a, G(0)); MA(7, a, G(1)); MA(8, a, G(2)); MA(9, a, G(3)); MA(0, a, G19(4));
    MA(1, a, G19(5)); MA(2, a, G19(6)); MA(3, a, G19(7)); MA(4, a, G19(8)); MA(5, a, G19(9));
    a = f->l[7]; a2 = V4_ADD(a, a);
    MA(7, a, G(0)); MA(8, a2, G(1)); MA(9, a, G(2)); MA(0, a2, G19(3)); MA(1, a, G19(4));
    MA(2, a2, G19(5)); MA(3, a, G19(6)); MA(4, a2, G19(7)); MA(5, a, G19(8)); MA(6, a2, G19(9));
    a = f->l[8];
    MA(8, a, G(0)); MA(9, a, G(1)); MA(0, a, G19(2)); MA(1, a, G19(3)); MA(2, a, G19(4));
    MA(3, a, G19(5)); MA(4, a, G19(6)); MA(5, a, G19(7)); MA(6, a, G19(8)); MA(7, a, G19(9));
    a = f->l[9]; a2 = V4_ADD(a, a);
    MA(9, a, G(0)); MA(0, a2, G19(1)); MA(1, a, G19(2)); MA(2, a2, G19(3)); MA(3, a, G19(4));
    MA(4, a2, G19(5)); MA(5, a, G19(6)); MA(6, a2, G19(7)); MA(7, a, G19(8)); MA(8, a2, G19(9));
#undef G
#undef G19
#undef M0
#undef MA
    FE4_CARRY(out, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

AVX2_INLINE void fe4_mul(fe4 *out, const fe4 *f, const fe4 *g)
{
    fe4 g19;
    fe4_mul19(&g19, g);
    fe4_mul_pre(out, f, g, &g19);
}

/*
 * The ten unreduced columns h0..h9 of f^2, declared in the caller's scope;
 * f reduced.
 *
 * Row i multiplies f_i, 2 f_i, or 4 f_i when both limbs are odd, by the
 * limbs f_j with j >= i, read from memory.  The products that wrap past
 * 2^255 use the scaled limbs 38 f_5, 19 f_6, 38 f_7, 19 f_8, 38 f_9.
 */
#define FE4_SQ_ROWS(f)                                                              \
    const __m256i v19 = V4_SET1(19), v38 = V4_SET1(38);                             \
    __m256i h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, a, a2, w38_5, w19_6, w38_7, w19_8, w38_9; \
    V4_MULM(w38_5, v38, (f)->l[5]);                                                 \
    V4_MULM(w19_6, v19, (f)->l[6]);                                                 \
    V4_MULM(w38_7, v38, (f)->l[7]);                                                 \
    V4_MULM(w19_8, v19, (f)->l[8]);                                                 \
    V4_MULM(w38_9, v38, (f)->l[9]);                                                 \
    a = (f)->l[0]; a2 = V4_ADD(a, a);                                               \
    SQ0(0, a, F(0)); SQ0(1, a2, F(1)); SQ0(2, a2, F(2)); SQ0(3, a2, F(3)); SQ0(4, a2, F(4)); \
    SQ0(5, a2, F(5)); SQ0(6, a2, F(6)); SQ0(7, a2, F(7)); SQ0(8, a2, F(8)); SQ0(9, a2, F(9)); \
    a = V4_ADD((f)->l[1], (f)->l[1]); a2 = V4_ADD(a, a);    /* 2 f1 and 4 f1 */     \
    SQA(0, a, w38_9); SQA(2, a, F(1)); SQA(3, a, F(2)); SQA(4, a2, F(3)); SQA(5, a, F(4)); \
    SQA(6, a2, F(5)); SQA(7, a, F(6)); SQA(8, a2, F(7)); SQA(9, a, F(8));           \
    a = (f)->l[2]; a2 = V4_ADD(a, a);                                               \
    SQA(0, a2, w19_8); SQA(1, a, w38_9); SQA(4, a, F(2)); SQA(5, a2, F(3)); SQA(6, a2, F(4)); \
    SQA(7, a2, F(5)); SQA(8, a2, F(6)); SQA(9, a2, F(7));                           \
    a = V4_ADD((f)->l[3], (f)->l[3]); a2 = V4_ADD(a, a);    /* 2 f3 and 4 f3 */     \
    SQA(0, a, w38_7); SQA(1, a, w19_8); SQA(2, a, w38_9); SQA(6, a, F(3)); SQA(7, a, F(4)); \
    SQA(8, a2, F(5)); SQA(9, a, F(6));                                              \
    a = (f)->l[4]; a2 = V4_ADD(a, a);                                               \
    SQA(0, a2, w19_6); SQA(1, a, w38_7); SQA(2, a2, w19_8); SQA(3, a, w38_9); SQA(8, a, F(4)); \
    SQA(9, a2, F(5));                                                               \
    a = (f)->l[5]; a2 = V4_ADD(a, a);                                               \
    SQA(0, a, w38_5); SQA(1, a2, w19_6); SQA(2, a2, w38_7); SQA(3, a2, w19_8); SQA(4, a2, w38_9); \
    a = (f)->l[6]; a2 = V4_ADD(a, a);                                               \
    SQA(2, a, w19_6); SQA(3, a, w38_7); SQA(4, a2, w19_8); SQA(5, a, w38_9);        \
    a = (f)->l[7]; a2 = V4_ADD(a, a);                                               \
    SQA(4, a, w38_7); SQA(5, a2, w19_8); SQA(6, a2, w38_9);                         \
    a = (f)->l[8];                                                                  \
    SQA(6, a, w19_8); SQA(7, a, w38_9);                                             \
    a = (f)->l[9];                                                                  \
    SQA(8, a, w38_9)

#define F(j)         f->l[j]
#define SQ0(k, x, y) V4_MULM(h##k, x, y)
#define SQA(k, x, y) V4_MULADDM(h##k, x, y)

AVX2_INLINE void fe4_sq(fe4 *out, const fe4 *f)
{
    FE4_SQ_ROWS(f);
    FE4_CARRY(out, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

/* Bring an element with limbs below 2^63 back to reduced form. */
AVX2_INLINE void fe4_reduce(fe4 *a)
{
    __m256i h0 = a->l[0], h1 = a->l[1], h2 = a->l[2], h3 = a->l[3], h4 = a->l[4];
    __m256i h5 = a->l[5], h6 = a->l[6], h7 = a->l[7], h8 = a->l[8], h9 = a->l[9];
    FE4_CARRY(a, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

/* Out-of-line copies for the inversion chain, which is not worth inlining. */
AVX2_FN static NOINLINE void fe4_mul_call(fe4 *out, const fe4 *f, const fe4 *g)
{
    fe4_mul(out, f, g);
}

AVX2_FN static NOINLINE void fe4_sq_times(fe4 *out, const fe4 *f, int n)
{
    fe4 t = *f;
    while (n-- > 0)
        fe4_sq(&t, &t);
    *out = t;
}

/* out = z^(p-2), the same addition chain as fe_invert; z reduced */
AVX2_FN static NOINLINE void fe4_invert(fe4 *out, const fe4 *z)
{
    fe4 z2, z9, z11, z_5_0, z_10_0, z_20_0, z_50_0, z_100_0, t;

    fe4_sq_times(&z2, z, 1);
    fe4_sq_times(&t, &z2, 2);
    fe4_mul_call(&z9, &t, z);
    fe4_mul_call(&z11, &z9, &z2);
    fe4_sq_times(&t, &z11, 1);
    fe4_mul_call(&z_5_0, &t, &z9);
    fe4_sq_times(&t, &z_5_0, 5);     fe4_mul_call(&z_10_0, &t, &z_5_0);
    fe4_sq_times(&t, &z_10_0, 10);   fe4_mul_call(&z_20_0, &t, &z_10_0);
    fe4_sq_times(&t, &z_20_0, 20);   fe4_mul_call(&t, &t, &z_20_0);
    fe4_sq_times(&t, &t, 10);        fe4_mul_call(&z_50_0, &t, &z_10_0);
    fe4_sq_times(&t, &z_50_0, 50);   fe4_mul_call(&z_100_0, &t, &z_50_0);
    fe4_sq_times(&t, &z_100_0, 100); fe4_mul_call(&t, &t, &z_100_0);
    fe4_sq_times(&t, &t, 50);        fe4_mul_call(&t, &t, &z_50_0);
    fe4_sq_times(&t, &t, 5);         fe4_mul_call(out, &t, &z11);
}

/*
 * Exact canonical encoding for the rare element FE4_FINISH cannot complete:
 * hand each lane to the scalar code.  h holds ten limbs below 2^27.
 */
AVX2_FN static NOINLINE void fe4_to_words_slow(__m256i w[4], const __m256i h[10])
{
    uint64_t limbs[10][4], words[4][4];

    for (int k = 0; k < 10; k++)
        _mm256_storeu_si256((__m256i *)(void *)limbs[k], h[k]);
    for (int l = 0; l < 4; l++) {
        uint64_t one[10], out[4];
        fe t;
        for (int k = 0; k < 10; k++)
            one[k] = limbs[k][l];
        fe_from_limbs25(&t, one);
        fe_to_words(out, &t);
        for (int k = 0; k < 4; k++)
            words[k][l] = out[k];
    }
    for (int k = 0; k < 4; k++)
        w[k] = _mm256_loadu_si256((const __m256i *)(const void *)words[k]);
}

/*
 * Turn ten columns (each < 2^63) into canonical words: w[k] receives word k
 * of every lane.
 *
 * One carry chain, the wrap of 2^255 to 19 and two more steps leave limbs
 * 0, 1 and 3..9 exact and limb 2 at most 2^26.  That is already the
 * canonical value unless limb 2 overflowed or limb 0 is within 19 of 2^26,
 * the only cases in which a carry is pending or the value can reach p.  They
 * occur about once in a million vectors and take the slow, exact path.
 */
#define FE4_STEP(lo, hi, bits, mask)                                        \
    do {                                                                    \
        hi = V4_ADD(hi, V4_SHR(lo, bits));                                  \
        lo = V4_AND(lo, mask);                                              \
    } while (0)

#define FE4_FINISH(w, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)               \
    do {                                                                    \
        const __m256i m26_ = V4_SET1(0x3ffffff), m25_ = V4_SET1(0x1ffffff); \
        __m256i c_;                                                         \
        FE4_STEP(h0, h1, 26, m26_); FE4_STEP(h1, h2, 25, m25_);             \
        FE4_STEP(h2, h3, 26, m26_); FE4_STEP(h3, h4, 25, m25_);             \
        FE4_STEP(h4, h5, 26, m26_); FE4_STEP(h5, h6, 25, m25_);             \
        FE4_STEP(h6, h7, 26, m26_); FE4_STEP(h7, h8, 25, m25_);             \
        FE4_STEP(h8, h9, 26, m26_);                                         \
        c_ = V4_SHR(h9, 25);                                                \
        h9 = V4_AND(h9, m25_);                                              \
        h0 = V4_ADD(h0, V4_ADD(V4_ADD(V4_SHL(c_, 4), V4_SHL(c_, 1)), c_));  /* + 19c */ \
        FE4_STEP(h0, h1, 26, m26_); FE4_STEP(h1, h2, 25, m25_);             \
        c_ = V4_OR(_mm256_cmpgt_epi32(h0, V4_SET1(0x3ffffec)), _mm256_cmpgt_epi32(h2, m26_)); \
        if (UNLIKELY(_mm256_movemask_epi8(c_) != 0)) {                      \
            const __m256i limbs_[10] = {h0, h1, h2, h3, h4, h5, h6, h7, h8, h9}; \
            fe4_to_words_slow(w, limbs_);                                   \
        } else {                                                            \
            (w)[0] = V4_OR(V4_OR(h0, V4_SHL(h1, 26)), V4_SHL(h2, 51));      \
            (w)[1] = V4_OR(V4_OR(V4_SHR(h2, 13), V4_SHL(h3, 13)), V4_SHL(h4, 38)); \
            (w)[2] = V4_OR(V4_OR(h5, V4_SHL(h6, 25)), V4_SHL(h7, 51));      \
            (w)[3] = V4_OR(V4_OR(V4_SHR(h7, 13), V4_SHL(h8, 12)), V4_SHL(h9, 38)); \
        }                                                                   \
    } while (0)

/* canonical words of x; limbs of x below 2^63 */
AVX2_INLINE void fe4_to_words(__m256i w[4], const fe4 *x)
{
    __m256i h0 = x->l[0], h1 = x->l[1], h2 = x->l[2], h3 = x->l[3], h4 = x->l[4];
    __m256i h5 = x->l[5], h6 = x->l[6], h7 = x->l[7], h8 = x->l[8], h9 = x->l[9];
    FE4_FINISH(w, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

/*
 * canonical words of f^2 + e, the last step of every key: the addition rides
 * on the square's carry chain.  f reduced, limbs of e below 2^29.
 */
AVX2_INLINE void fe4_sq_add_to_words(__m256i w[4], const fe4 *f, const fe4 *e)
{
    FE4_SQ_ROWS(f);
    h0 = V4_ADD(h0, e->l[0]); h1 = V4_ADD(h1, e->l[1]); h2 = V4_ADD(h2, e->l[2]);
    h3 = V4_ADD(h3, e->l[3]); h4 = V4_ADD(h4, e->l[4]); h5 = V4_ADD(h5, e->l[5]);
    h6 = V4_ADD(h6, e->l[6]); h7 = V4_ADD(h7, e->l[7]); h8 = V4_ADD(h8, e->l[8]);
    h9 = V4_ADD(h9, e->l[9]);
    FE4_FINISH(w, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

#undef F
#undef SQ0
#undef SQA

/* ---- CRC-32 of groups of four messages: grp[g][word][lane] -> out[4g + lane] ---- */

typedef void (*crc_groups_fn)(const uint64_t (*grp)[4][4], uint32_t *out, size_t groups);

#define CRC_Y_SHR8(x) _mm256_srli_si256((x), 8)
#define CRC_Y_SHL1(x) _mm256_slli_epi64((x), 1)
#define CRC_REDUCE_Y(T, X01, X23, KA, KB, KC)                                       \
    CRC_REDUCE(T, X01, X23, KA, KB, KC, _mm256_clmulepi64_epi128, _mm256_xor_si256, \
               CRC_Y_SHR8, CRC_Y_SHL1)

/* AVX2 CPUs without VPCLMULQDQ (before Ice Lake and Zen 3): one lane at a time */
__attribute__((target("avx2,pclmul")))
static void crc_groups_pclmul(const uint64_t (*grp)[4][4], uint32_t *out, size_t groups)
{
    const __m128i ka = _mm_set_epi64x((long long)CRC_K1, (long long)CRC_K0);
    const __m128i kb = _mm_set_epi64x((long long)CRC_MU, (long long)CRC_K2);
    const __m128i kc = _mm_set_epi64x(0, (long long)CRC_P1);

    for (size_t g = 0; g < groups; g++) {
        for (int l = 0; l < 4; l++) {
            const __m128i x01 = _mm_set_epi64x((long long)grp[g][1][l], (long long)grp[g][0][l]);
            const __m128i x23 = _mm_set_epi64x((long long)grp[g][3][l], (long long)grp[g][2][l]);
            __m128i t;
            CRC_REDUCE_X(t, x01, x23, ka, kb, kc);
            out[4 * g + (size_t)l] = (uint32_t)_mm_extract_epi32(t, 2) ^ CRC_ZERO32;
        }
    }
}

/* two messages per 256-bit multiplication */
__attribute__((target("avx2,pclmul,vpclmulqdq")))
static void crc_groups_vpclmul(const uint64_t (*grp)[4][4], uint32_t *out, size_t groups)
{
    const __m256i ka = _mm256_set_epi64x((long long)CRC_K1, (long long)CRC_K0,
                                         (long long)CRC_K1, (long long)CRC_K0);
    const __m256i kb = _mm256_set_epi64x((long long)CRC_MU, (long long)CRC_K2,
                                         (long long)CRC_MU, (long long)CRC_K2);
    const __m256i kc = _mm256_set_epi64x(0, (long long)CRC_P1, 0, (long long)CRC_P1);
    const __m128i zero32 = _mm_set1_epi32((int)CRC_ZERO32);

    for (size_t g = 0; g < groups; g++) {
        const __m256i w0 = _mm256_loadu_si256((const __m256i *)(const void *)grp[g][0]);
        const __m256i w1 = _mm256_loadu_si256((const __m256i *)(const void *)grp[g][1]);
        const __m256i w2 = _mm256_loadu_si256((const __m256i *)(const void *)grp[g][2]);
        const __m256i w3 = _mm256_loadu_si256((const __m256i *)(const void *)grp[g][3]);
        /* each 128-bit half holds [W0, W1] or [W2, W3] of one lane: a = lanes 0 and 2, b = 1 and 3 */
        const __m256i a01 = _mm256_unpacklo_epi64(w0, w1), a23 = _mm256_unpacklo_epi64(w2, w3);
        const __m256i b01 = _mm256_unpackhi_epi64(w0, w1), b23 = _mm256_unpackhi_epi64(w2, w3);
        __m256i ta, tb, r;

        CRC_REDUCE_Y(ta, a01, a23, ka, kb, kc);
        CRC_REDUCE_Y(tb, b01, b23, ka, kb, kc);
        /* the CRC is 32-bit element 2 of each half: gather lanes 0..3 into one XMM */
        r = _mm256_unpackhi_epi32(ta, tb);
        r = _mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storeu_si128((__m128i *)(void *)(out + 4 * g),
                         _mm_xor_si128(_mm256_castsi256_si128(r), zero32));
    }
}

static crc_groups_fn crc_groups = crc_groups_pclmul;

/* ---- the kernel ---- */

typedef struct {
    lane_t   lane[4];
    uint64_t um[10][4], vm[10][4];                  /* window centres, reduced */
    uint64_t prefix[HALF_WINDOW + 1][10][4];
} walk4_t;

static struct {
    uint64_t u[10], v[10];
} TAB4[HALF_WINDOW + 1], TAB4Q;                     /* TAB and TABQ in radix 2^25.5 */

static int walk4_available(void)
{
    return g_cpu.avx2 && g_cpu.pclmul;
}

static void walk4_init(void)
{
    for (int i = 1; i <= HALF_WINDOW; i++) {
        fe_to_limbs25(TAB4[i].u, &TAB[i].u);
        fe_to_limbs25(TAB4[i].v, &TAB[i].v);
    }
    fe_to_limbs25(TAB4Q.u, &TABQ.u);
    fe_to_limbs25(TAB4Q.v, &TABQ.v);
    if (g_cpu.vpclmul)
        crc_groups = crc_groups_vpclmul;
}

static void walk4_set_centre(void *state, unsigned l, const mpoint *c)
{
    walk4_t *w = (walk4_t *)state;
    uint64_t h[10];

    fe_to_limbs25(h, &c->u);
    for (int k = 0; k < 10; k++)
        w->um[k][l] = h[k];
    fe_to_limbs25(h, &c->v);
    for (int k = 0; k < 10; k++)
        w->vm[k][l] = h[k];
}

static void walk4_seed_lane(walk4_t *w, unsigned l)
{
    mpoint centre;

    lane_seed(&w->lane[l], &centre);
    walk4_set_centre(w, l, &centre);
}

static void walk4_seed(void *state)
{
    for (unsigned l = 0; l < 4; l++)
        walk4_seed_lane((walk4_t *)state, l);
}

/* A lane reports at most one key per base: later matches in the window are dropped. */
static NOINLINE void walk4_on_match(walk4_t *w, unsigned *retired, unsigned l, unsigned j,
                                    uint32_t id)
{
    if (*retired & (1u << l))
        return;
    engine_on_match(&w->lane[l], j, id);
    *retired |= 1u << l;
}

static NOINLINE void walk4_dump(const walk_dump *dump, const uint64_t grp[4][4],
                                const uint32_t *id, unsigned j)
{
    for (unsigned l = 0; l < 4; l++) {
        for (int k = 0; k < 4; k++)
            dump->words[l * WINDOW + j][k] = grp[k][l];
        dump->id[l * WINDOW + j] = id[l];
    }
}

AVX2_FN static void walk4_window(void *state, const walk_dump *dump)
{
    walk4_t *w = (walk4_t *)state;
    fe4 um, vm, bias_u, bias_v, e0, e, p, g19, acc, inv, d, t, lam, x, ui, vi, pre;
    __m256i words[4];
    _Alignas(32) uint64_t grp[2][4][4];
    _Alignas(32) uint32_t id[8];
    unsigned retired;

    /*
     * Window constants, all with non-negative limbs:
     *   bias_u = 2p - u_M           u_i - u_M = u_i + bias_u          (3x)
     *   bias_v = 2p - v_M           v_i - v_M = v_i + bias_v          (3x)
     *   e0     = 4p - A - u_M       u(M +/- P_i) = lambda^2 + e0 - u_i
     */
    fe4_load(&um, &w->um[0][0]);
    fe4_load(&vm, &w->vm[0][0]);
    for (int k = 0; k < 10; k++) {
        bias_u.l[k] = V4_SUB(V4_SET1(TWO_P_25[k]), um.l[k]);
        bias_v.l[k] = V4_SUB(V4_SET1(TWO_P_25[k]), vm.l[k]);
        e0.l[k] = V4_SUB(V4_SET1(FOUR_P_25[k]), um.l[k]);
    }
    e0.l[0] = V4_SUB(e0.l[0], V4_SET1(CURVE_A));

    /* Forward pass: prefix[i] = (u_Q - u_M) * prod_{k<=i} (u_k - u_M). */
    fe4_broadcast(&ui, TAB4Q.u);
    for (int k = 0; k < 10; k++)
        p.l[k] = V4_ADD(ui.l[k], bias_u.l[k]);
    fe4_store(&w->prefix[0][0][0], &p);
    for (int i = 1; i <= HALF_WINDOW; i++) {
        fe4_broadcast(&ui, TAB4[i].u);
        for (int k = 0; k < 10; k++)
            d.l[k] = V4_ADD(ui.l[k], bias_u.l[k]);
        fe4_mul19(&g19, &p);
        fe4_mul_pre(&p, &d, &p, &g19);
        fe4_store(&w->prefix[i][0][0], &p);
    }

    /* A lane whose product is zero met a table point: its results are void. */
    fe4_to_words(words, &p);
    retired = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
        V4_OR(V4_OR(words[0], words[1]), V4_OR(words[2], words[3])), _mm256_setzero_si256())));

    fe4_invert(&acc, &p);

    /* Backward pass: one inverse per step, eight keys out. */
    for (int i = HALF_WINDOW; i >= 1; i--) {
        fe4_load(&pre, &w->prefix[i - 1][0][0]);
        fe4_broadcast(&ui, TAB4[i].u);
        fe4_broadcast(&vi, TAB4[i].v);

        fe4_mul19(&g19, &acc);
        fe4_mul_pre(&inv, &pre, &acc, &g19);        /* 1 / (u_i - u_M) */
        for (int k = 0; k < 10; k++)
            d.l[k] = V4_ADD(ui.l[k], bias_u.l[k]);
        fe4_mul_pre(&acc, &d, &acc, &g19);

        fe4_mul19(&g19, &inv);
        for (int k = 0; k < 10; k++)
            e.l[k] = V4_SUB(e0.l[k], ui.l[k]);

        for (int k = 0; k < 10; k++)
            t.l[k] = V4_ADD(vi.l[k], bias_v.l[k]);
        fe4_mul_pre(&lam, &t, &inv, &g19);
        fe4_sq_add_to_words(words, &lam, &e);       /* M + P_i */
        for (int k = 0; k < 4; k++)
            _mm256_storeu_si256((__m256i *)(void *)grp[0][k], words[k]);

        for (int k = 0; k < 10; k++)
            t.l[k] = V4_ADD(vi.l[k], vm.l[k]);
        fe4_mul_pre(&lam, &t, &inv, &g19);
        fe4_sq_add_to_words(words, &lam, &e);       /* M - P_i */
        for (int k = 0; k < 4; k++)
            _mm256_storeu_si256((__m256i *)(void *)grp[1][k], words[k]);

        crc_groups((const uint64_t (*)[4][4])grp, id, 2);

        if (UNLIKELY(dump != NULL)) {
            walk4_dump(dump, (const uint64_t (*)[4])grp[0], id, (unsigned)(HALF_WINDOW + i));
            walk4_dump(dump, (const uint64_t (*)[4])grp[1], id + 4, (unsigned)(HALF_WINDOW - i));
        }
        for (unsigned n = 0; n < 8; n++) {
            if (UNLIKELY(pattern_match(id[n])))
                walk4_on_match(w, &retired, n & 3,
                               (unsigned)(n < 4 ? HALF_WINDOW + i : HALF_WINDOW - i), id[n]);
        }
    }

    /* The centres themselves. */
    fe4_to_words(words, &um);
    for (int k = 0; k < 4; k++)
        _mm256_storeu_si256((__m256i *)(void *)grp[0][k], words[k]);
    crc_groups((const uint64_t (*)[4][4])grp, id, 1);
    if (UNLIKELY(dump != NULL))
        walk4_dump(dump, (const uint64_t (*)[4])grp[0], id, HALF_WINDOW);
    for (unsigned n = 0; n < 4; n++) {
        if (UNLIKELY(pattern_match(id[n])))
            walk4_on_match(w, &retired, n, HALF_WINDOW, id[n]);
    }

    /* Move every centre: M' = M + Q, with acc = 1 / (u_Q - u_M). */
    fe4_broadcast(&ui, TAB4Q.u);
    fe4_broadcast(&vi, TAB4Q.v);
    for (int k = 0; k < 10; k++)
        t.l[k] = V4_ADD(vi.l[k], bias_v.l[k]);
    fe4_mul(&lam, &t, &acc);
    fe4_sq(&x, &lam);
    for (int k = 0; k < 10; k++)
        x.l[k] = V4_ADD(x.l[k], V4_SUB(e0.l[k], ui.l[k]));
    fe4_reduce(&x);                                 /* u' = lambda^2 - A - u_M - u_Q */
    for (int k = 0; k < 10; k++)
        d.l[k] = V4_ADD(um.l[k], V4_SUB(V4_SET1(TWO_P_25[k]), x.l[k]));
    fe4_mul(&t, &d, &lam);
    for (int k = 0; k < 10; k++)
        t.l[k] = V4_ADD(t.l[k], bias_v.l[k]);
    fe4_reduce(&t);                                 /* v' = lambda * (u_M - u') - v_M */
    fe4_store(&w->um[0][0], &x);
    fe4_store(&w->vm[0][0], &t);

    for (unsigned l = 0; l < 4; l++) {
        if ((retired >> l) & 1 || w->lane[l].win + 1 >= LANE_MAX_WINDOWS)
            walk4_seed_lane(w, l);
        else
            w->lane[l].win++;
    }
}

static const backend_t BACKEND_AVX2 = {
    "avx2", 4, sizeof(walk4_t), walk4_available, walk4_seed, walk4_window, walk4_set_centre,
};

#endif /* MTV_X86 */

static const backend_t *const BACKENDS[] = {
    /* usually-fastest first; a tie in calibration goes to the earlier entry */
#ifdef MTV_X86
    &BACKEND_AVX2,
#endif
    &BACKEND_SCALAR,
};

static const backend_t *backend_find(const char *name)
{
    for (size_t i = 0; i < ARRAY_LEN(BACKENDS); i++)
        if (strcmp(name, BACKENDS[i]->name) == 0)
            return BACKENDS[i];
    return NULL;
}

/* ======================================================================== */
/* 11. Worker threads and the run loop                                       */
/* ======================================================================== */

typedef struct {
    _Alignas(64) atomic_ullong keys;        /* candidates this thread has tried   */
    const backend_t *forced;                /* NULL: measure and choose           */
    const backend_t *chosen;                /* the kernel the thread settled on   */
    int cpu_slot;                           /* >= 0: bind the thread to that CPU  */
    thread_t thread;
} worker_t;

static worker_t g_workers[MAX_THREADS];

#define CALIBRATION_SECONDS 0.03

/* Search with one kernel until `seconds` have passed (0: until told to stop). */
static double worker_search(worker_t *wk, const backend_t *be, double seconds)
{
    void *state = xalloc_aligned(be->state_size);
    const unsigned long long per_window = (unsigned long long)be->lanes * WINDOW;
    unsigned long long done = 0;
    double start, elapsed = 0.0;

    memset(state, 0, be->state_size);
    be->seed(state);
    start = now_seconds();
    while (!atomic_load_explicit(&g_run.stop, memory_order_relaxed)) {
        be->window(state, NULL);
        atomic_fetch_add_explicit(&wk->keys, per_window, memory_order_relaxed);
        done += per_window;
        if (seconds > 0.0 && (elapsed = now_seconds() - start) >= seconds)
            break;
    }
    secure_zero(state, be->state_size);
    xfree_aligned(state);
    return elapsed > 0.0 ? (double)done / elapsed : 0.0;
}

/*
 * Which kernel is fastest depends on the core: the performance and
 * efficiency cores of one hybrid CPU can disagree.  Unless a kernel was
 * forced, every thread therefore times each kernel on its own core for a
 * moment and keeps the winner.  The timed windows are ordinary search
 * windows, so no work is wasted.
 */
static void worker_main(void *arg)
{
    worker_t *wk = (worker_t *)arg;
    const backend_t *best = wk->forced;

    if (wk->cpu_slot >= 0)
        os_pin_thread(wk->cpu_slot);
    if (best == NULL) {
        double best_rate = -1.0;
        for (size_t i = 0; i < ARRAY_LEN(BACKENDS); i++) {
            double rate;
            if (!BACKENDS[i]->available())
                continue;
            rate = worker_search(wk, BACKENDS[i], CALIBRATION_SECONDS);
            if (rate > best_rate) {
                best_rate = rate;
                best = BACKENDS[i];
            }
        }
    }
    wk->chosen = best;
    (void)worker_search(wk, best, 0.0);
}

static void format_duration(char *out, size_t n, double seconds)
{
    if (seconds < 1.0)
        snprintf(out, n, "%.0f ms", seconds * 1e3);
    else if (seconds < 120.0)
        snprintf(out, n, "%.1f s", seconds);
    else if (seconds < 7200.0)
        snprintf(out, n, "%.1f min", seconds / 60.0);
    else
        snprintf(out, n, "%.1f h", seconds / 3600.0);
}

typedef struct {
    unsigned long long keys;
    double seconds;
    int threads;
    char kernels[64];                       /* e.g. "avx2 x16, scalar x16" */
} run_result;

/*
 * Run nthreads workers until the wanted hits are found, the time or key
 * budget runs out (0 = unlimited), or the user presses Ctrl-C.  be == NULL
 * lets every thread choose its own kernel; pin binds thread i to CPU i.
 */
static run_result engine_run(const backend_t *be, int nthreads, int pin, double time_limit,
                             unsigned long long key_limit, int show_progress)
{
    run_result res = {0, 0.0, 0, ""};
    const double expected = pattern_expected_keys();
    double start, last_report;

    atomic_store(&g_run.stop, 0);
    g_run.hits = 0;
    g_run.progress_shown = 0;

    for (int i = 0; i < nthreads; i++) {
        worker_t *wk = &g_workers[i];
        atomic_store(&wk->keys, 0);
        wk->forced = be;
        wk->chosen = NULL;
        wk->cpu_slot = pin ? i : -1;
        wk->thread.fn = worker_main;
        wk->thread.arg = wk;
        if (thread_start(&wk->thread) != 0)
            break;
        res.threads++;
    }
    if (res.threads == 0)
        die("could not start any worker thread");

    start = last_report = now_seconds();
    for (;;) {
        double now;

        sleep_ms(20);
        if (g_interrupted)
            atomic_store(&g_run.stop, 1);
        res.keys = 0;
        for (int i = 0; i < res.threads; i++)
            res.keys += atomic_load_explicit(&g_workers[i].keys, memory_order_relaxed);
        now = now_seconds();
        res.seconds = now - start;
        if ((key_limit != 0 && res.keys >= key_limit) ||
            (time_limit > 0.0 && res.seconds >= time_limit))
            atomic_store(&g_run.stop, 1);
        if (atomic_load(&g_run.stop))
            break;

        if (show_progress && now - last_report >= 1.0) {
            const double rate = (double)res.keys / res.seconds;
            char mean[32], p95[32];
            format_duration(mean, sizeof mean, expected / rate);
            format_duration(p95, sizeof p95, 3.0 * expected / rate);
            output_lock();
            fprintf(stderr, "\r%.3g keys, %.1f M/s, %.0f%% chance of a hit by now"
                            " (mean %s, 95%% within %s)   ",
                    (double)res.keys, rate / 1e6,
                    100.0 * (1.0 - exp(-(double)res.keys / expected)), mean, p95);
            fflush(stderr);
            g_run.progress_shown = 1;
            output_unlock();
            last_report = now;
        }
    }

    for (int i = 0; i < res.threads; i++)
        thread_join(&g_workers[i].thread);
    res.keys = 0;
    for (int i = 0; i < res.threads; i++)
        res.keys += atomic_load(&g_workers[i].keys);
    res.seconds = now_seconds() - start;
    if (g_run.progress_shown)
        fprintf(stderr, "\r%78s\r", "");
    g_run.progress_shown = 0;

    for (size_t b = 0; b < ARRAY_LEN(BACKENDS); b++) {
        int n = 0;
        for (int i = 0; i < res.threads; i++)
            n += g_workers[i].chosen == BACKENDS[b];
        if (n > 0) {
            const size_t used = strlen(res.kernels);
            snprintf(res.kernels + used, sizeof res.kernels - used, "%s%s x%d",
                     used ? ", " : "", BACKENDS[b]->name, n);
        }
    }
    return res;
}

/* ======================================================================== */
/* 12. Self-tests                                                            */
/* ======================================================================== */

static int t_failures;
static int t_verbose;

static void t_check(int ok, const char *fmt, ...)
{
    va_list ap;

    if (!ok)
        t_failures++;
    if (!t_verbose && ok)
        return;
    va_start(ap, fmt);
    printf("  %s  ", ok ? "ok  " : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static void t_x25519(const char *name, const char *k_hex, const char *u_hex, const char *want_hex)
{
    uint8_t k[32], u[32], want[32], got[32];

    hex_decode(k, k_hex, 32);
    hex_decode(u, u_hex, 32);
    hex_decode(want, want_hex, 32);
    x25519_ref(got, k, u);
    t_check(memcmp(got, want, 32) == 0, "reference X25519: %s", name);
}

static void test_reference(int full)
{
    static const char nine[] = "0900000000000000000000000000000000000000000000000000000000000000";
    uint8_t k[32] = {9}, u[32] = {9}, r[32], want[32];
    const int rounds = full ? 1000 : 1;

    t_check(crc32_ref((const uint8_t *)"123456789", 9) == 0xcbf43926u,
            "reference CRC-32: check value cbf43926");

    t_x25519("RFC 7748 5.2 vector 1",
             "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
             "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
             "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
    t_x25519("RFC 7748 5.2 vector 2",
             "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
             "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
             "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957");
    t_x25519("RFC 7748 6.1 Alice's public key",
             "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a", nine,
             "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
    t_x25519("RFC 7748 6.1 Bob's public key",
             "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb", nine,
             "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");
    t_x25519("RFC 7748 6.1 shared secret",
             "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
             "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
             "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");

    for (int i = 1; i <= rounds; i++) {
        x25519_ref(r, k, u);
        memcpy(u, k, 32);
        memcpy(k, r, 32);
        if (i == 1) {
            hex_decode(want, "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079", 32);
            t_check(memcmp(k, want, 32) == 0, "reference X25519: RFC 7748 5.2 after 1 iteration");
        }
    }
    if (full) {
        hex_decode(want, "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51", 32);
        t_check(memcmp(k, want, 32) == 0, "reference X25519: RFC 7748 5.2 after 1000 iterations");
    }

    /* A published Meshtastic key: private key, public key, node ID. */
    hex_decode(k, "484a47ae2975b17d04b3b75f03d51eb377ea4fa4ea930da9e0339a1f80d22348", 32);
    hex_decode(want, "82050a2d4230fae10ebc9e1d902bae8f30914e9b06a4b24ba8214b554188a347", 32);
    x25519_ref(r, k, BASE_U);
    t_check(memcmp(r, want, 32) == 0, "Meshtastic vector: public key");
    t_check(crc32_ref(r, 32) == 0xdc805a49u, "Meshtastic vector: node ID !dc805a49");
}

static void random_field_bytes(uint8_t b[32], int round)
{
    static const char *const edge[] = {
        "0000000000000000000000000000000000000000000000000000000000000000",   /* 0       */
        "0100000000000000000000000000000000000000000000000000000000000000",   /* 1       */
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",   /* p - 1   */
        "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",   /* p       */
        "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",   /* p + 1   */
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",   /* 2^255-1 */
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",   /* bit 255 */
        "ffffffffffff0700000000000000000000000000000000000000000000000000",   /* 2^51-1  */
    };
    if (round < (int)ARRAY_LEN(edge))
        hex_decode(b, edge[round], 32);
    else
        os_random(b, 32);
}

static void test_field(int full)
{
    const int rounds = full ? 2000 : 64;
    int bad_mul = 0, bad_sq = 0, bad_inv = 0, bad_addsub = 0, bad_canon = 0, bad_finish = 0;

    for (int i = 0; i < rounds; i++) {
        uint8_t ab[32], bb[32], want[32], got[32];
        gf ga, gb, gr;
        fe fa, fb, fr, ft;

        random_field_bytes(ab, i);
        random_field_bytes(bb, (i * 7 + 3) % (rounds + 8));
        gf_unpack(ga, ab);
        gf_unpack(gb, bb);
        fe_from_bytes(&fa, ab);
        fe_from_bytes(&fb, bb);

        gf_pack(want, ga);
        fe_to_bytes(got, &fa);
        bad_canon += memcmp(want, got, 32) != 0;

        gf_mul(gr, ga, gb);
        gf_pack(want, gr);
        fe_mul(&fr, &fa, &fb);
        fe_to_bytes(got, &fr);
        bad_mul += memcmp(want, got, 32) != 0;

        /* the same product from loose operands: (a + b) * (a - b) = a^2 - b^2 */
        fe_add_lazy(&fr, &fa, &fb);
        fe_sub_lazy(&ft, &fa, &fb);
        fe_mul(&fr, &fr, &ft);
        gf_add(gr, ga, gb);
        gf_sub(gb, ga, gb);
        gf_mul(gr, gr, gb);
        gf_pack(want, gr);
        fe_to_bytes(got, &fr);
        bad_addsub += memcmp(want, got, 32) != 0;

        gf_mul(gr, ga, ga);
        gf_pack(want, gr);
        fe_sq(&fr, &fa);
        fe_to_bytes(got, &fr);
        bad_sq += memcmp(want, got, 32) != 0;

        if (i < (full ? 64 : 10)) {
            gf_invert(gr, ga);
            gf_pack(want, gr);
            fe_invert(&fr, &fa);
            fe_to_bytes(got, &fr);
            bad_inv += memcmp(want, got, 32) != 0;
        }
    }
    /*
     * a^2 + e straight to canonical words.  The limbs of e are drawn from
     * values around every threshold the shortcut depends on, so both its
     * fast exit and its fallback are exercised.
     */
    for (int i = 0; i < 40 * rounds; i++) {
        static const uint64_t pick[] = {
            0, 1, 18, 19, MASK51 - 19, MASK51 - 18, MASK51 - 1, MASK51, MASK51 + 1, MASK51 + 19,
            2 * MASK51, 2 * MASK51 + 2, ((uint64_t)1 << 53) - 1, 0x123456789abcdULL,
        };
        uint8_t ab[32], sel[5];
        uint64_t want[4], got[4];
        fe fa, fe_, ft;

        random_field_bytes(ab, i % 3 == 0 ? 0 : i);
        fe_from_bytes(&fa, ab);
        os_random(sel, sizeof sel);
        for (int k = 0; k < 5; k++)
            fe_.v[k] = pick[sel[k] % ARRAY_LEN(pick)];
        fe_sq(&ft, &fa);
        fe_add_lazy(&ft, &ft, &fe_);
        fe_to_words(want, &ft);
        fe_sq_add_to_words(got, &fa, &fe_);
        bad_finish += memcmp(want, got, sizeof want) != 0;
    }
    t_check(bad_finish == 0, "field: square-add-encode shortcut, fast exit and fallback");
    t_check(bad_canon == 0, "field: canonical encoding matches the reference (%d values)", rounds);
    t_check(bad_mul == 0, "field: multiplication matches the reference");
    t_check(bad_addsub == 0, "field: lazy add/sub feeding a multiplication");
    t_check(bad_sq == 0, "field: squaring matches the reference");
    t_check(bad_inv == 0, "field: inversion matches the reference");
}

static void test_curve(int full)
{
    const int rounds = full ? 24 : 2;
    int bad = 0;

    for (int i = 0; i < rounds; i++) {
        uint8_t k[32], want[32], got[32];
        mpoint r;

        os_random(k, 32);
        k[0] &= 248;
        k[31] &= 127;
        k[31] |= 64;
        if (mont_scalarmult_base(&r, k) != 0) {
            bad++;
            continue;
        }
        fe_to_bytes(got, &r.u);
        x25519_ref(want, k, BASE_U);
        bad += memcmp(want, got, 32) != 0 || !mont_on_curve(&r);
    }
    t_check(bad == 0, "curve: affine double-and-add matches the reference ladder");
}

static void test_crc(int full)
{
    static const struct {
        const char *name;
        crc_fn fn;
        int *available;
    } engines[] = {
        {"slicing-by-8", crc_msgs_table, NULL},
#ifdef MTV_ARM_CRC
        {"ARMv8 CRC32", crc_msgs_armv8, NULL},
#endif
#ifdef MTV_X86
        {"PCLMULQDQ", crc_msgs_pclmul, &g_cpu.pclmul},
#endif
    };
    const int rounds = full ? 4096 : 64;

    for (size_t e = 0; e < ARRAY_LEN(engines); e++) {
        int bad = 0;

        if (engines[e].available != NULL && !*engines[e].available)
            continue;
        for (int i = 0; i < rounds; i++) {
            uint8_t bytes[2][32];
            uint64_t msg[2][4];
            uint32_t got[2];

            random_field_bytes(bytes[0], i);
            os_random(bytes[1], 32);
            if (i == 8)
                memset(bytes[1], 0, 32);
            for (int m = 0; m < 2; m++)
                for (int k = 0; k < 4; k++)
                    msg[m][k] = load64_le(bytes[m] + 8 * k);
            engines[e].fn((const uint64_t (*)[4])msg, got, 2);
            bad += got[0] != crc32_ref(bytes[0], 32) || got[1] != crc32_ref(bytes[1], 32);
        }
        t_check(bad == 0, "CRC-32 engine %s matches the reference", engines[e].name);
    }
}

static void test_pattern(void)
{
    static const char *const good[] = {"a", "dead", "!c0de", "DC80*a", "dead,beef,cafe",
                                       "!dead,!beef", "deadbeef", "????beef", "0000000?", "dead,dead"};
    static const char *const bad[] = {"", "!", "g", "dead,", "dead,abc", "123456789", "****",
                                      "00000000", "00000003", "ffffffff", "dead beef"};
    const pattern_t saved = g_pat;
    int wrong = 0;

    for (size_t i = 0; i < ARRAY_LEN(good); i++)
        wrong += pattern_parse(good[i]) != NULL;
    for (size_t i = 0; i < ARRAY_LEN(bad); i++)
        wrong += pattern_parse(bad[i]) == NULL;
    t_check(wrong == 0, "patterns: parser accepts and rejects the right forms");

    wrong = pattern_parse("dc80*a") != NULL;
    wrong += !pattern_match(0xdc805a49u) || pattern_match(0xdc805b49u) || pattern_match(0xdc815a49u);
    wrong += g_pat.fixed_digits != 5 || g_pat.count != 1;
    wrong += pattern_parse("dead,beef,dead") != NULL || g_pat.count != 2;
    wrong += !pattern_match(0xdead0000u) || !pattern_match(0xbeefffffu) || pattern_match(0xcafe0000u);
    wrong += pattern_parse("0000000?") != NULL;
    wrong += pattern_match(0u) || pattern_match(3u) || !pattern_match(4u) || !pattern_match(15u);
    t_check(wrong == 0, "patterns: matching, wildcards, lists, reserved IDs");
    g_pat = saved;
}

/*
 * Run a backend for a few windows with matching disabled and compare what it
 * computed with the reference ladder: every key of the first window when
 * `full`, otherwise the keys at the window edges and the centre.
 */
static void test_walk(const backend_t *be, int full)
{
    const unsigned lanes = be->lanes;
    const size_t slots = (size_t)lanes * WINDOW;
    const pattern_t saved = g_pat;
    void *state = xalloc_aligned(be->state_size);
    walk_dump dump;
    lane_t before[MAX_LANES];
    unsigned checked = 0, bad = 0;

    dump.words = malloc(slots * sizeof *dump.words);
    dump.id = malloc(slots * sizeof *dump.id);
    if (dump.words == NULL || dump.id == NULL)
        die("out of memory");
    memset(state, 0, be->state_size);
    pattern_set_none();
    be->seed(state);

    for (int win = 0; win < 3; win++) {
        memcpy(before, state, lanes * sizeof(lane_t));
        memset(dump.words, 0xa5, slots * sizeof *dump.words);
        be->window(state, &dump);
        for (unsigned l = 0; l < lanes; l++) {
            for (unsigned j = 0; j < WINDOW; j++) {
                const unsigned i = j > HALF_WINDOW ? j - HALF_WINDOW : HALF_WINDOW - j;
                const int edge = i <= 1 || i == HALF_WINDOW;
                uint8_t sk[32], want[32], got[32];

                if (!(edge || (full && win == 0)))
                    continue;
                scalar_offset(sk, before[l].base, before[l].win * WINDOW + j);
                x25519_ref(want, sk, BASE_U);
                words_to_bytes(got, dump.words[l * WINDOW + j]);
                bad += memcmp(want, got, 32) != 0 || dump.id[l * WINDOW + j] != crc32_ref(want, 32);
                checked++;
                secure_zero(sk, sizeof sk);
            }
        }
    }
    t_check(bad == 0, "backend %s: %u keys equal the reference ladder and CRC", be->name, checked);

    g_pat = saved;
    secure_zero(state, be->state_size);
    xfree_aligned(state);
    free(dump.words);
    free(dump.id);
}

/*
 * A window centre that coincides with a table point makes one denominator
 * zero.  That never happens by chance (probability about 2^-240), so force
 * it: the lane must be abandoned and reseeded, the other lanes must carry on
 * undisturbed, and the reseeded lane must be right on the next window.
 */
static void test_exceptional_centre(const backend_t *be)
{
    const unsigned lanes = be->lanes, victim = be->lanes - 1;
    const size_t slots = (size_t)lanes * WINDOW;
    const pattern_t saved = g_pat;
    void *state = xalloc_aligned(be->state_size);
    walk_dump dump;
    lane_t before[MAX_LANES], after[MAX_LANES];
    unsigned bad = 0;

    dump.words = malloc(slots * sizeof *dump.words);
    dump.id = malloc(slots * sizeof *dump.id);
    if (dump.words == NULL || dump.id == NULL)
        die("out of memory");
    memset(state, 0, be->state_size);
    pattern_set_none();
    be->seed(state);
    be->set_centre(state, victim, &TAB[7]);

    for (int pass = 0; pass < 2; pass++) {
        memcpy(before, state, lanes * sizeof(lane_t));
        be->window(state, &dump);
        memcpy(after, state, lanes * sizeof(lane_t));
        for (unsigned l = 0; l < lanes; l++) {
            static const unsigned probe[] = {0, HALF_WINDOW - 1, HALF_WINDOW, HALF_WINDOW + 1,
                                             WINDOW - 1};
            if (pass == 0 && l == victim) {
                bad += memcmp(before[l].base, after[l].base, 32) == 0 || after[l].win != 0;
                continue;
            }
            bad += after[l].win != before[l].win + 1;
            for (size_t n = 0; n < ARRAY_LEN(probe); n++) {
                uint8_t sk[32], want[32], got[32];
                scalar_offset(sk, before[l].base, before[l].win * WINDOW + probe[n]);
                x25519_ref(want, sk, BASE_U);
                words_to_bytes(got, dump.words[l * WINDOW + probe[n]]);
                bad += memcmp(want, got, 32) != 0;
                secure_zero(sk, sizeof sk);
            }
        }
    }
    t_check(bad == 0, "backend %s: a centre on a table point is abandoned cleanly", be->name);

    g_pat = saved;
    secure_zero(state, be->state_size);
    xfree_aligned(state);
    free(dump.words);
    free(dump.id);
}

/* Search for real, capture the keys, and check each one from scratch. */
static void test_search(const backend_t *be)
{
    enum { WANT = 6 };
    const pattern_t saved = g_pat;
    const out_mode_t saved_mode = g_run.mode;
    const uint64_t saved_wanted = g_run.hits_wanted;
    hit_t hits[WANT];
    int bad = 0;

    memset(hits, 0, sizeof hits);
    bad += pattern_parse("a5,5a") != NULL;
    g_run.mode = OUT_CAPTURE;
    g_run.capture = hits;
    g_run.hits_wanted = WANT;
    (void)engine_run(be, 1, 0, 30.0, 0, 0);
    bad += g_run.hits != WANT;

    for (int i = 0; i < WANT; i++) {
        uint8_t pk[32];
        const uint8_t *sk = hits[i].secret;
        uint32_t id;

        x25519_ref(pk, sk, BASE_U);
        id = crc32_ref(pk, 32);
        bad += memcmp(pk, hits[i].public_key, 32) != 0 || id != hits[i].node_id;
        bad += !((id >> 24) == 0xa5 || (id >> 24) == 0x5a);
        bad += (sk[0] & 7) != 0 || (sk[31] & 0xc0) != 0x40;         /* clamped */
        for (int k = 0; k < i; k++)
            bad += memcmp(sk, hits[k].secret, 32) == 0;
    }
    t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
            "backend %s: end-to-end search returns %d verified, distinct keys", be->name, WANT);

    secure_zero(hits, sizeof hits);
    g_pat = saved;
    g_run.mode = saved_mode;
    g_run.hits_wanted = saved_wanted;
    g_run.capture = NULL;
}

#ifdef MTV_X86
/* The AVX2 field code and CRC engines against their scalar counterparts. */
AVX2_FN static void test_avx2(int full)
{
    const int rounds = full ? 600 : 32;
    int bad_mul = 0, bad_sq = 0, bad_words = 0, bad_inv = 0, bad_crc = 0;

    for (int i = 0; i < rounds; i++) {
        uint64_t la[10][4], lb[10][4], lr[10][4];
        uint64_t grp[2][4][4];
        uint32_t ids[8];
        fe a[4], b[4], want, got;
        fe4 va, vb, vr, vt;

        for (int l = 0; l < 4; l++) {
            uint8_t bytes[32];
            uint64_t h[10];
            random_field_bytes(bytes, 4 * i + l);
            fe_from_bytes(&a[l], bytes);
            fe_canonical(&a[l]);
            fe_to_limbs25(h, &a[l]);
            for (int k = 0; k < 10; k++)
                la[k][l] = h[k];
            random_field_bytes(bytes, (4 * i + l) * 5 + 2);
            fe_from_bytes(&b[l], bytes);
            fe_canonical(&b[l]);
            fe_to_limbs25(h, &b[l]);
            for (int k = 0; k < 10; k++)
                lb[k][l] = h[k];
        }
        fe4_load(&va, &la[0][0]);
        fe4_load(&vb, &lb[0][0]);

#define T_COMPARE(counter, vec, expr)                                   \
        do {                                                            \
            fe4_store(&lr[0][0], &(vec));                                      \
            for (int l = 0; l < 4; l++) {                               \
                uint64_t h[10];                                         \
                for (int k = 0; k < 10; k++)                            \
                    h[k] = lr[k][l];                                    \
                fe_from_limbs25(&got, h);                               \
                expr;                                                   \
                counter += !fe_equal(&got, &want);                      \
            }                                                           \
        } while (0)

        fe4_mul(&vr, &va, &vb);
        T_COMPARE(bad_mul, vr, fe_mul(&want, &a[l], &b[l]));

        /* operands at the 3x bound: (a + 2p) * (b + 2p) */
        for (int k = 0; k < 10; k++) {
            vt.l[k] = V4_ADD(va.l[k], V4_SET1(TWO_P_25[k]));
            vr.l[k] = V4_ADD(vb.l[k], V4_SET1(TWO_P_25[k]));
        }
        fe4_mul(&vr, &vt, &vr);
        T_COMPARE(bad_mul, vr, fe_mul(&want, &a[l], &b[l]));

        fe4_sq(&vr, &va);
        T_COMPARE(bad_sq, vr, fe_sq(&want, &a[l]));

        if (i < 12) {
            fe4_invert(&vr, &va);
            T_COMPARE(bad_inv, vr, fe_invert(&want, &a[l]));
        }
#undef T_COMPARE

        /* both group CRC engines on the encodings of a and b */
        for (int l = 0; l < 4; l++) {
            uint64_t ref[4];
            fe_to_words(ref, &a[l]);
            for (int k = 0; k < 4; k++)
                grp[0][k][l] = ref[k];
            fe_to_words(ref, &b[l]);
            for (int k = 0; k < 4; k++)
                grp[1][k][l] = ref[k] ^ (i == 3 ? ~(uint64_t)0 : 0);   /* also a non-field pattern */
        }
        for (int engine = 0; engine < 2; engine++) {
            if (engine == 1 && !g_cpu.vpclmul)
                continue;
            (engine ? crc_groups_vpclmul : crc_groups_pclmul)((const uint64_t (*)[4][4])grp, ids, 2);
            for (int g = 0; g < 2; g++) {
                for (int l = 0; l < 4; l++) {
                    uint8_t bytes[32];
                    const uint64_t one[4] = {grp[g][0][l], grp[g][1][l], grp[g][2][l], grp[g][3][l]};
                    words_to_bytes(bytes, one);
                    bad_crc += ids[4 * g + l] != crc32_ref(bytes, 32);
                }
            }
        }
    }

    /*
     * Canonical encoding of x and of f^2 + e, with limbs drawn from values
     * around every threshold FE4_FINISH tests, so that its fast exit and its
     * slow path both run many times.
     */
    for (int i = 0; i < 60 * rounds; i++) {
        static const uint64_t even[] = {
            0, 1, 19, 0x3ffffeb, 0x3ffffec, 0x3ffffed, 0x3fffffe, 0x3ffffff, 0x4000000,
            0x4000013, 0x7fffffe, 0xffffffc, 0x1fffffff, 0x2345678,
        };
        static const uint64_t odd[] = {
            0, 1, 0x1fffffe, 0x1ffffff, 0x2000000, 0x3fffffe, 0x7fffffc, 0xfffffff, 0x1234567,
        };
        uint64_t lf[10][4], le[10][4], words[4][4];
        uint8_t sel[10][4], zero_f;
        fe4 vf, ve;
        __m256i w[4];

        os_random(sel, sizeof sel);
        os_random(&zero_f, 1);
        for (int l = 0; l < 4; l++) {
            uint8_t bytes[32];
            uint64_t h[10];
            fe a;
            random_field_bytes(bytes, (zero_f >> l) & 1 ? 0 : 99);     /* f = 0 isolates e */
            fe_from_bytes(&a, bytes);
            fe_canonical(&a);
            fe_to_limbs25(h, &a);
            for (int k = 0; k < 10; k++) {
                lf[k][l] = h[k];
                le[k][l] = (k & 1) ? odd[sel[k][l] % ARRAY_LEN(odd)]
                                   : even[sel[k][l] % ARRAY_LEN(even)];
            }
        }
        fe4_load(&vf, &lf[0][0]);
        fe4_load(&ve, &le[0][0]);

        for (int mode = 0; mode < 2; mode++) {
            if (mode == 0)
                fe4_to_words(w, &ve);
            else
                fe4_sq_add_to_words(w, &vf, &ve);
            for (int k = 0; k < 4; k++)
                _mm256_storeu_si256((__m256i *)(void *)words[k], w[k]);
            for (int l = 0; l < 4; l++) {
                uint64_t h[10], ref[4];
                fe a, e, t;
                for (int k = 0; k < 10; k++)
                    h[k] = lf[k][l];
                fe_from_limbs25(&a, h);
                for (int k = 0; k < 10; k++)
                    h[k] = le[k][l];
                fe_from_limbs25(&e, h);
                fe_carry(&e);
                if (mode == 0) {
                    t = e;
                } else {
                    fe_sq(&t, &a);
                    fe_add_lazy(&t, &t, &e);
                }
                fe_to_words(ref, &t);
                for (int k = 0; k < 4; k++)
                    bad_words += words[k][l] != ref[k];
            }
        }
    }

    t_check(bad_mul == 0, "avx2 field: multiplication matches the scalar code (reduced, 3x)");
    t_check(bad_sq == 0, "avx2 field: squaring matches the scalar code");
    t_check(bad_inv == 0, "avx2 field: inversion matches the scalar code");
    t_check(bad_words == 0, "avx2 field: canonical encoding and square-add-encode, both paths");
    t_check(bad_crc == 0, "avx2 CRC-32 engines (PCLMULQDQ%s) match the reference",
            g_cpu.vpclmul ? ", VPCLMULQDQ" : "");
}
#endif

/* Returns the number of failed checks.  `only` restricts the backend tests. */
static int self_test(int full, const backend_t *only)
{
    t_failures = 0;
    t_verbose = full;

    test_reference(full);
    test_field(full);
    test_curve(full);
    test_crc(full);
    test_pattern();
#ifdef MTV_X86
    if (walk4_available())
        test_avx2(full);
#endif
    for (size_t i = 0; i < ARRAY_LEN(BACKENDS); i++) {
        const backend_t *be = BACKENDS[i];
        if ((only != NULL && be != only) || !be->available())
            continue;
        test_walk(be, full);
        if (full) {
            test_exceptional_centre(be);
            test_search(be);
        }
    }
    return t_failures;
}

/* ======================================================================== */
/* 13. Command line                                                          */
/* ======================================================================== */

static void usage(FILE *out)
{
    fprintf(out,
        "mtvanity %s - vanity node IDs for Meshtastic firmware 2.8+\n"
        "%s\n"
        "\n"
        "usage: mtvanity PATTERN [options]\n"
        "       mtvanity --benchmark [SECONDS]\n"
        "       mtvanity --self-test\n"
        "\n"
        "PATTERN  1 to 8 hex digits, with an optional leading '!'.  Fewer than 8\n"
        "         digits is a prefix: dead means !dead????.  The characters * ? . x\n"
        "         stand for any digit: 'dc80*a'.  A comma-separated list matches any\n"
        "         of its entries: 'dead,beef,cafe'.\n"
        "\n"
        "options:\n"
        "  --hits N          stop after N keys (default 1)\n"
        "  --threads N       worker threads (default: every CPU)\n"
        "  --max-keys N      give up after trying about N keys\n"
        "  --backend NAME    auto (default: each thread times the kernels on its own\n"
        "                    core and keeps the fastest), avx2 or scalar\n"
        "  --no-pin          do not bind threads to CPUs\n"
        "  --json            print each key as one JSON object per line\n"
        "  --quiet           no progress or summary on stderr\n"
        "  --benchmark [S]   measure every kernel for S seconds each (default 5)\n"
        "  --self-test       run every known-answer and cross-check test\n"
        "  --version, --help\n"
        "\n"
        "Keys go to stdout, everything else to stderr.  Install a key with\n"
        "  meshtastic --set security.private_key \"base64:...\"\n"
        "after backing up the old one: a new key is a new node identity.\n",
        MTV_VERSION, MTV_ATTRIBUTION);
}

/* Parse a non-negative integer argument strictly; returns -1 on any error. */
static long long parse_count(const char *s)
{
    char *end;
    long long v;

    if (s == NULL || *s < '0' || *s > '9')
        return -1;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno != 0 || *end != '\0')
        return -1;
    return v;
}

static void run_benchmark(const backend_t *forced, int nthreads, int pin, double seconds)
{
    pattern_set_none();
    g_run.mode = OUT_CAPTURE;
    g_run.hits_wanted = 0;
    printf("benchmark: %d thread%s, %.0f s per line\n", nthreads, nthreads == 1 ? "" : "s", seconds);

    /* each kernel alone, then (i == count) every thread choosing for itself */
    for (size_t i = 0; i <= ARRAY_LEN(BACKENDS) && !g_interrupted; i++) {
        const backend_t *be = i < ARRAY_LEN(BACKENDS) ? BACKENDS[i] : NULL;
        const char *name = be ? be->name : "auto";
        run_result r;
        char full_id[32];

        if (forced != NULL && be != forced)
            continue;
        if (be != NULL && !be->available()) {
            printf("  %-7s not supported by this CPU\n", name);
            continue;
        }
        r = engine_run(be, nthreads, pin, seconds, 0, 0);
        format_duration(full_id, sizeof full_id, 4294967296.0 * r.seconds / (double)r.keys);
        printf("  %-7s %8.1f M keys/s   full 8-digit ID in about %s", name,
               (double)r.keys / r.seconds / 1e6, full_id);
        if (be == NULL)
            printf("   [%s]", r.kernels);
        printf("\n");
        fflush(stdout);
    }
}

int main(int argc, char **argv)
{
    const char *pattern = NULL, *backend_name = NULL, *err;
    const backend_t *be;
    int nthreads = 0, quiet = 0, json = 0, bench = 0, selftest = 0, pin = 1;
    double bench_seconds = 5.0;
    unsigned long long key_limit = 0;
    long long hits_wanted = 1;
    run_result res;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = i + 1 < argc ? argv[i + 1] : NULL;

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(a, "--version") == 0) {
            printf("mtvanity %s\n%s\nLicense: MIT\n", MTV_VERSION, MTV_ATTRIBUTION);
            return 0;
        } else if (strcmp(a, "--threads") == 0) {
            const long long v = parse_count(next);
            if (v < 1 || v > MAX_THREADS)
                die("--threads needs a number between 1 and 1024");
            nthreads = (int)v;
            i++;
        } else if (strcmp(a, "--hits") == 0) {
            hits_wanted = parse_count(next);
            if (hits_wanted < 1)
                die("--hits needs a positive number");
            i++;
        } else if (strcmp(a, "--max-keys") == 0) {
            const long long v = parse_count(next);
            if (v < 1)
                die("--max-keys needs a positive number");
            key_limit = (unsigned long long)v;
            i++;
        } else if (strcmp(a, "--backend") == 0) {
            if (next == NULL)
                die("--backend needs a name");
            backend_name = next;
            i++;
        } else if (strcmp(a, "--no-pin") == 0) {
            pin = 0;
        } else if (strcmp(a, "--json") == 0) {
            json = 1;
        } else if (strcmp(a, "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(a, "--self-test") == 0) {
            selftest = 1;
        } else if (strcmp(a, "--benchmark") == 0) {
            bench = 1;
            if (next != NULL && parse_count(next) > 0) {
                bench_seconds = (double)parse_count(next);
                i++;
            }
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "mtvanity: unknown option %s (try --help)\n", a);
            return 2;
        } else if (pattern == NULL) {
            pattern = a;
        } else {
            die("only one pattern is allowed; join alternatives with commas");
        }
    }

    if (!selftest && !bench) {
        if (pattern == NULL) {
            usage(stderr);
            return 2;
        }
        err = pattern_parse(pattern);       /* report mistakes before doing any work */
        if (err != NULL) {
            fprintf(stderr, "mtvanity: bad pattern \"%s\": %s\n", pattern, err);
            return 2;
        }
    }

    os_disable_core_dumps();
    output_lock_init();
#ifdef MTV_X86
    cpu_detect();
#endif
    crc_init();
    if (curve_init() != 0)
        die("internal error: curve set-up failed its own checks");
#ifdef MTV_X86
    walk4_init();
#endif
    pattern_set_none();

    if (selftest) {
        int failed;
        printf("mtvanity %s self-test\n", MTV_VERSION);
        failed = self_test(1, NULL);
        if (failed)
            printf("%d check(s) FAILED\n", failed);
        else
            printf("all checks passed\n");
        return failed ? 1 : 0;
    }

    /* be == NULL means "auto": each thread picks its own fastest kernel */
    be = NULL;
    if (backend_name != NULL && strcmp(backend_name, "auto") != 0) {
        be = backend_find(backend_name);
        if (be == NULL) {
            fprintf(stderr, "mtvanity: unknown --backend; choose one of: auto");
            for (size_t i = 0; i < ARRAY_LEN(BACKENDS); i++)
                fprintf(stderr, " %s", BACKENDS[i]->name);
            fprintf(stderr, "\n");
            return 2;
        }
        if (!be->available())
            die("this CPU does not support the requested --backend");
    }
    if (nthreads == 0)
        nthreads = os_cpu_count();
    if (nthreads > MAX_THREADS)
        nthreads = MAX_THREADS;
    /* binding only makes sense when there is one thread for every CPU */
    pin = pin && nthreads == os_cpu_count();
    signal(SIGINT, on_interrupt);

    /* Never search with arithmetic that has not just been checked. */
    if (self_test(0, be) != 0)
        die("the start-up self-test failed; refusing to generate keys");

    if (bench) {
        run_benchmark(be, nthreads, pin, bench_seconds);
        return 0;
    }

    err = pattern_parse(pattern);
    if (err != NULL)
        die("internal error: the pattern no longer parses");
    g_run.mode = json ? OUT_JSON : OUT_TEXT;
    g_run.hits_wanted = (uint64_t)hits_wanted;
    if (!quiet) {
        fprintf(stderr, "searching %d target%s with %d fixed digit%s: about %.3g keys per hit, "
                        "%d thread%s\n",
                g_pat.count, g_pat.count == 1 ? "" : "s",
                g_pat.fixed_digits, g_pat.fixed_digits == 1 ? "" : "s",
                pattern_expected_keys(), nthreads, nthreads == 1 ? "" : "s");
    }

    res = engine_run(be, nthreads, pin, 0.0, key_limit, !quiet);

    if (!quiet) {
        fprintf(stderr, "%" PRIu64 " key%s found; %.3g tried in %.2f s (%.1f M/s; kernels: %s)%s\n",
                g_run.hits, g_run.hits == 1 ? "" : "s", (double)res.keys, res.seconds,
                (double)res.keys / res.seconds / 1e6, res.kernels[0] ? res.kernels : "none",
                g_interrupted ? "; interrupted" : "");
    }
    if (atomic_load(&g_run.faults) != 0)
        fprintf(stderr, "mtvanity: warning: %llu candidate(s) failed verification and were "
                        "discarded; please report this\n", atomic_load(&g_run.faults));
    return g_run.hits >= g_run.hits_wanted ? 0 : 3;
}
