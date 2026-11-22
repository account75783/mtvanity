/*
 * mtvanity - vanity node IDs and public keys for Meshtastic firmware 2.8+
 *
 * Authored by Claude, assisted by a human.
 *
 * Since firmware 2.8 a Meshtastic node number is the CRC-32 of the node's
 * 32-byte X25519 public key, displayed as "!%08x".  This program searches
 * for private keys whose node ID, public key, or both look the way you ask.
 * (Older firmware takes the number from the MAC address; a key changes
 * nothing there.)
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
 *   Add -s -static for a release binary that needs no shared libraries.
 *   A 64-bit target is required (the field arithmetic uses 128-bit
 *   integers).  On Windows at most 64 logical CPUs are used.
 *
 * USE
 *   ./mtvanity --self-test       check every component against a reference
 *   ./mtvanity --benchmark       keys per second of each kernel on this machine
 *   ./mtvanity c0de              a key whose node ID begins !c0de
 *   ./mtvanity 'id:*beef'        a key whose node ID ends in beef
 *   ./mtvanity key:Mesh          a key whose public key (base64) begins Mesh
 *   ./mtvanity id:c0de+key:Me    one key that does both
 *   ./mtvanity dead beef cafe    three keys, searched for together
 *   ./mtvanity --help            patterns and options
 *   Section 7 below describes the pattern language in full.
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
 *   8. One comparison per key.  Whatever is asked for is boiled down to
 *      the bits every match must share; a key is examined closely only if
 *      those bits fit.  Any number of targets of the same shape cost one
 *      hashed look-up, and when only public keys are constrained the
 *      CRC-32 is not computed at all.
 *
 * SECURITY
 *   - Keys come only from the operating system's CSPRNG.
 *   - Every hit is recomputed by an independent implementation (a plain
 *     RFC 7748 ladder on 16-bit limbs and a bitwise CRC) before it is
 *     shown, and must agree with the fast path in both public key and node
 *     ID.  A fault in the fast path can therefore lose a key but can never
 *     print a wrong one.  The same reference checks the fast path every
 *     time the program starts.
 *   - A base scalar is retired after its first hit, so two keys this
 *     program prints are never a small known offset apart.
 *   - Private keys are written to stdout only, never to disk or logs, and
 *     straight to the operating system, so that no C library buffer keeps
 *     a copy.  Core dumps (Linux) and crash dumps (Windows, best effort)
 *     are disabled.
 *   - The search is deliberately NOT constant-time: it is an offline tool.
 *     Do not run it on a machine you share with people you do not trust.
 *   - A vanity key is an ordinary random key with a few chosen public
 *     properties, and anyone can make another that shares them: a node ID
 *     or the start of a public key never proves who a node is.  Pick an ID
 *     nobody on your mesh already uses.
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
#  include <sys/ioctl.h>
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

#define MTV_VERSION     "V1.1"
#define MTV_REVISION    4           /* of the source, so that a stale build can be told apart */
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

/* printf-style checking, in the dialect of whichever printf the platform's headers select */
#ifdef __MINGW_PRINTF_FORMAT
#  define PRINTF_LIKE(f, a) __attribute__((format(__MINGW_PRINTF_FORMAT, f, a)))
#else
#  define PRINTF_LIKE(f, a) __attribute__((format(printf, f, a)))
#endif

__extension__ typedef unsigned __int128 u128;

/* ======================================================================== */
/* 1. Output and small utilities                                             */
/* ======================================================================== */

/* Wipe memory in a way the optimiser may not remove. */
static void secure_zero(void *p, size_t n)
{
    volatile unsigned char *v = (volatile unsigned char *)p;
    while (n--)
        *v++ = 0;
}

/*
 * Everything this program prints goes through os_write, one call per
 * message, and never through the C library's streams.  On Windows those
 * streams hand an unbuffered console its text one character at a time; with
 * every core busy searching, each of those tiny writes has to wait its turn
 * and a line is seen being typed out from left to right.  Writing directly
 * also leaves no copy of a private key behind in a stream buffer.
 */
enum { TO_STDOUT = 1, TO_STDERR = 2 };

/* Write all n bytes of text.  Returns 0, or -1 if the stream will not take them. */
static int os_write(int to, const char *text, size_t n)
{
#ifdef _WIN32
    const HANDLE h = GetStdHandle(to == TO_STDOUT ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
    char chunk[8192];       /* with "\r\n" for "\n", as the C library's text mode writes it */
    int rc = 0;

    while (n > 0 && rc == 0) {
        DWORD len = 0, at = 0;

        while (n > 0 && len < sizeof chunk - 1) {
            if (*text == '\n')
                chunk[len++] = '\r';
            chunk[len++] = *text++;
            n--;
        }
        while (at < len && rc == 0) {
            DWORD done = 0;
            if (!WriteFile(h, chunk + at, len - at, &done, NULL) || done == 0)
                rc = -1;
            at += done;
        }
        secure_zero(chunk, len);    /* it may have been a private key */
    }
    return rc;
#else
    while (n > 0) {
        const ssize_t r = write(to == TO_STDOUT ? STDOUT_FILENO : STDERR_FILENO, text, n);

        if (r < 0 && errno == EINTR)
            continue;
        if (r < 0 && errno == EAGAIN) {     /* a terminal someone left in non-blocking mode */
            const struct timespec nap = {0, 1000000};
            nanosleep(&nap, NULL);
            continue;
        }
        if (r <= 0)
            return -1;
        text += r;
        n -= (size_t)r;
    }
    return 0;
#endif
}

/* Format a message and write it in one piece.  Messages are short; a longer one is cut off. */
static int os_vprintf(int to, const char *fmt, va_list ap)
{
    char text[4096];
    const int n = vsnprintf(text, sizeof text, fmt, ap);

    if (n < 0)
        return -1;
    return os_write(to, text, (size_t)n < sizeof text ? (size_t)n : sizeof text - 1);
}

PRINTF_LIKE(1, 2)
static void out_printf(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    (void)os_vprintf(TO_STDOUT, fmt, ap);
    va_end(ap);
}

PRINTF_LIKE(1, 2)
static void err_printf(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    (void)os_vprintf(TO_STDERR, fmt, ap);
    va_end(ap);
}

static void die(const char *msg)
{
    err_printf("mtvanity: %s\n", msg);
    exit(2);
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

static const char HEX_ALPHABET[] = "0123456789abcdef";
static const char BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char ascii_lower(char c)
{
    return c >= 'A' && c <= 'Z' ? (char)(c - 'A' + 'a') : c;
}

/* The other case of a letter; any other character as it is. */
static char ascii_swap_case(char c)
{
    if (c >= 'A' && c <= 'Z')
        return (char)(c - 'A' + 'a');
    if (c >= 'a' && c <= 'z')
        return (char)(c - 'a' + 'A');
    return c;
}

static void hex_encode(char *out, const uint8_t *in, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = HEX_ALPHABET[in[i] >> 4];
        out[2 * i + 1] = HEX_ALPHABET[in[i] & 15];
    }
    out[2 * n] = '\0';
}

/* Standard base64 with padding; out needs 4 * ceil(n / 3) + 1 bytes. */
static void base64_encode(char *out, const uint8_t *in, size_t n)
{
    const char *const tab = BASE64_ALPHABET;
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
    SYSTEM_INFO info;
    int n = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask))
        for (; process_mask != 0; process_mask >>= 1)
            n += (int)(process_mask & 1);
    if (n > 0)
        return n;
    /* no usable mask (the process is spread over several processor groups) */
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 1;
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

/*
 * Let whatever is waiting for this CPU run now.  Every search thread calls
 * this between two windows, a few thousand times a second.
 *
 * Windows makes a thread that becomes ready wait until the time slice of an
 * equally important running thread is over, tens of milliseconds each time.
 * With a search thread on every CPU the rest of the machine then runs in
 * slow motion, the console first of all: with 32 threads on 32 CPUs the
 * status line was seen to land on a new row at each update, and with 24
 * threads it stayed in place.  A yield is one system call when nobody is
 * waiting, far below a thousandth of the time a window takes.  It does
 * mean that the search gives way to other busy programs instead of taking
 * turns with them.  Linux lets a thread that wakes up in ahead of one that
 * has been running, so nothing is needed there.
 */
static void os_yield(void)
{
#ifdef _WIN32
    (void)SwitchToThread();
#endif
}

/*
 * The width of the terminal behind stderr, in columns; 0 if stderr is a file
 * or a pipe.  Asked afresh for every progress line, since windows get resized.
 */
static int os_stderr_columns(void)
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;

    /* this succeeds for a console only; text wraps at the width of its buffer */
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &info))
        return 0;
    return info.dwSize.X > 0 ? info.dwSize.X : 80;
#else
    struct winsize size;

    if (!isatty(STDERR_FILENO))
        return 0;
    return ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 ? size.ws_col : 80;
#endif
}

#ifdef _WIN32
/*
 * Last-chance exception handler.  Ending the process here keeps Windows
 * Error Reporting from attaching to it and writing a dump of memory that
 * holds private keys.  The exit status is the exception code, as it would
 * have been without the handler.
 */
static LONG WINAPI on_fatal_exception(EXCEPTION_POINTERS *info)
{
    const UINT code = (info != NULL && info->ExceptionRecord != NULL)
                          ? (UINT)info->ExceptionRecord->ExceptionCode
                          : 3u;
    TerminateProcess(GetCurrentProcess(), code);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

/*
 * A crash must not leave private keys in a core file or a crash dump.
 * Best effort on Windows: the error mode and the handler cover ordinary
 * faults, but a fail-fast abort raised by the system itself bypasses both.
 */
static void os_disable_core_dumps(void)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(on_fatal_exception);
#else
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
    secure_zero(e, sizeof(gf));
    secure_zero(f, sizeof(gf));
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
/* 7. Targets: what a key has to look like                                   */
/* ======================================================================== */

/*
 * The Meshtastic apps show a key as two strings:
 *
 *   node ID      "!" and 8 hex digits: the CRC-32 of the public key
 *   public key   43 base64 characters and a closing "=", or 64 hex digits
 *
 * A pattern says, for some positions of one of these strings, which symbols
 * are acceptable there.  A target is one or two patterns that a single key
 * must satisfy; every target of a run is given its own key.
 *
 * Pattern syntax, the same for every string:
 *   0-9 a-z ...  the character itself
 *   .            any one character
 *   *            any run of characters, at most once: "dead*" (or plain
 *                "dead") is "starts with", "*beef" is "ends with", and
 *                "de*ef" is both
 *   ?d ?l ?u     a digit, a lower-case letter, an upper-case letter
 *   ?h ?H        one of 0-9a-f, one of 0-9A-F
 *   ?s           + or /
 *   ?a           any one character
 * The ?x forms are the built-in charsets of hashcat masks, and as there a ?
 * always takes the next character with it.  Hex has no case, so in the two
 * hex strings ?l and ?u both mean a letter a-f.
 */

enum { FORM_ID, FORM_KEY, FORM_KEYHEX, FORM_COUNT };

typedef struct {
    const char *tag;                /* introduces the pattern on the command line */
    const char *noun;
    const char *alphabet;           /* symbol value -> character                  */
    unsigned    bits;               /* per symbol                                 */
    unsigned    length;             /* symbols in the whole string                */
    int         caseless;           /* a letter stands for both of its cases      */
} form_t;

static const form_t FORMS[FORM_COUNT] = {
    {"id:",     "node ID",          HEX_ALPHABET,    4, 8,  1},
    {"key:",    "public key",       BASE64_ALPHABET, 6, 43, 0},
    {"keyhex:", "public key (hex)", HEX_ALPHABET,    4, 64, 1},
};

typedef struct {
    uint8_t  pos;                   /* position in the string            */
    uint64_t allowed;               /* bit v set: symbol v is acceptable */
} rule_t;

#define PATTERN_TEXT 160

typedef struct {
    int      form;
    unsigned count;                 /* rules; 0: the pattern is absent   */
    rule_t   rule[64];
    char     text[PATTERN_TEXT];    /* canonical spelling, for display   */
} pattern_t;

#define TARGET_TEXT (2 * PATTERN_TEXT + 64)

typedef struct {
    pattern_t id, key;
    uint32_t  id_mask, id_value;            /* node-ID bits all matches have in common   */
    uint64_t  key_mask[4], key_value[4];    /* the same for the public key, word by word */
    double    chance;                       /* that a random key matches                 */
    char      text[TARGET_TEXT];            /* "node ID !dead* + public key Mesh*"       */
    uint64_t  wanted, found;                /* keys; found is guarded by the output lock */
    uint32_t  first_id;                     /* node ID of its first key, for the summary */
} target_t;

static target_t g_target[MAX_TARGETS];
static int      g_targets;

/* The firmware reserves node numbers below 4 and the broadcast address. */
static const uint32_t RESERVED_IDS[] = {0, 1, 2, 3, 0xffffffffu};

static inline int node_id_reserved(uint32_t id)
{
    return id < 4u || id == 0xffffffffu;
}

/*
 * Which bit of the underlying number carries bit s of the symbol at a
 * position: a bit of the 32-bit node ID, or bit n of the public key read as
 * a 256-bit little-endian integer (bit n of the key is bit n % 64 of word
 * n / 64 and bit n % 8 of byte n / 8).  -1 for the two padding bits base64
 * appends to 256 bits of data.
 */
static int form_bit(int form, unsigned pos, unsigned s)
{
    switch (form) {
    case FORM_ID:
        return (int)(28 - 4 * pos + s);
    case FORM_KEYHEX:                   /* two digits per byte, the high one first */
        return (int)(8 * (pos / 2) + (pos % 2 ? 0u : 4u) + s);
    default: {                          /* the bytes' bits, high bit first, six at a time */
        const unsigned n = 6 * pos + 5 - s;
        return n < 256 ? (int)(8 * (n / 8) + 7 - n % 8) : -1;
    }
    }
}

/*
 * The symbols that can occur at a position at all.  Bit 255 of a public key
 * is always clear and base64 padding is zero, so the last characters of a
 * key string cannot take every value.
 */
static uint64_t form_possible(int form, unsigned pos)
{
    const unsigned bits = FORMS[form].bits;
    uint64_t set = 0;

    for (unsigned v = 0; v < (1u << bits); v++) {
        int ok = 1;
        for (unsigned s = 0; s < bits; s++) {
            const int b = form_bit(form, pos, s);
            if (((v >> s) & 1) && form != FORM_ID && (b < 0 || b == 255))
                ok = 0;
        }
        if (ok)
            set |= (uint64_t)1 << v;
    }
    return set;
}

/* The symbol a character stands for, or -1. */
static int form_symbol(int form, char c)
{
    const form_t *f = &FORMS[form];

    for (unsigned v = 0; f->alphabet[v] != '\0'; v++)
        if (f->alphabet[v] == c || (f->caseless && f->alphabet[v] == ascii_lower(c)))
            return (int)v;
    return -1;
}

/* The symbol at a position of a public key's string. */
static inline unsigned key_symbol(int form, const uint8_t pk[32], unsigned pos)
{
    if (form == FORM_KEYHEX)
        return (pk[pos / 2] >> (pos % 2 ? 0 : 4)) & 15u;
    {
        const unsigned n = 6 * pos, byte = n / 8;
        const unsigned next = byte + 1 < 32 ? (unsigned)pk[byte + 1] : 0u;
        const unsigned two = ((unsigned)pk[byte] << 8) | next;
        return (two >> (10 - n % 8)) & 63u;
    }
}

static int pattern_matches_key(const pattern_t *p, const uint8_t pk[32])
{
    for (unsigned i = 0; i < p->count; i++)
        if (!((p->rule[i].allowed >> key_symbol(p->form, pk, p->rule[i].pos)) & 1))
            return 0;
    return 1;
}

static int pattern_matches_id(const pattern_t *p, uint32_t id)
{
    for (unsigned i = 0; i < p->count; i++)
        if (!((p->rule[i].allowed >> ((id >> (28 - 4 * p->rule[i].pos)) & 15u)) & 1))
            return 0;
    return 1;
}

/* Does this key belong to the target?  A key with a reserved node ID belongs to none. */
static int target_matches(const target_t *t, const uint8_t pk[32], uint32_t id)
{
    return pattern_matches_key(&t->key, pk) && pattern_matches_id(&t->id, id) &&
           !node_id_reserved(id);
}

/* ---- parsing ---- */

static char g_parse_error[320];

PRINTF_LIKE(1, 2)
static const char *parse_error(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(g_parse_error, sizeof g_parse_error, fmt, ap);
    va_end(ap);
    return g_parse_error;
}

/* hashcat's built-in charsets, as far as they make sense for these strings */
static const struct {
    char        letter;
    const char *chars;
} CLASSES[] = {
    {'d', "0123456789"},
    {'l', "abcdefghijklmnopqrstuvwxyz"},
    {'u', "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    {'h', "0123456789abcdef"},
    {'H', "0123456789ABCDEF"},
    {'s', "+/"},
};

/* If `letter` names a class, store its symbols for this form (possibly none) and return 1. */
static int class_set(int form, char letter, uint64_t *set)
{
    if (letter == 'a') {
        *set = FORMS[form].bits == 6 ? ~(uint64_t)0 : 0xffffu;
        return 1;
    }
    for (size_t k = 0; k < ARRAY_LEN(CLASSES); k++) {
        if (CLASSES[k].letter != letter)
            continue;
        *set = 0;
        for (const char *c = CLASSES[k].chars; *c != '\0'; c++) {
            const int v = form_symbol(form, *c);
            if (v >= 0)
                *set |= (uint64_t)1 << v;
        }
        return 1;
    }
    return 0;
}

/* "'e' cannot be character 43 of the 43 in a public key: that one is always one of ..." */
static const char *pattern_impossible(int form, unsigned pos, uint64_t possible, const char *what)
{
    const form_t *f = &FORMS[form];
    char list[65];
    size_t n = 0;

    for (unsigned v = 0; f->alphabet[v] != '\0'; v++)
        if ((possible >> v) & 1)
            list[n++] = f->alphabet[v];
    list[n] = '\0';
    return parse_error("'%s' cannot be character %u of the %u in a %s: that one is always one "
                       "of %s", what, pos + 1, f->length, f->noun, list);
}

#define NO_STAR 0xffffu

/*
 * Parse the n characters at s as a pattern for one string.  Returns NULL on
 * success or a message saying what is wrong.
 */
static const char *pattern_parse(pattern_t *p, int form, const char *s, size_t n, int ignore_case)
{
    const form_t *f = &FORMS[form];
    const uint64_t every = f->bits == 6 ? ~(uint64_t)0 : 0xffffu;
    uint64_t tok[64];                   /* the symbols each character of the pattern accepts */
    char tok_text[64][3];               /* and how it was written, for messages              */
    char shown[PATTERN_TEXT - 16];      /* at most 2 * 64 + 2 characters and the terminator  */
    size_t used = 0, i = 0;
    unsigned count = 0, star = NO_STAR, prefix;
    int closed = 0, folded = 0;

    memset(p, 0, sizeof *p);
    p->form = form;
    if (form == FORM_ID) {
        shown[used++] = '!';
        if (n > 0 && s[0] == '!')
            i = 1;
    }

    for (; i < n; i++) {
        const char c = s[i];
        uint64_t set;

        if (closed)
            return parse_error("nothing can follow the '=' that ends a public key");
        if (c == '*') {
            if (star != NO_STAR)
                return parse_error("a pattern can hold only one *");
            star = count;
            shown[used++] = '*';
            continue;
        }
        if (c == '=' && form == FORM_KEY) {
            closed = 1;
            continue;
        }
        if (count == f->length)
            return parse_error("too long: a %s has %u characters", f->noun, f->length);

        if (c == '?') {
            if (i + 1 == n || !class_set(form, s[i + 1], &set))
                return parse_error("a ? needs one of d l u h H s a after it, as in ?d for a "
                                   "digit; use . for any one character");
            if (set == 0)
                return parse_error("?%c matches nothing in a %s", s[i + 1], f->noun);
            shown[used++] = '?';
            shown[used++] = s[++i];
            memcpy(tok_text[count], shown + used - 2, 2);
            tok_text[count][2] = '\0';
        } else if (c == '.') {
            set = every;
            shown[used++] = '.';
        } else {
            const int v = form_symbol(form, c);
            if (v < 0) {
                if (form == FORM_ID)
                    return parse_error("'%c' is not a hex digit: a node ID pattern uses 0-9, a-f "
                                       "and . for any digit (a public-key pattern starts with "
                                       "key:)", c);
                return parse_error("'%c' cannot occur in a %s", c, f->noun);
            }
            set = (uint64_t)1 << v;
            if (ignore_case && !f->caseless) {
                const int other = form_symbol(form, ascii_swap_case(c));
                if (other >= 0 && other != v) {
                    set |= (uint64_t)1 << other;
                    folded = 1;
                }
            }
            shown[used++] = f->caseless ? f->alphabet[v] : c;
        }
        if (c != '?') {
            tok_text[count][0] = shown[used - 1];
            tok_text[count][1] = '\0';
        }
        tok[count++] = set;
    }

    if (count == 0)
        return parse_error("a %s pattern needs at least one character", f->noun);
    if (star == NO_STAR) {
        if (closed && count != f->length)
            return parse_error("'=' is where a public key ends: put a * in front "
                               "(key:*%.*s) to match the end", (int)n, s);
        if (count < f->length)
            shown[used++] = '*';        /* a short pattern is a prefix */
        prefix = count;
    } else {
        prefix = star;
    }
    if (form == FORM_KEY && (count == f->length || prefix < count))
        shown[used++] = '=';            /* the pattern reaches the end of the key */
    shown[used] = '\0';

    /* characters before the * count from the start, those after it from the end */
    for (unsigned k = 0; k < count; k++) {
        const unsigned pos = k < prefix ? k : f->length - (count - k);
        const uint64_t possible = form_possible(form, pos);
        const uint64_t allowed = tok[k] & possible;

        if (allowed == 0)
            return pattern_impossible(form, pos, possible, tok_text[k]);
        if (allowed != possible) {
            p->rule[p->count].pos = (uint8_t)pos;
            p->rule[p->count].allowed = allowed;
            p->count++;
        }
    }
    if (p->count == 0)
        return parse_error("a %s pattern must pin down at least one character", f->noun);

    snprintf(p->text, sizeof p->text, "%s%s", shown, folded ? " (any case)" : "");
    return NULL;
}

/* Which form's tag begins the n characters at s; -1 if none does. */
static int tag_form(const char *s, size_t n, size_t *len)
{
    for (int form = 0; form < FORM_COUNT; form++) {
        const char *tag = FORMS[form].tag;
        const size_t t = strlen(tag);
        size_t k = 0;

        if (n < t)
            continue;
        while (k < t && ascii_lower(s[k]) == tag[k])
            k++;
        if (k == t) {
            *len = t;
            return form;
        }
    }
    return -1;
}

/* The bits on which every symbol of a rule agrees, and their values. */
static void rule_common(unsigned bits, const rule_t *r, unsigned *mask, unsigned *value)
{
    unsigned all_and = (1u << bits) - 1, all_or = 0;

    for (unsigned v = 0; v < (1u << bits); v++) {
        if ((r->allowed >> v) & 1) {
            all_and &= v;
            all_or |= v;
        }
    }
    *mask = ~(all_and ^ all_or) & ((1u << bits) - 1);
    *value = all_and & *mask;
}

/* Derive everything the search needs from a target's two patterns. */
static const char *target_finish(target_t *t)
{
    char text[TARGET_TEXT];
    double ids = 1.0;
    unsigned size[8];

    /* the bits all matching keys share: the kernels filter on these */
    for (unsigned i = 0; i < t->id.count; i++) {
        unsigned mask, value;
        rule_common(4, &t->id.rule[i], &mask, &value);
        for (unsigned s = 0; s < 4; s++) {
            const int b = form_bit(FORM_ID, t->id.rule[i].pos, s);
            if ((mask >> s) & 1) {
                t->id_mask |= (uint32_t)1 << b;
                t->id_value |= (uint32_t)((value >> s) & 1) << b;
            }
        }
    }
    for (unsigned i = 0; i < t->key.count; i++) {
        const unsigned bits = FORMS[t->key.form].bits;
        unsigned mask, value;
        rule_common(bits, &t->key.rule[i], &mask, &value);
        for (unsigned s = 0; s < bits; s++) {
            const int b = form_bit(t->key.form, t->key.rule[i].pos, s);
            if (((mask >> s) & 1) && b >= 0) {
                t->key_mask[b / 64] |= (uint64_t)1 << (b % 64);
                t->key_value[b / 64] |= (uint64_t)((value >> s) & 1) << (b % 64);
            }
        }
    }

    /* the node IDs the pattern accepts, less the reserved ones among them */
    for (unsigned pos = 0; pos < 8; pos++)
        size[pos] = 16;
    for (unsigned i = 0; i < t->id.count; i++)
        size[t->id.rule[i].pos] = (unsigned)__builtin_popcountll(t->id.rule[i].allowed);
    for (unsigned pos = 0; pos < 8; pos++)
        ids *= size[pos];
    for (size_t r = 0; r < ARRAY_LEN(RESERVED_IDS); r++)
        ids -= pattern_matches_id(&t->id, RESERVED_IDS[r]);
    if (ids <= 0.0)
        return parse_error("that node ID is reserved by the firmware "
                           "(!00000000 to !00000003 and !ffffffff)");
    t->chance = ids / 4294967296.0;
    for (unsigned i = 0; i < t->key.count; i++) {
        const uint64_t possible = form_possible(t->key.form, t->key.rule[i].pos);
        t->chance *= (double)__builtin_popcountll(t->key.rule[i].allowed) /
                     (double)__builtin_popcountll(possible);
    }

    /* composed in a buffer of its own: the pieces live in the same structure as the result */
    if (t->id.count != 0 && t->key.count != 0)
        snprintf(text, sizeof text, "%s %s + %s %s", FORMS[FORM_ID].noun, t->id.text,
                 FORMS[t->key.form].noun, t->key.text);
    else if (t->id.count != 0)
        snprintf(text, sizeof text, "%s %s", FORMS[FORM_ID].noun, t->id.text);
    else
        snprintf(text, sizeof text, "%s %s", FORMS[t->key.form].noun, t->key.text);
    memcpy(t->text, text, strlen(text) + 1);
    return NULL;
}

/*
 * Parse one target: "dead", "id:*beef", "key:Mesh", "id:c0de+key:Me?d".  An
 * untagged pattern is a node ID.  Returns NULL or a message.
 */
static const char *target_parse(target_t *t, const char *s, size_t n, int ignore_case)
{
    size_t start = 0;

    memset(t, 0, sizeof *t);
    for (;;) {
        size_t end = start, taglen = 0, unused;
        const char *err;
        pattern_t *p;
        int form;

        /* a pattern runs up to a '+' that is followed by another tagged pattern */
        while (end < n && !(s[end] == '+' && tag_form(s + end + 1, n - end - 1, &unused) >= 0))
            end++;
        form = tag_form(s + start, end - start, &taglen);
        if (form < 0) {
            form = FORM_ID;
            taglen = 0;
        }
        p = form == FORM_ID ? &t->id : &t->key;
        if (p->count != 0)
            return parse_error("a target takes one node ID pattern and one public key pattern");
        err = pattern_parse(p, form, s + start + taglen, end - start - taglen, ignore_case);
        if (err != NULL)
            return err;
        if (end == n)
            break;
        start = end + 1;
    }
    return target_finish(t);
}

static void targets_clear(void)
{
    g_targets = 0;
}

/*
 * Add the targets of one command-line argument: patterns separated by commas.
 * The Windows command prompt does not treat single quotes as quotes, so an
 * argument that arrives wrapped in them is taken without them.
 */
static const char *targets_add(const char *arg, int ignore_case)
{
    const char *p = arg, *end = arg + strlen(arg);

    if (end - p >= 2 && p[0] == '\'' && end[-1] == '\'') {
        p++;
        end--;
    }
    for (;;) {
        const char *comma = memchr(p, ',', (size_t)(end - p));
        const size_t n = (size_t)((comma != NULL ? comma : end) - p);
        const char *err;

        if (n == 0)
            return parse_error("empty target");
        if (g_targets == MAX_TARGETS)
            return parse_error("too many targets (the limit is %d)", MAX_TARGETS);
        err = target_parse(&g_target[g_targets], p, n, ignore_case);
        if (err != NULL)
            return err;
        g_targets++;
        if (comma == NULL)
            return NULL;
        p = comma + 1;
    }
}

/* Ask for `each` keys per target; returns how many keys that is in all. */
static uint64_t targets_want(uint64_t each)
{
    for (int k = 0; k < g_targets; k++) {
        g_target[k].wanted = each;
        g_target[k].found = 0;
        g_target[k].first_id = 0;
    }
    return each * (uint64_t)g_targets;
}

/*
 * The average number of keys that must be tried until every target has all
 * the keys it still wants.  Reads `found`: the caller holds the output lock
 * or no worker is running.
 *
 * A target that wants k more keys at chance c is treated as one event of
 * rate c / k; the wait for all targets is then the maximum of independent
 * exponentials, whose mean is the integral of 1 - prod(1 - exp(-r_i x)).
 * It is evaluated on a geometric grid, since the rates can differ by many
 * orders of magnitude.
 */
static double targets_expected_keys(void)
{
    double rate[MAX_TARGETS], lo = 0.0, hi = 0.0, total = 0.0, x, last;
    const double step = 1.01;
    int n = 0;

    for (int k = 0; k < g_targets; k++) {
        const target_t *t = &g_target[k];
        if (t->found >= t->wanted)
            continue;
        rate[n] = t->chance / (double)(t->wanted - t->found);
        lo = n == 0 || rate[n] < lo ? rate[n] : lo;
        hi = n == 0 || rate[n] > hi ? rate[n] : hi;
        n++;
    }
    if (n == 0)
        return 0.0;
    if (n == 1)
        return 1.0 / rate[0];

    x = 1e-6 / hi;                      /* below this the integrand is 1 */
    total = x;
    last = 1.0;
    while (x < 60.0 / lo) {
        const double next = x * step;
        double all = 1.0, now;
        for (int i = 0; i < n; i++)
            all *= -expm1(-rate[i] * next);
        now = 1.0 - all;
        total += 0.5 * (last + now) * (next - x);
        last = now;
        x = next;
    }
    return total;
}

/* ---- the filter the kernels apply to every key ---- */

/*
 * Examining every target exactly for every key would cost more than making
 * the key, so the kernels first apply a cheap necessary condition.  Each
 * target contributes the bits that all of its matches share inside one
 * 64-bit quantity, a word of the public key or the node ID; targets that
 * look at the same bits form a group, and a group is tested with a single
 * comparison or a single hashed look-up.  Only keys that pass a group are
 * examined exactly.
 *
 * A node ID costs a CRC-32 per key.  When no group reads it the kernels skip
 * that work, and the CRC is computed just for the keys that get through.
 *
 * A filter never changes once it is published.  When a target has all its
 * keys a new filter without it is published, and the old one stays valid
 * until the run is over, so the kernels need no lock: they pick up the
 * current filter once per window.
 */
#define SRC_ID            4         /* sources 0..3 are the words of the public key */
#define FILTER_HASH_BITS  12
#define FILTER_SLOTS      (1u << FILTER_HASH_BITS)

typedef struct {
    uint64_t mask;                  /* the bits of the source this group looks at   */
    uint64_t value;                 /* what they must equal, for a group of one     */
    uint8_t  source;
    uint8_t  single;
    uint16_t first, count;          /* its targets: member[first .. first + count)  */
    uint64_t bitmap[FILTER_SLOTS / 64];     /* hashed values of a larger group      */
} fgroup_t;

typedef struct {
    unsigned groups;
    int      need_id;               /* some group reads the node ID                 */
    fgroup_t group[MAX_TARGETS];
    uint8_t  member[MAX_TARGETS];           /* target numbers, group by group       */
    uint64_t member_value[MAX_TARGETS];     /* the masked source of that target     */
} filter_t;

/*
 * Wants no key, yet costs exactly what a search for one node ID costs: the
 * benchmark and the self-tests run under it.  Its group has no members, so a
 * key that happens to pass is dropped at once.
 */
static const filter_t FILTER_NONE = {
    .groups = 1,
    .need_id = 1,
    .group = {{.mask = 0xffffffffu, .value = 0xffffffffu, .source = SRC_ID, .single = 1}},
};

static _Atomic(const filter_t *) g_filter = &FILTER_NONE;

static struct {
    filter_t *version[MAX_TARGETS + 1];     /* every filter published during this run */
    int       versions;
} g_filters;

static ALWAYS_INLINE const filter_t *filter_current(void)
{
    return atomic_load_explicit(&g_filter, memory_order_acquire);
}

static ALWAYS_INLINE unsigned filter_slot(uint64_t masked)
{
    return (unsigned)((masked * 0x9e3779b97f4a7c15ULL) >> (64 - FILTER_HASH_BITS));
}

static ALWAYS_INLINE int fgroup_pass(const fgroup_t *g, uint64_t source)
{
    const uint64_t v = source & g->mask;

    if (g->single)
        return v == g->value;
    {
        const unsigned slot = filter_slot(v);
        return (int)((g->bitmap[slot >> 6] >> (slot & 63)) & 1);
    }
}

/* May the key with these words (and this node ID, if the filter reads one) be wanted? */
static ALWAYS_INLINE int filter_pass(const filter_t *f, const uint64_t w[4], uint32_t id)
{
    for (unsigned g = 0; g < f->groups; g++) {
        const fgroup_t *grp = &f->group[g];
        if (fgroup_pass(grp, grp->source == SRC_ID ? (uint64_t)id : w[grp->source]))
            return 1;
    }
    return 0;
}

/* Build the filter for the targets that still want keys. */
static void filter_build(filter_t *f)
{
    struct {
        int active, word, key_bits, id_bits, use_id;
        unsigned group;
    } c[MAX_TARGETS];
    unsigned next[MAX_TARGETS];
    int need_id = 0;

    memset(f, 0, sizeof *f);
    memset(c, 0, sizeof c);

    /*
     * Which quantity should each target be filtered on?  A key word is free;
     * the node ID is worth its CRC only for a target that has no public-key
     * pattern, or one too loose to stop most keys.  Once some target needs
     * the ID anyway, every target uses whichever of its two filters is the
     * tighter.
     */
    for (int k = 0; k < g_targets; k++) {
        const target_t *t = &g_target[k];

        if (t->found >= t->wanted)
            continue;
        c[k].active = 1;
        for (int w = 1; w < 4; w++)
            if (__builtin_popcountll(t->key_mask[w]) > __builtin_popcountll(t->key_mask[c[k].word]))
                c[k].word = w;
        c[k].key_bits = __builtin_popcountll(t->key_mask[c[k].word]);
        c[k].id_bits = __builtin_popcount(t->id_mask);
        c[k].use_id = t->id.count != 0 && (t->key.count == 0 || c[k].key_bits < 6);
        need_id |= c[k].use_id;
    }
    for (int k = 0; k < g_targets && need_id; k++) {
        const target_t *t = &g_target[k];
        if (c[k].active && t->id.count != 0 && t->key.count != 0)
            c[k].use_id = c[k].id_bits > c[k].key_bits;
    }
    need_id = 0;                        /* that may have left no group reading the ID */
    for (int k = 0; k < g_targets; k++)
        need_id |= c[k].active && c[k].use_id;
    f->need_id = need_id;

    /* targets with the same source and mask share a group */
    for (int k = 0; k < g_targets; k++) {
        const target_t *t = &g_target[k];
        uint64_t mask;
        uint8_t source;
        unsigned g = 0;

        if (!c[k].active)
            continue;
        source = c[k].use_id ? SRC_ID : (uint8_t)c[k].word;
        mask = c[k].use_id ? t->id_mask : t->key_mask[c[k].word];
        while (g < f->groups && !(f->group[g].source == source && f->group[g].mask == mask))
            g++;
        if (g == f->groups) {
            f->group[g].source = source;
            f->group[g].mask = mask;
            f->groups++;
        }
        f->group[g].count++;
        c[k].group = g;
    }

    /* lay the members out group by group */
    for (unsigned g = 0, at = 0; g < f->groups; g++) {
        f->group[g].first = (uint16_t)at;
        next[g] = at;
        at += f->group[g].count;
    }
    for (int k = 0; k < g_targets; k++) {
        const target_t *t = &g_target[k];
        if (!c[k].active)
            continue;
        f->member[next[c[k].group]] = (uint8_t)k;
        f->member_value[next[c[k].group]] = c[k].use_id ? t->id_value : t->key_value[c[k].word];
        next[c[k].group]++;
    }
    for (unsigned g = 0; g < f->groups; g++) {
        fgroup_t *grp = &f->group[g];
        grp->single = grp->count == 1;
        grp->value = f->member_value[grp->first];
        for (unsigned m = grp->first; m < (unsigned)grp->first + grp->count; m++) {
            const unsigned slot = filter_slot(f->member_value[m]);
            grp->bitmap[slot >> 6] |= (uint64_t)1 << (slot & 63);
        }
    }
}

/* Back to the filter that matches nothing.  No worker may be running. */
static void filter_reset(void)
{
    atomic_store(&g_filter, &FILTER_NONE);
    for (int i = 0; i < g_filters.versions; i++)
        xfree_aligned(g_filters.version[i]);
    g_filters.versions = 0;
}

/*
 * Build a filter for the targets still wanted and hand it to the kernels.
 * The caller holds the output lock, or no worker is running.  At most one
 * filter is published per finished target, plus the first.
 */
static void filter_publish(void)
{
    filter_t *f;

    if (g_filters.versions == (int)ARRAY_LEN(g_filters.version))
        die("internal error: more filters than targets");
    f = (filter_t *)xalloc_aligned(sizeof *f);
    filter_build(f);
    g_filters.version[g_filters.versions++] = f;
    atomic_store_explicit(&g_filter, f, memory_order_release);
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
    int      target;                /* the target this key was found for       */
} hit_t;

typedef enum { OUT_TEXT, OUT_JSON, OUT_CAPTURE } out_mode_t;

static struct {
    out_mode_t  mode;
    uint64_t    hits_wanted;        /* over all targets                        */
    uint64_t    hits;               /* guarded by the output lock              */
    hit_t      *capture;            /* OUT_CAPTURE: array of hits_wanted slots */
    int         progress_width;     /* columns of the progress line shown, or 0 */
    int         output_failed;      /* stdout refused a key; guarded by the lock */
    atomic_int  stop;
    atomic_ullong faults;           /* fast-path results the reference refused */
} g_run;

static volatile sig_atomic_t g_interrupted;

#define PROGRESS_COLUMNS     100    /* no progress line is wider than this              */
#define PROGRESS_LOG_SECONDS 10.0   /* until the first progress line in a log (see below) */
#define PROGRESS_LOG_SLOWEST 3600.0 /* and the longest wait between two of them           */

/*
 * The widest line that can be redrawn in place right now; less than 1 when
 * stderr is not a terminal.  Three columns of the terminal stay free: two
 * for the ^C that a terminal prints behind the line on Ctrl-C, and the last
 * one, because a character there makes some terminals move on to the next
 * row, and a line that has wrapped can no longer be redrawn or erased.
 */
static int progress_limit(void)
{
    const int room = os_stderr_columns() - 3;

    return room < PROGRESS_COLUMNS ? room : PROGRESS_COLUMNS;
}

/*
 * A progress line is put together from pieces, the most important first.
 * A piece that does not fit is left out with all that follow it, so that a
 * narrow terminal shows a shorter line and not one cut off in mid-word.
 */
typedef struct {
    char text[PROGRESS_COLUMNS + 1];
    int  limit;                     /* columns there are to fill            */
    int  closed;                    /* a piece did not fit: nothing follows */
} status_t;

PRINTF_LIKE(2, 3)
static void status_add(status_t *st, const char *fmt, ...)
{
    char piece[PROGRESS_COLUMNS + 1];
    const int used = (int)strlen(st->text);
    const int room = (st->limit < PROGRESS_COLUMNS ? st->limit : PROGRESS_COLUMNS) - used;
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(piece, sizeof piece, fmt, ap);
    va_end(ap);
    if (st->closed || n < 0 || n > room) {
        st->closed = 1;
        return;
    }
    memcpy(st->text + used, piece, (size_t)n + 1);
}

/*
 * Draw a line, made for the width the terminal has now, over the progress
 * line: in one write, and padded so that it also covers a longer line drawn
 * before.  The caller of this and of progress_clear holds the output lock
 * or is the only thread writing.
 */
static void progress_draw(const status_t *line)
{
    char out[PROGRESS_COLUMNS + 1];
    const int width = (int)strlen(line->text);

    if (line->limit < 1)
        return;
    if (g_run.progress_width > line->limit)
        g_run.progress_width = line->limit;     /* the window has become narrower */
    if (g_run.progress_width < width)
        g_run.progress_width = width;
    out[0] = '\r';
    memset(out + 1, ' ', (size_t)g_run.progress_width);
    memcpy(out + 1, line->text, (size_t)width);
    (void)os_write(TO_STDERR, out, (size_t)g_run.progress_width + 1);
}

/*
 * Erase the progress line, if one is showing, over its full width so that
 * nothing of it is left behind whatever is printed next.  After Ctrl-C that
 * includes the ^C which the terminal has put behind it.
 */
static void progress_clear(void)
{
    char out[PROGRESS_COLUMNS + 4];
    const int limit = progress_limit();
    int width = g_run.progress_width;

    g_run.progress_width = 0;
    if (limit < 1)
        return;
    if (width > limit)
        width = limit;
    if (g_interrupted)
        width += 2;
    if (width == 0)
        return;
    out[0] = '\r';
    memset(out + 1, ' ', (size_t)width);
    out[width + 1] = '\r';
    (void)os_write(TO_STDERR, out, (size_t)width + 2);
}

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

/* Print one key as its own block: a header naming the target, then the key. */
static void print_hit(uint64_t number, const hit_t *h)
{
    char sk_hex[65], pk_hex[65], sk_b64[45], pk_b64[45], which[32] = "";
    char block[2048];
    int n;

    hex_encode(sk_hex, h->secret, 32);
    hex_encode(pk_hex, h->public_key, 32);
    base64_encode(sk_b64, h->secret, 32);
    base64_encode(pk_b64, h->public_key, 32);

    if (g_run.mode == OUT_JSON) {
        n = snprintf(block, sizeof block,
                     "{\"target\":%d,\"pattern\":\"%s\","
                     "\"node_id\":\"!%08" PRIx32 "\",\"node_num\":%" PRIu32 ","
                     "\"private_key_hex\":\"%s\",\"private_key_base64\":\"%s\","
                     "\"public_key_hex\":\"%s\",\"public_key_base64\":\"%s\","
                     "\"command\":\"meshtastic --set security.private_key \\\"base64:%s\\\"\"}\n",
                     h->target + 1, g_target[h->target].text, h->node_id, h->node_id,
                     sk_hex, sk_b64, pk_hex, pk_b64, sk_b64);
    } else {
        if (g_targets > 1)
            snprintf(which, sizeof which, " | target %d", h->target + 1);
        n = snprintf(block, sizeof block,
                     "==== key %" PRIu64 " of %" PRIu64 "%s | %s ====\n"
                     "  node ID:              !%08" PRIx32 "\n"
                     "  public key (base64):  %s\n"
                     "  public key (hex):     %s\n"
                     "  private key (base64): %s\n"
                     "  private key (hex):    %s\n"
                     "  install with:         meshtastic --set security.private_key \"base64:%s\"\n"
                     "\n",
                     number, g_run.hits_wanted, which, g_target[h->target].text, h->node_id,
                     pk_b64, pk_hex, sk_b64, sk_hex, sk_b64);
    }

    /*
     * One write for the whole block, so that it appears at once and unbroken.
     * A key goes nowhere else: if stdout does not take it, it is lost, and
     * the caller ends the search.
     */
    progress_clear();
    if (n < 0 || (size_t)n >= sizeof block || os_write(TO_STDOUT, block, (size_t)n) != 0)
        g_run.output_failed = 1;
    secure_zero(block, sizeof block);
    secure_zero(sk_hex, sizeof sk_hex);
    secure_zero(sk_b64, sizeof sk_b64);
}

/*
 * The fast path says candidate j of this lane has public key fast_pk and node
 * ID fast_id, and that a target wants it.  Recompute everything
 * independently; only a confirmed key is reported.  Always returns 1: the
 * lane's base is finished either way.
 */
static NOINLINE int engine_verify(const lane_t *lane, unsigned j, const uint8_t fast_pk[32],
                                  uint32_t fast_id)
{
    hit_t h;

    scalar_offset(h.secret, lane->base, lane->win * WINDOW + j);
    x25519_ref(h.public_key, h.secret, BASE_U);
    h.node_id = crc32_ref(h.public_key, 32);
    h.target = -1;

    if (memcmp(h.public_key, fast_pk, 32) != 0 || h.node_id != fast_id) {
        atomic_fetch_add(&g_run.faults, 1);
    } else {
        output_lock();
        /*
         * One key serves one target.  If several that still want a key accept
         * this one, it goes to the hardest to please: a key for "dead" must
         * not be spent on "de".
         */
        for (int k = 0; k < g_targets; k++)
            if (g_target[k].found < g_target[k].wanted &&
                target_matches(&g_target[k], h.public_key, h.node_id) &&
                (h.target < 0 || g_target[k].chance < g_target[h.target].chance))
                h.target = k;
        if (h.target >= 0 && !g_run.output_failed) {
            target_t *t = &g_target[h.target];

            if (g_run.mode == OUT_CAPTURE)
                g_run.capture[g_run.hits] = h;
            else
                print_hit(g_run.hits + 1, &h);
            if (g_run.output_failed) {
                atomic_store(&g_run.stop, 1);   /* this key is lost; the next would be too */
            } else {
                if (t->found == 0)
                    t->first_id = h.node_id;
                if (++t->found == t->wanted)
                    filter_publish();           /* the kernels stop looking for it */
                if (++g_run.hits == g_run.hits_wanted)
                    atomic_store(&g_run.stop, 1);
            }
        }
        output_unlock();
    }
    secure_zero(&h, sizeof h);
    return 1;
}

/*
 * A kernel's filter let candidate j of a lane through.  `words` is the public
 * key the kernel computed, and `id` its node ID if the kernel computed one.
 * Decide exactly, on those values, whether a target of the filter wants the
 * key; if so, send it to the reference check.  Returns 1 if the lane's base
 * must be retired, 0 if the filter merely let through a key nobody wants.
 */
static NOINLINE int engine_candidate(const filter_t *f, const lane_t *lane, unsigned j,
                                     const uint64_t words[4], int have_id, uint32_t id)
{
    uint8_t pk[32];
    int wanted = 0;

    words_to_bytes(pk, words);
    for (unsigned g = 0; g < f->groups && !wanted; g++) {
        const fgroup_t *grp = &f->group[g];
        const uint64_t v = (grp->source == SRC_ID ? (uint64_t)id : words[grp->source]) & grp->mask;

        for (unsigned m = grp->first; m < (unsigned)grp->first + grp->count && !wanted; m++) {
            const target_t *t = &g_target[f->member[m]];

            if (f->member_value[m] != v || !pattern_matches_key(&t->key, pk))
                continue;
            if (!have_id) {                     /* the kernel skipped the CRC; this key needs it */
                crc_msgs((const uint64_t (*)[4])(const void *)words, &id, 1);
                have_id = 1;
            }
            wanted = pattern_matches_id(&t->id, id) && !node_id_reserved(id);
        }
    }
    return wanted ? engine_verify(lane, j, pk, id) : 0;
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
    const filter_t *const f = filter_current();
    const int want_id = f->need_id || dump != NULL;
    fe e0, e, d, acc, inv, t, lam;
    uint64_t msg[2][4];
    uint32_t id[2] = {0, 0};
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

        if (want_id)
            crc_msgs((const uint64_t (*)[4])msg, id, 2);

        if (UNLIKELY(dump != NULL)) {
            memcpy(dump->words[HALF_WINDOW + i], msg[0], 32);
            memcpy(dump->words[HALF_WINDOW - i], msg[1], 32);
            dump->id[HALF_WINDOW + i] = id[0];
            dump->id[HALF_WINDOW - i] = id[1];
        }
        if (UNLIKELY(filter_pass(f, msg[0], id[0])) && !retired)
            retired = engine_candidate(f, lane, (unsigned)(HALF_WINDOW + i), msg[0], want_id,
                                       id[0]);
        if (UNLIKELY(filter_pass(f, msg[1], id[1])) && !retired)
            retired = engine_candidate(f, lane, (unsigned)(HALF_WINDOW - i), msg[1], want_id,
                                       id[1]);
    }

    /* The centre itself. */
    fe_to_words(msg[0], &w->um);
    if (want_id)
        crc_msgs((const uint64_t (*)[4])msg, id, 1);
    if (UNLIKELY(dump != NULL)) {
        memcpy(dump->words[HALF_WINDOW], msg[0], 32);
        dump->id[HALF_WINDOW] = id[0];
    }
    if (UNLIKELY(filter_pass(f, msg[0], id[0])) && !retired)
        retired = engine_candidate(f, lane, HALF_WINDOW, msg[0], want_id, id[0]);

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

/*
 * The filter on eight keys at once: bit n of the result is set if key n may
 * be wanted, where keys 0..3 are the lanes of grp[0] and keys 4..7 those of
 * grp[1].  A group of one target is a vector comparison; a larger group
 * falls back to one look-up per key.
 */
AVX2_INLINE unsigned filter8(const filter_t *f, const uint64_t (*grp)[4][4], const uint32_t *id)
{
    unsigned hit = 0;

    for (unsigned g = 0; g < f->groups; g++) {
        const fgroup_t *fg = &f->group[g];

        if (fg->source == SRC_ID) {
            if (fg->single) {
                const __m256i v = V4_AND(_mm256_loadu_si256((const __m256i *)(const void *)id),
                                         _mm256_set1_epi32((int)(uint32_t)fg->mask));
                hit |= (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(v, _mm256_set1_epi32((int)(uint32_t)fg->value))));
            } else {
                for (unsigned n = 0; n < 8; n++)
                    hit |= (unsigned)fgroup_pass(fg, id[n]) << n;
            }
        } else {
            const unsigned k = fg->source;

            if (fg->single) {
                const __m256i m = V4_SET1(fg->mask), want = V4_SET1(fg->value);
                const __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)grp[0][k]);
                const __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)grp[1][k]);
                hit |= (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(
                           _mm256_cmpeq_epi64(V4_AND(a, m), want))) |
                       (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(
                           _mm256_cmpeq_epi64(V4_AND(b, m), want))) << 4;
            } else {
                for (unsigned n = 0; n < 8; n++)
                    hit |= (unsigned)fgroup_pass(fg, grp[n >> 2][k][n & 3]) << n;
            }
        }
    }
    return hit;
}

/*
 * Follow up the keys the filter let through: bit n of `hit` is lane n & 3 of
 * grp[n >> 2], which holds candidate j_first or j_second of its lane.  A lane
 * reports at most one key per base, so once it is retired its later
 * candidates in the window are dropped.
 */
static NOINLINE void walk4_candidates(walk4_t *w, const filter_t *f, unsigned *retired,
                                      unsigned hit, const uint64_t (*grp)[4][4],
                                      const uint32_t *id, int have_id, unsigned j_first,
                                      unsigned j_second)
{
    for (unsigned n = 0; n < 8; n++) {
        const unsigned l = n & 3, g = n >> 2;
        uint64_t words[4];

        if (!((hit >> n) & 1) || (*retired & (1u << l)))
            continue;
        for (int k = 0; k < 4; k++)
            words[k] = grp[g][k][l];
        if (engine_candidate(f, &w->lane[l], g == 0 ? j_first : j_second, words, have_id, id[n]))
            *retired |= 1u << l;
    }
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
    const filter_t *const f = filter_current();
    const int want_id = f->need_id || dump != NULL;
    fe4 um, vm, bias_u, bias_v, e0, e, p, g19, acc, inv, d, t, lam, x, ui, vi, pre;
    __m256i words[4];
    _Alignas(32) uint64_t grp[2][4][4];
    _Alignas(32) uint32_t id[8] = {0};
    unsigned retired, hit;

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

        if (want_id)
            crc_groups((const uint64_t (*)[4][4])grp, id, 2);

        if (UNLIKELY(dump != NULL)) {
            walk4_dump(dump, (const uint64_t (*)[4])grp[0], id, (unsigned)(HALF_WINDOW + i));
            walk4_dump(dump, (const uint64_t (*)[4])grp[1], id + 4, (unsigned)(HALF_WINDOW - i));
        }
        hit = filter8(f, (const uint64_t (*)[4][4])grp, id);
        if (UNLIKELY(hit != 0))
            walk4_candidates(w, f, &retired, hit, (const uint64_t (*)[4][4])grp, id, want_id,
                             (unsigned)(HALF_WINDOW + i), (unsigned)(HALF_WINDOW - i));
    }

    /* The centres themselves. */
    fe4_to_words(words, &um);
    for (int k = 0; k < 4; k++)
        _mm256_storeu_si256((__m256i *)(void *)grp[0][k], words[k]);
    if (want_id)
        crc_groups((const uint64_t (*)[4][4])grp, id, 1);
    if (UNLIKELY(dump != NULL))
        walk4_dump(dump, (const uint64_t (*)[4])grp[0], id, HALF_WINDOW);
    hit = filter8(f, (const uint64_t (*)[4][4])grp, id) & 0xfu;    /* grp[1] is stale here */
    if (UNLIKELY(hit != 0))
        walk4_candidates(w, f, &retired, hit, (const uint64_t (*)[4][4])grp, id, want_id,
                         HALF_WINDOW, HALF_WINDOW);

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
#define LONG_SEARCH_SECONDS 86400.0         /* worth a warning */
#define SLOW_TERMINAL_SECONDS 0.1           /* for one status line: worth a remark */

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
        os_yield();
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

/* A key count for people: "65.5 thousand", "1.72 billion". */
static void format_count(char *out, size_t n, double count)
{
    static const char *const unit[] = {"", " thousand", " million", " billion", " trillion",
                                       " quadrillion", " quintillion"};
    size_t u = 0;
    double v = count;

    if (count >= 0.9995e21) {               /* beyond words */
        snprintf(out, n, "%.3g", count);
        return;
    }
    while (v >= 999.5 && u + 1 < ARRAY_LEN(unit)) {
        v /= 1e3;
        u++;
    }
    if (u == 0)
        snprintf(out, n, "%.0f", v);
    else
        snprintf(out, n, "%.*f%s", v < 9.995 ? 2 : v < 99.95 ? 1 : 0, v, unit[u]);
}

#define SECONDS_PER_YEAR 31557600.0

static void format_duration(char *out, size_t n, double seconds)
{
    if (seconds < 1.0)
        snprintf(out, n, "%.0f ms", seconds * 1e3);
    else if (seconds < 120.0)
        snprintf(out, n, "%.1f s", seconds);
    else if (seconds < 7200.0)
        snprintf(out, n, "%.1f min", seconds / 60.0);
    else if (seconds < 172800.0)
        snprintf(out, n, "%.1f h", seconds / 3600.0);
    else if (seconds < 2.0 * SECONDS_PER_YEAR)
        snprintf(out, n, "%.0f days", seconds / 86400.0);
    else if (seconds < 1e6 * SECONDS_PER_YEAR)
        snprintf(out, n, "%.0f years", seconds / SECONDS_PER_YEAR);
    else
        snprintf(out, n, "%.2g years", seconds / SECONDS_PER_YEAR);
}

typedef struct {
    unsigned long long keys;
    double seconds;
    int threads;
    char kernels[64];                       /* e.g. "avx2 x16, scalar x16" */
    double terminal_wait;                   /* the longest handing over one status line took */
} run_result;

/*
 * Run nthreads workers until the wanted hits are found, the time or key
 * budget runs out (0 = unlimited), or the user presses Ctrl-C.  be == NULL
 * lets every thread choose its own kernel; pin binds thread i to CPU i.
 */
static run_result engine_run(const backend_t *be, int nthreads, int pin, double time_limit,
                             unsigned long long key_limit, int show_progress)
{
    run_result res = {0, 0.0, 0, "", 0.0};
    /* on a terminal one line is redrawn in place; a file or a pipe gets a line now and then */
    const int live = show_progress && os_stderr_columns() > 0;
    double expected = 0.0, start, last_report, log_at = PROGRESS_LOG_SECONDS;
    uint64_t estimated_at = UINT64_MAX;     /* g_run.hits when `expected` was worked out */
    int warned = 0;

    atomic_store(&g_run.stop, 0);
    g_run.hits = 0;
    g_run.progress_width = 0;
    g_run.output_failed = 0;
    filter_reset();
    if (g_targets > 0)
        filter_publish();

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

        if (show_progress && now - last_report >= 1.0 && res.keys > 0) {
            const double rate = (double)res.keys / res.seconds;
            status_t line = {"", live ? progress_limit() : PROGRESS_COLUMNS, 0};
            double waited = now_seconds() - now;    /* for the terminal to say how wide it is */
            char count[40], mean[40], p95[40];

            format_count(count, sizeof count, (double)res.keys);
            output_lock();
            if (g_run.hits != estimated_at) {       /* a key was found: the outlook changed */
                estimated_at = g_run.hits;
                expected = targets_expected_keys();
            }
            format_duration(mean, sizeof mean, expected / rate);
            if (g_run.hits_wanted != 1)
                status_add(&line, "%" PRIu64 " of %" PRIu64 " found", g_run.hits,
                           g_run.hits_wanted);
            status_add(&line, "%s%s keys", line.text[0] != '\0' ? ", " : "", count);
            status_add(&line, ", %.1f M/s", rate / 1e6);
            if (g_run.hits_wanted != 1) {
                status_add(&line, ", about %s more on average", mean);
            } else if (expected / rate > LONG_SEARCH_SECONDS) {
                status_add(&line, ", average wait %s", mean);
            } else {
                format_duration(p95, sizeof p95, 3.0 * expected / rate);
                status_add(&line, ", %.0f%% likely by now",
                           100.0 * (1.0 - exp(-(double)res.keys / expected)));
                status_add(&line, ", avg %s", mean);
                status_add(&line, ", 95%% by %s", p95);
            }
            if (!warned && expected / rate > LONG_SEARCH_SECONDS) {
                char longest[40], whose[32] = "the search";
                double most = 0.0;
                int hard = 0;                       /* the target that needs the most work */

                for (int k = 0; k < g_targets; k++) {
                    const target_t *t = &g_target[k];
                    const double work = t->found < t->wanted
                                            ? (double)(t->wanted - t->found) / t->chance : 0.0;
                    if (work > most) {
                        most = work;
                        hard = k;
                    }
                }
                if (g_targets > 1)
                    snprintf(whose, sizeof whose, "target %d alone", hard + 1);
                format_duration(longest, sizeof longest, most / rate);
                warned = 1;
                progress_clear();
                err_printf("note: at this speed %s takes about %s on average.\n"
                           "      Each character less makes it 16 times (hex) or 64 times "
                           "(base64) shorter.\n", whose, longest);
            }
            if (live) {
                const double before = now_seconds();
                progress_draw(&line);
                waited += now_seconds() - before;
                if (waited > res.terminal_wait)
                    res.terminal_wait = waited;
            } else if (res.seconds >= log_at) {
                /* a log is not the place for a line a second: wait twice as long each time */
                err_printf("%s\n", line.text);
                log_at += log_at < PROGRESS_LOG_SLOWEST ? log_at : PROGRESS_LOG_SLOWEST;
            }
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
    progress_clear();
    filter_reset();

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

PRINTF_LIKE(2, 3)
static void t_check(int ok, const char *fmt, ...)
{
    char what[256];
    va_list ap;

    if (!ok)
        t_failures++;
    if (!t_verbose && ok)
        return;
    va_start(ap, fmt);
    vsnprintf(what, sizeof what, fmt, ap);
    va_end(ap);
    out_printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
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

/* ---- targets and the filter ---- */

/*
 * The tests' own reading of a pattern, applied to the string a person would
 * compare by eye.  It shares nothing with the parser or the matcher above.
 */
static int t_class(char letter, char c, int hex)
{
    switch (letter) {
    case 'd': return c >= '0' && c <= '9';
    case 'l': return hex ? c >= 'a' && c <= 'f' : c >= 'a' && c <= 'z';
    case 'u': return hex ? c >= 'a' && c <= 'f' : c >= 'A' && c <= 'Z';
    case 'h': return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    case 'H': return (c >= '0' && c <= '9') || (hex ? c >= 'a' && c <= 'f' : c >= 'A' && c <= 'F');
    case 's': return c == '+' || c == '/';
    default:  return 1;                         /* ?a */
    }
}

static int t_glob(const char *pat, const char *str, int hex, int fold)
{
    if (*pat == '\0')
        return *str == '\0';
    if (*pat == '*') {
        for (;; str++) {
            if (t_glob(pat + 1, str, hex, fold))
                return 1;
            if (*str == '\0')
                return 0;
        }
    }
    if (*str == '\0')
        return 0;
    if (pat[0] == '?' && pat[1] != '\0' && strchr("adluhHs", pat[1]) != NULL)
        return t_class(pat[1], *str, hex) && t_glob(pat + 2, str + 1, hex, fold);
    if (*pat == '.')
        return t_glob(pat + 1, str + 1, hex, fold);
    if (hex || fold ? ascii_lower(*pat) != ascii_lower(*str) : *pat != *str)
        return 0;
    return t_glob(pat + 1, str + 1, hex, fold);
}

/* Does a key satisfy string-level patterns for its three strings (NULL: anything)? */
static int t_expect(const uint8_t pk[32], uint32_t id, const char *id_glob, const char *key_glob,
                    const char *hex_glob, int fold)
{
    char id_str[9], b64[45], hex[65];

    snprintf(id_str, sizeof id_str, "%08" PRIx32, id);
    base64_encode(b64, pk, 32);
    hex_encode(hex, pk, 32);
    return (id_glob == NULL || t_glob(id_glob, id_str, 1, 0)) &&
           (key_glob == NULL || t_glob(key_glob, b64, 0, fold)) &&
           (hex_glob == NULL || t_glob(hex_glob, hex, 1, 0)) && !(id < 4u || id == 0xffffffffu);
}

/* A random public-key encoding (bit 255 clear) and a random node ID. */
static void t_random_key(uint8_t pk[32], uint32_t *id)
{
    uint8_t r[4];

    os_random(pk, 32);
    pk[31] &= 127;
    os_random(r, sizeof r);
    *id = (uint32_t)r[0] | (uint32_t)r[1] << 8 | (uint32_t)r[2] << 16 | (uint32_t)r[3] << 24;
}

/* The n-th (from 0, wrapping) symbol of a set. */
static unsigned t_pick(uint64_t set, unsigned n)
{
    n %= (unsigned)__builtin_popcountll(set);
    for (unsigned v = 0;; v++)
        if (((set >> v) & 1) && n-- == 0)
            return v;
}

/* Bend a key and node ID until they fit a target, choosing at random where it leaves a choice. */
static void t_plant(const target_t *t, uint8_t pk[32], uint32_t *id)
{
    uint8_t choice[2][64];

    os_random(choice, sizeof choice);
    for (unsigned i = 0; i < t->id.count; i++) {
        const unsigned v = t_pick(t->id.rule[i].allowed, choice[0][i]);
        for (unsigned s = 0; s < 4; s++) {
            const int b = form_bit(FORM_ID, t->id.rule[i].pos, s);
            *id = (*id & ~((uint32_t)1 << b)) | (uint32_t)((v >> s) & 1) << b;
        }
    }
    for (unsigned i = 0; i < t->key.count; i++) {
        const unsigned v = t_pick(t->key.rule[i].allowed, choice[1][i]);
        for (unsigned s = 0; s < FORMS[t->key.form].bits; s++) {
            const int b = form_bit(t->key.form, t->key.rule[i].pos, s);
            if (b >= 0)
                pk[b / 8] = (uint8_t)((pk[b / 8] & ~(1u << (b % 8))) | ((v >> s) & 1) << (b % 8));
        }
    }
}

static void t_words(uint64_t w[4], const uint8_t pk[32])
{
    for (int k = 0; k < 4; k++)
        w[k] = load64_le(pk + 8 * k);
}

/* Parse one specification into the global target list; returns 1 if it was accepted. */
static int t_targets(const char *spec, int ignore_case)
{
    targets_clear();
    return targets_add(spec, ignore_case) == NULL;
}

static void test_targets(int full)
{
    static const char *const good[] = {
        "a", "dead", "!c0de", "DC80*a", "dead,beef,cafe", "!dead,!beef", "deadbeef", "....beef",
        "0000000.", "dead,dead", "dead,abc", "id:dead", "ID:*beef", "id:de*ef", "id:d..d.00d",
        "id:?d?d?d?d?d?d?d?d", "0000000?d", "fffffff?l", "?a?hbe?H?a", "key:Mesh", "KEY:Mesh",
        "key:*Radio=", "key:*8", "key:Me*gQ=", "key:?u?l?d?s", "key:a+b", "key:Me.",
        "keyhex:c0ffee", "keyhex:*7f", "keyhex:?h?d", "id:dead+key:Me", "key:Me+id:dead",
        "dead+keyhex:00", "id:c0+key:a+b",
    };
    static const char *const bad[] = {
        "", "!", "g", "dead,", ",dead", "123456789", "****", "*", "....", "id:", "key:", "00000000",
        "00000003", "ffffffff", "dead beef", "id:?s", "id:de*ad*", "key:Me=", "key:a=b", "key:*a*b",
        "key:Mesh!", "key:*Node", "key:*e=", "key:*IA", "keyhex:zz", "keyhex:*8f", "keyhex:?s",
        "id:dead+id:beef", "key:a+keyhex:b", "Mesh", "dead????", "d??d", "dea?", "deadxxxx",
        "key:Me?", "key:?x", "key:??",
    };
    static const struct {
        const char *spec, *text;
        int fold;
    } shown[] = {
        {"dead", "node ID !dead*", 0},
        {"dead*", "node ID !dead*", 0},
        {"!DEAD", "node ID !dead*", 0},
        {"id:dead", "node ID !dead*", 0},
        {"*BEEF", "node ID !*beef", 0},
        {"id:de*ef", "node ID !de*ef", 0},
        {"deadbeef", "node ID !deadbeef", 0},
        {"d..d", "node ID !d..d*", 0},
        {"c0?de", "node ID !c0?de*", 0},
        {"c0.de", "node ID !c0.de*", 0},
        {"key:Mesh", "public key Mesh*", 0},
        {"key:*Q0", "public key *Q0=", 0},
        {"key:*Q0=", "public key *Q0=", 0},
        {"key:a*b*", "", 0},
        {"keyhex:C0FFEE", "public key (hex) c0ffee*", 0},
        {"id:c0+key:?uA", "node ID !c0* + public key ?uA*", 0},
        {"key:?uA+id:c0", "node ID !c0* + public key ?uA*", 0},
        {"key:me", "public key me* (any case)", 1},
        {"key:42", "public key 42*", 1},
    };
    static const struct {
        const char *spec;
        double one_in;                  /* 1 / chance */
    } odds[] = {
        {"dead", 65536.0},
        {"deadbeef", 4294967296.0},
        {"0000000.", 4294967296.0 / 12.0},
        {"id:?d?d?d?d?d?d?d?d", 4294967296.0 / (1e8 - 4.0)},
        {"key:M", 64.0 / (1.0 - 5.0 / 4294967296.0)},
        {"key:*8", 16.0 / (1.0 - 5.0 / 4294967296.0)},
        {"key:*Q0", 512.0 / (1.0 - 5.0 / 4294967296.0)},
        {"keyhex:*7f", 128.0 / (1.0 - 5.0 / 4294967296.0)},
        {"key:?d?d", 4096.0 / 100.0 / (1.0 - 5.0 / 4294967296.0)},
        {"id:dead+key:M", 65536.0 * 64.0},
    };
    /* a specification and what it means for each of the key's strings (NULL: anything) */
    static const struct {
        const char *spec;
        int fold;
        const char *id, *key, *hex;
    } cases[] = {
        {"a", 0, "a*", NULL, NULL},
        {"!c0", 0, "c0*", NULL, NULL},
        {"id:*5a", 0, "*5a", NULL, NULL},
        {"id:d*d", 0, "d*d", NULL, NULL},
        {"7.e", 0, "7.e*", NULL, NULL},
        {"id:?d?l*", 0, "?d?l*", NULL, NULL},
        {"id:*?u?d", 0, "*?u?d", NULL, NULL},
        {"id:?d?d?d?d?d?d?d?d", 0, "?d?d?d?d?d?d?d?d", NULL, NULL},
        {"....beef", 0, "....beef", NULL, NULL},
        {"0000000.", 0, "0000000.", NULL, NULL},
        {"key:M", 0, NULL, "M*", NULL},
        {"key:Me*", 0, NULL, "Me*", NULL},
        {"key:*8=", 0, NULL, "*8=", NULL},
        {"key:*Q0", 0, NULL, "*Q0=", NULL},
        {"key:a*w=", 0, NULL, "a*w=", NULL},
        {"key:?u?l?d?s", 0, NULL, "?u?l?d?s*", NULL},
        {"key:.?hx", 0, NULL, ".?hx*", NULL},
        {"key:?H?a.", 0, NULL, "?H*", NULL},
        {"key:+/", 0, NULL, "+/*", NULL},
        {"key:me", 1, NULL, "me*", NULL},
        {"key:*?d?l?u?d", 0, NULL, "*?d?l?u?d=", NULL},
        {"keyhex:c0", 0, NULL, NULL, "c0*"},
        {"keyhex:AB", 0, NULL, NULL, "ab*"},
        {"keyhex:*7f", 0, NULL, NULL, "*7f"},
        {"keyhex:?d?l*?d", 0, NULL, NULL, "?d?l*?d"},
        {"id:a+key:b", 0, "a*", "b*", NULL},
        {"key:b+id:*a", 0, "*a", "b*", NULL},
        {"d+keyhex:0", 0, "d*", NULL, "0*"},
    };
    const int rounds = full ? 4000 : 64;
    const double two_unequal = 65536.0 + 256.0 - 1.0 / (1.0 / 65536.0 + 1.0 / 256.0);
    uint64_t seen[2][3];                /* symbols met at the last positions of the key strings */
    uint64_t possible[2][64];
    int wrong = 0, bad_symbol = 0, bad_possible = 0, bad_match = 0, bad_mask = 0, vacuous = 0;

    /* the parser */
    for (size_t i = 0; i < ARRAY_LEN(good); i++)
        wrong += !t_targets(good[i], 0);
    for (size_t i = 0; i < ARRAY_LEN(bad); i++)
        wrong += t_targets(bad[i], 0);
    wrong += !t_targets("dead,beef,cafe", 0) || g_targets != 3;
    wrong += !t_targets("dead,dead", 0) || g_targets != 2;
    wrong += !t_targets("'id:*beef,key:Me'", 0) || g_targets != 2 ||
             strcmp(g_target[0].text, "node ID !*beef") != 0;
    wrong += t_targets("'dead", 0) || t_targets("''", 0) || t_targets("'", 0);
    wrong += !t_targets("id:c0+key:a+b", 0) || g_targets != 1 ||
             strcmp(g_target[0].key.text, "a+b*") != 0;
    {                                           /* a whole public key, and one character more */
        char whole[64] = "key:";
        memset(whole + 4, 'A', 43);
        strcpy(whole + 47, "=");
        wrong += !t_targets(whole, 0) || strlen(g_target[0].key.text) != 44;
        strcpy(whole + 47, "A");
        wrong += t_targets(whole, 0);
    }
    t_check(wrong == 0, "targets: the parser accepts and rejects the right forms");

    wrong = 0;
    for (size_t i = 0; i < ARRAY_LEN(shown); i++) {
        if (shown[i].text[0] == '\0')
            wrong += t_targets(shown[i].spec, shown[i].fold);
        else
            wrong += !t_targets(shown[i].spec, shown[i].fold) ||
                     strcmp(g_target[0].text, shown[i].text) != 0;
    }
    for (size_t i = 0; i < ARRAY_LEN(odds); i++)
        wrong += !t_targets(odds[i].spec, 0) ||
                 fabs(1.0 / g_target[0].chance / odds[i].one_in - 1.0) > 1e-9;
    t_check(wrong == 0, "targets: canonical spelling and the odds of a match");

    /* how positions of the three strings map to bits of the key and the node ID */
    memset(seen, 0, sizeof seen);
    for (int k = 0; k < 2; k++)
        for (unsigned pos = 0; pos < FORMS[k == 0 ? FORM_KEY : FORM_KEYHEX].length; pos++)
            possible[k][pos] = form_possible(k == 0 ? FORM_KEY : FORM_KEYHEX, pos);
    for (int i = 0; i < rounds; i++) {
        char id_str[9], str[2][65];
        uint8_t pk[32];
        uint32_t id;

        t_random_key(pk, &id);
        snprintf(id_str, sizeof id_str, "%08" PRIx32, id);
        base64_encode(str[0], pk, 32);
        hex_encode(str[1], pk, 32);
        for (unsigned pos = 0; pos < 8; pos++) {
            unsigned v = 0;
            for (unsigned s = 0; s < 4; s++)
                v |= ((id >> form_bit(FORM_ID, pos, s)) & 1u) << s;
            bad_symbol += HEX_ALPHABET[v] != id_str[pos];
        }
        for (int k = 0; k < 2; k++) {
            const int form = k == 0 ? FORM_KEY : FORM_KEYHEX;
            const form_t *f = &FORMS[form];

            for (unsigned pos = 0; pos < f->length; pos++) {
                const unsigned sym = key_symbol(form, pk, pos);
                unsigned v = 0;
                for (unsigned s = 0; s < f->bits; s++) {
                    const int b = form_bit(form, pos, s);
                    if (b >= 0)
                        v |= ((unsigned)(pk[b / 8] >> (b % 8)) & 1u) << s;
                }
                bad_symbol += v != sym || f->alphabet[sym] != str[k][pos];
                bad_possible += !((possible[k][pos] >> sym) & 1);
                if (pos + 3 >= f->length)
                    seen[k][pos + 3 - f->length] |= (uint64_t)1 << sym;
            }
        }
    }
    bad_symbol += strlen(BASE64_ALPHABET) != 64 || strlen(HEX_ALPHABET) != 16;
    /* the full run has seen every symbol that can end a key, and no other */
    for (int k = 0; k < 2 && full; k++)
        for (unsigned e = 0; e < 3; e++)
            bad_possible += seen[k][e] !=
                            possible[k][FORMS[k == 0 ? FORM_KEY : FORM_KEYHEX].length - 3 + e];
    bad_possible += __builtin_popcountll(form_possible(FORM_KEY, 42)) != 16 ||
                    __builtin_popcountll(form_possible(FORM_KEY, 41)) != 32 ||
                    __builtin_popcountll(form_possible(FORM_KEY, 40)) != 64 ||
                    __builtin_popcountll(form_possible(FORM_KEYHEX, 62)) != 8 ||
                    __builtin_popcountll(form_possible(FORM_KEYHEX, 63)) != 16;
    t_check(bad_symbol == 0, "targets: string positions map to the right bits of key and node ID");
    t_check(bad_possible == 0, "targets: the characters that can end a public key");

    /* matching, against the tests' own reading of each pattern on the printed strings */
    for (size_t c = 0; c < ARRAY_LEN(cases); c++) {
        const target_t *t = &g_target[0];
        int hits = 0;

        if (!t_targets(cases[c].spec, cases[c].fold) || g_targets != 1) {
            bad_match++;
            continue;
        }
        for (int i = 0; i < rounds; i++) {
            uint64_t w[4];
            uint8_t pk[32];
            uint32_t id;
            int got, want;

            t_random_key(pk, &id);
            if (i & 1)
                t_plant(t, pk, &id);            /* every other sample is made to fit */
            got = target_matches(t, pk, id);
            want = t_expect(pk, id, cases[c].id, cases[c].key, cases[c].hex, cases[c].fold);
            bad_match += got != want;
            hits += got;
            if (got) {
                t_words(w, pk);
                bad_mask += (id & t->id_mask) != t->id_value;
                for (int k = 0; k < 4; k++)
                    bad_mask += (w[k] & t->key_mask[k]) != t->key_value[k];
            }
        }
        vacuous += hits == 0;
    }
    t_check(bad_match == 0 && vacuous == 0,
            "targets: %d patterns match exactly the keys a reader would expect",
            (int)ARRAY_LEN(cases));
    t_check(bad_mask == 0, "targets: the fixed bits of a pattern hold for every match");

    /* the expected work for several targets: harmonic numbers when the odds are equal */
    wrong = !t_targets("dead", 0);
    targets_want(1);
    wrong += fabs(targets_expected_keys() / 65536.0 - 1.0) > 1e-9;
    targets_want(3);
    wrong += fabs(targets_expected_keys() / (3.0 * 65536.0) - 1.0) > 1e-9;
    wrong += !t_targets("dead,beef,cafe,f00d,c0de", 0);
    targets_want(1);
    wrong += fabs(targets_expected_keys() / (65536.0 * 137.0 / 60.0) - 1.0) > 1e-3;
    wrong += !t_targets("dead,ab", 0);          /* 1/a + 1/b - 1/(a + b) */
    targets_want(1);
    wrong += fabs(targets_expected_keys() / two_unequal - 1.0) > 1e-3;
    g_target[1].found = 1;                      /* a finished target no longer counts */
    wrong += fabs(targets_expected_keys() / 65536.0 - 1.0) > 1e-9;
    t_check(wrong == 0, "targets: expected work for one and for several targets");

    targets_clear();
}

/* Is the target among those a filter would have examined for this key? */
static int t_filter_claims(const filter_t *f, int target, const uint64_t w[4], uint32_t id)
{
    for (unsigned g = 0; g < f->groups; g++) {
        const fgroup_t *grp = &f->group[g];
        const uint64_t v = (grp->source == SRC_ID ? (uint64_t)id : w[grp->source]) & grp->mask;

        if (!fgroup_pass(grp, grp->source == SRC_ID ? (uint64_t)id : w[grp->source]))
            continue;
        for (unsigned m = grp->first; m < (unsigned)grp->first + grp->count; m++)
            if (f->member[m] == target && f->member_value[m] == v)
                return 1;
    }
    return 0;
}

#ifdef MTV_X86
/* The vector filter against the scalar one on eight keys, some of them made to fit. */
AVX2_FN static int t_filter8(const filter_t *f)
{
    uint64_t grp[2][4][4];
    uint32_t id[8];
    uint8_t coin[8];
    unsigned want = 0;

    os_random(coin, sizeof coin);
    for (unsigned n = 0; n < 8; n++) {
        uint64_t w[4];
        uint8_t pk[32];

        t_random_key(pk, &id[n]);
        if (g_targets > 0 && (coin[n] & 1))
            t_plant(&g_target[coin[n] % (unsigned)g_targets], pk, &id[n]);
        t_words(w, pk);
        for (int k = 0; k < 4; k++)
            grp[n >> 2][k][n & 3] = w[k];
        want |= (unsigned)filter_pass(f, w, id[n]) << n;
    }
    return filter8(f, (const uint64_t (*)[4][4])grp, id) != want;
}
#endif

static void test_filter(int full)
{
    static const struct {
        const char *spec;
        unsigned groups;
        int need_id;
    } sets[] = {
        {"dead", 1, 1},
        {"dead,beef,cafe", 1, 1},                   /* one shape: one hashed group      */
        {"dead,*beef,de*ef,c0de", 3, 1},
        {"key:Me", 1, 0},                           /* no node ID: the CRC is skipped   */
        {"key:Me,key:Yo,key:*w=", 2, 0},
        {"keyhex:c0de,keyhex:*7f,key:*8", 3, 0},
        {"id:dead+key:Mesh", 1, 0},                 /* filtered on the key alone        */
        {"id:deadbeef+key:?l", 1, 1},               /* that key pattern fixes no bit    */
        {"id:dead+key:Mesh,cafe", 2, 1},
        {"id:?d?d,key:?l?l", 2, 1},                 /* nothing fixed: all keys pass     */
        {"0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f", 1, 1},
        {"key:A,key:B,key:C,id:d+key:D,id:e+key:E", 1, 0},
        {"id:d+keyhex:a", 1, 0},                    /* a tie goes to the key: no CRC    */
    };
    const int rounds = full ? 400 : 16;
    filter_t *f = (filter_t *)xalloc_aligned(sizeof *f);
    int bad_shape = 0, missed = 0, kept = 0, bad_vector = 0;

    for (size_t s = 0; s < ARRAY_LEN(sets); s++) {
        if (!t_targets(sets[s].spec, 0)) {
            bad_shape++;
            continue;
        }
        targets_want(1);
        filter_build(f);
        bad_shape += f->groups != sets[s].groups || f->need_id != sets[s].need_id;

        for (int i = 0; i < rounds; i++) {
            const int k = i % g_targets;
            uint64_t w[4];
            uint8_t pk[32];
            uint32_t id;

            t_random_key(pk, &id);
            t_plant(&g_target[k], pk, &id);
            t_words(w, pk);
            if (target_matches(&g_target[k], pk, id))
                missed += !filter_pass(f, w, id) || !t_filter_claims(f, k, w, id);
        }
#ifdef MTV_X86
        if (walk4_available())
            for (int i = 0; i < rounds; i++)
                bad_vector += t_filter8(f);
#endif
    }
    t_check(bad_shape == 0, "filter: targets are grouped as intended, with and without node IDs");
    t_check(missed == 0, "filter: no key that a target wants is filtered out");
#ifdef MTV_X86
    if (walk4_available())
        t_check(bad_vector == 0, "filter: the vector form agrees with the scalar form");
#else
    (void)bad_vector;
#endif

    /* a target that has its key leaves the filter */
    if (t_targets("dead,beef", 0)) {
        targets_want(1);
        g_target[0].found = 1;
        filter_build(f);
        for (int i = 0; i < rounds; i++) {
            uint64_t w[4];
            uint8_t pk[32];
            uint32_t id;

            t_random_key(pk, &id);
            t_plant(&g_target[0], pk, &id);
            t_words(w, pk);
            kept += filter_pass(f, w, id);
        }
    } else {
        kept = 1;
    }
    t_check(kept == 0 && f->groups == 1, "filter: a finished target is no longer looked for");

    xfree_aligned(f);
    targets_clear();
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
    void *state = xalloc_aligned(be->state_size);
    walk_dump dump;
    lane_t before[MAX_LANES];
    unsigned checked = 0, bad = 0;

    dump.words = malloc(slots * sizeof *dump.words);
    dump.id = malloc(slots * sizeof *dump.id);
    if (dump.words == NULL || dump.id == NULL)
        die("out of memory");
    memset(state, 0, be->state_size);
    targets_clear();
    filter_reset();                     /* nothing matches; every node ID is computed */
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
    void *state = xalloc_aligned(be->state_size);
    walk_dump dump;
    lane_t before[MAX_LANES], after[MAX_LANES];
    unsigned bad = 0;

    dump.words = malloc(slots * sizeof *dump.words);
    dump.id = malloc(slots * sizeof *dump.id);
    if (dump.words == NULL || dump.id == NULL)
        die("out of memory");
    memset(state, 0, be->state_size);
    targets_clear();
    filter_reset();
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

    secure_zero(state, be->state_size);
    xfree_aligned(state);
    free(dump.words);
    free(dump.id);
}

/* A target for the end-to-end test and the tests' own reading of it, string by string. */
typedef struct {
    const char *spec;
    int fold;
    const char *id, *key, *hex;
} t_wanted;

/*
 * Search for real, for several targets at once, capture the keys, and check
 * each one from scratch: the key pair, the node ID, the pattern of the target
 * it was found for, and that every target got exactly its share.
 */
static int t_search(const backend_t *be, const t_wanted *want, int count, int each)
{
    const int total = count * each;
    const out_mode_t saved_mode = g_run.mode;
    hit_t *hits = (hit_t *)calloc((size_t)total, sizeof *hits);
    int bad = 0;

    if (hits == NULL)
        die("out of memory");
    targets_clear();
    for (int k = 0; k < count; k++)
        bad += targets_add(want[k].spec, want[k].fold) != NULL;
    bad += g_targets != count;
    g_run.mode = OUT_CAPTURE;
    g_run.capture = hits;
    g_run.hits_wanted = targets_want((uint64_t)each);
    (void)engine_run(be, 1, 0, 60.0, 0, 0);
    bad += g_run.hits != (uint64_t)total;
    for (int k = 0; k < g_targets; k++)
        bad += g_target[k].found != (uint64_t)each;

    for (int i = 0; i < total && bad == 0; i++) {
        const uint8_t *sk = hits[i].secret;
        const int k = hits[i].target;
        uint8_t pk[32];
        uint32_t id;

        x25519_ref(pk, sk, BASE_U);
        id = crc32_ref(pk, 32);
        bad += memcmp(pk, hits[i].public_key, 32) != 0 || id != hits[i].node_id;
        bad += k < 0 || k >= count ||
               !t_expect(pk, id, want[k].id, want[k].key, want[k].hex, want[k].fold);
        bad += (sk[0] & 7) != 0 || (sk[31] & 0xc0) != 0x40;         /* clamped */
        for (int e = 0; e < i; e++)
            bad += memcmp(sk, hits[e].secret, 32) == 0;
    }

    secure_zero(hits, (size_t)total * sizeof *hits);
    free(hits);
    g_run.mode = saved_mode;
    g_run.hits_wanted = 0;
    g_run.capture = NULL;
    targets_clear();
    return bad;
}

static void test_search(const backend_t *be)
{
    static const t_wanted mixed[] = {
        {"a5", 0, "a5*", NULL, NULL},
        {"id:*5a", 0, "*5a", NULL, NULL},
        {"id:?d?l?d", 0, "?d?l?d*", NULL, NULL},
        {"key:Q", 0, NULL, "Q*", NULL},
        {"key:*8=", 0, NULL, "*8=", NULL},
        {"key:me", 1, NULL, "me*", NULL},
        {"keyhex:c0", 0, NULL, NULL, "c0*"},
        {"keyhex:*f", 0, NULL, NULL, "*f"},
        {"id:7+key:z", 0, "7*", "z*", NULL},
        {"a5", 0, "a5*", NULL, NULL},               /* the same target twice: two keys */
    };
    /* no target here reads every key's node ID, so the kernels skip the CRC */
    static const t_wanted keys_only[] = {
        {"key:Zz", 0, NULL, "Zz*", NULL},
        {"id:5+key:Q", 0, "5*", "Q*", NULL},
        {"keyhex:*3", 0, NULL, NULL, "*3"},
        {"key:?d?d", 0, NULL, "?d?d*", NULL},
    };
    int bad;

    bad = t_search(be, mixed, (int)ARRAY_LEN(mixed), 2);
    t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
            "backend %s: %d targets at once, node IDs and public keys, 2 verified keys each",
            be->name, (int)ARRAY_LEN(mixed));
    bad = t_search(be, keys_only, (int)ARRAY_LEN(keys_only), 3);
    t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
            "backend %s: public-key targets with the CRC left to the survivors", be->name);
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

/* The progress line is made of whole pieces, as many as there is room for. */
static void test_status(void)
{
    status_t wide = {"", 40, 0}, tight = {"", 16, 0}, none = {"", -3, 0};
    status_t *const all[] = {&wide, &tight, &none};

    for (size_t i = 0; i < ARRAY_LEN(all); i++) {
        status_add(all[i], "%d keys", 12345);
        status_add(all[i], ", %.1f M/s", 67.89);
        status_add(all[i], ", and a piece that is too long for a line of forty columns");
        status_add(all[i], ", x");              /* would fit, but nothing follows a gap */
    }
    t_check(strcmp(wide.text, "12345 keys, 67.9 M/s") == 0 &&
            strcmp(tight.text, "12345 keys") == 0 && none.text[0] == '\0',
            "progress line: whole pieces, as many as the terminal has room for");
}

/* Returns the number of failed checks.  `only` restricts the backend tests. */
static int self_test(int full, const backend_t *only)
{
    t_failures = 0;
    t_verbose = full;

    test_reference(full);
    test_field(full);
    test_curve(full);
    test_crc(full);
    test_targets(full);
    test_filter(full);
    test_status();
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

static void usage(int to)
{
    char text[4096];
    const int n = snprintf(text, sizeof text,
        "mtvanity %s - vanity node IDs and public keys for Meshtastic firmware 2.8+\n"
        "%s\n"
        "\n"
        "usage: mtvanity TARGET [TARGET ...] [options]\n"
        "       mtvanity --benchmark [SECONDS]\n"
        "       mtvanity --self-test\n"
        "\n"
        "A TARGET says what one key must look like.  Give several and they are\n"
        "searched together; every target gets its own key.\n"
        "  dead              node ID starts with dead         (same as id:dead)\n"
        "  id:*beef          node ID ends with beef\n"
        "  id:d..d.00d       a whole node ID with some digits left open\n"
        "  key:Mesh          public key starts with Mesh      (base64, as apps show it)\n"
        "  key:*Radio=       public key ends with Radio=\n"
        "  keyhex:c0ffee     public key written in hex starts with c0ffee\n"
        "  id:c0de+key:Me    one key that satisfies both\n"
        "\n"
        "Inside a pattern:\n"
        "  .          any one character\n"
        "  *          any run of characters, once: ab* starts, *ab ends, a*b both\n"
        "  ?d ?l ?u   a digit, a lower-case letter, an upper-case letter\n"
        "  ?h ?s ?a   one of 0-9a-f, + or /, any character   (as in hashcat masks)\n"
        "Quote a pattern that contains * or ? so that the shell leaves it alone.\n"
        "\n"
        "options:\n"
        "  --hits N           keys to find for each target (default 1)\n"
        "  -i, --ignore-case  letters in key: patterns match in either case\n"
        "  --threads N        worker threads (default: every CPU)\n"
        "  --max-keys N       give up after trying about N keys\n"
        "  --backend NAME     auto (default: each thread times the kernels on its own\n"
        "                     core and keeps the fastest), avx2 or scalar\n"
        "  --no-pin           do not bind threads to CPUs\n"
        "  --json             print each key as one JSON object per line\n"
        "  --quiet            no progress or summary on stderr\n"
        "  --benchmark [S]    measure every kernel for S seconds each (default 5)\n"
        "  --self-test        run every known-answer and cross-check test\n"
        "  --version, --help\n"
        "\n"
        "Keys go to stdout, everything else to stderr.  Install a key with\n"
        "  meshtastic --set security.private_key \"base64:...\"\n"
        "after backing up the old one: a new key is a new node identity.  Check\n"
        "first that nobody on your mesh already uses the node ID.\n",
        MTV_VERSION, MTV_ATTRIBUTION);

    if (n > 0)
        (void)os_write(to, text, (size_t)n < sizeof text ? (size_t)n : sizeof text - 1);
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
    targets_clear();
    g_run.mode = OUT_CAPTURE;
    g_run.hits_wanted = 0;
    out_printf("benchmark: %d thread%s, %.0f s per line\n", nthreads, nthreads == 1 ? "" : "s",
               seconds);

    /* each kernel alone, then (i == count) every thread choosing for itself */
    for (size_t i = 0; i <= ARRAY_LEN(BACKENDS) && !g_interrupted; i++) {
        const backend_t *be = i < ARRAY_LEN(BACKENDS) ? BACKENDS[i] : NULL;
        const char *name = be ? be->name : "auto";
        run_result r;
        char full_id[32], chosen[80] = "";

        if (forced != NULL && be != forced)
            continue;
        if (be != NULL && !be->available()) {
            out_printf("  %-7s not supported by this CPU\n", name);
            continue;
        }
        r = engine_run(be, nthreads, pin, seconds, 0, 0);
        if (r.keys == 0)
            continue;                       /* interrupted before the first window */
        format_duration(full_id, sizeof full_id, 4294967296.0 * r.seconds / (double)r.keys);
        if (be == NULL)
            snprintf(chosen, sizeof chosen, "   [%s]", r.kernels);
        out_printf("  %-7s %8.1f M keys/s   full 8-digit ID in about %s%s\n", name,
                   (double)r.keys / r.seconds / 1e6, full_id, chosen);
    }
}

/* Say what is about to be searched for, and how much work that is. */
static void print_plan(int nthreads, long long each)
{
    char count[40], extra[48] = "";
    int width = 0;

    /* first of all which build this is: a report of trouble is worth little without it */
    err_printf("mtvanity %s (source revision %d)\n", MTV_VERSION, MTV_REVISION);
    if (each > 1)
        snprintf(extra, sizeof extra, ", %lld keys %s", each, g_targets == 1 ? "wanted" : "each");
    if (g_targets == 1) {
        format_count(count, sizeof count, 1.0 / g_target[0].chance);
        err_printf("searching for %s: about %s keys per hit, %d thread%s%s\n\n",
                   g_target[0].text, count, nthreads, nthreads == 1 ? "" : "s", extra);
        return;
    }
    err_printf("searching for %d targets on %d thread%s%s\n", g_targets, nthreads,
               nthreads == 1 ? "" : "s", extra);
    for (int k = 0; k < g_targets; k++)
        width = (int)strlen(g_target[k].text) > width ? (int)strlen(g_target[k].text) : width;
    width = width > 44 ? 44 : width;
    for (int k = 0; k < g_targets; k++) {
        format_count(count, sizeof count, 1.0 / g_target[k].chance);
        err_printf("  target %-3d %-*s  about %s keys per hit\n", k + 1, width,
                   g_target[k].text, count);
    }
    err_printf("\n");
}

/* What came of it: one line in all, and one per target when there were several. */
static void print_outcome(const run_result *res)
{
    char count[40], found[64];
    int width = 0;

    format_count(count, sizeof count, (double)res->keys);
    if (g_run.hits_wanted == 1)
        snprintf(found, sizeof found, "%" PRIu64 " key%s found", g_run.hits,
                 g_run.hits == 1 ? "" : "s");
    else
        snprintf(found, sizeof found, "%" PRIu64 " of %" PRIu64 " keys found", g_run.hits,
                 g_run.hits_wanted);
    err_printf("%s; %s keys tried in %.2f s (%.1f M/s; kernels: %s)%s\n", found, count,
               res->seconds, (double)res->keys / res->seconds / 1e6,
               res->kernels[0] ? res->kernels : "none", g_interrupted ? "; interrupted" : "");

    if (g_targets > 1) {
        for (int k = 0; k < g_targets; k++)
            width = (int)strlen(g_target[k].text) > width ? (int)strlen(g_target[k].text) : width;
        width = width > 44 ? 44 : width;
        for (int k = 0; k < g_targets; k++) {
            const target_t *t = &g_target[k];
            char got[40];

            if (t->found == 0)
                snprintf(got, sizeof got, "not found");
            else if (t->wanted == 1)
                snprintf(got, sizeof got, "!%08" PRIx32, t->first_id);
            else
                snprintf(got, sizeof got, "%" PRIu64 " of %" PRIu64 " keys", t->found, t->wanted);
            err_printf("  target %-3d %-*s  %s\n", k + 1, width, t->text, got);
        }
    }
    /* a terminal that is slow to take the status line is worth knowing about */
    if (res->terminal_wait > SLOW_TERMINAL_SECONDS)
        err_printf("note: the terminal took up to %.2f s to take one status line.  If its display\n"
                   "      misbehaves, --threads N with N below the number of CPUs leaves it room.\n",
                   res->terminal_wait);
}

int main(int argc, char **argv)
{
    const char *spec[MAX_TARGETS], *backend_name = NULL, *err;
    const backend_t *be;
    int specs = 0, nthreads = 0, quiet = 0, json = 0, bench = 0, selftest = 0, pin = 1;
    int ignore_case = 0;
    double bench_seconds = 5.0;
    unsigned long long key_limit = 0;
    long long hits_wanted = 1;
    run_result res;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = i + 1 < argc ? argv[i + 1] : NULL;

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(TO_STDOUT);
            return 0;
        } else if (strcmp(a, "--version") == 0) {
            out_printf("mtvanity %s (source revision %d)\n%s\nLicense: MIT\n", MTV_VERSION,
                       MTV_REVISION, MTV_ATTRIBUTION);
            return 0;
        } else if (strcmp(a, "--threads") == 0) {
            const long long v = parse_count(next);
            if (v < 1 || v > MAX_THREADS)
                die("--threads needs a number between 1 and 1024");
            nthreads = (int)v;
            i++;
        } else if (strcmp(a, "--hits") == 0) {
            hits_wanted = parse_count(next);
            if (hits_wanted < 1 || hits_wanted > 1000000)
                die("--hits needs a number between 1 and 1000000");
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
        } else if (strcmp(a, "--ignore-case") == 0 || strcmp(a, "-i") == 0) {
            ignore_case = 1;
        } else if (strcmp(a, "--self-test") == 0) {
            selftest = 1;
        } else if (strcmp(a, "--benchmark") == 0) {
            const long long v = parse_count(next);  /* optional: the seconds per line */
            bench = 1;
            if (v >= 0) {
                if (v < 1 || v > 3600)
                    die("--benchmark takes between 1 and 3600 seconds per line");
                bench_seconds = (double)v;
                i++;
            }
        } else if (a[0] == '-' && a[1] == '-') {
            err_printf("mtvanity: unknown option %s (try --help)\n", a);
            return 2;
        } else {
            if (specs == MAX_TARGETS)
                die("too many targets (the limit is 64)");
            spec[specs++] = a;
        }
    }

    if (!selftest && !bench) {
        if (specs == 0) {
            usage(TO_STDERR);
            return 2;
        }
        targets_clear();                    /* report mistakes before doing any work */
        for (int k = 0; k < specs; k++) {
            err = targets_add(spec[k], ignore_case);
            if (err != NULL) {
                err_printf("mtvanity: bad target \"%s\": %s\n", spec[k], err);
                return 2;
            }
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

    if (selftest) {
        int failed;
        out_printf("mtvanity %s (source revision %d) self-test\n", MTV_VERSION, MTV_REVISION);
        failed = self_test(1, NULL);
        if (failed)
            out_printf("%d check(s) FAILED\n", failed);
        else
            out_printf("all checks passed\n");
        return failed ? 1 : 0;
    }

    /* be == NULL means "auto": each thread picks its own fastest kernel */
    be = NULL;
    if (backend_name != NULL && strcmp(backend_name, "auto") != 0) {
        be = backend_find(backend_name);
        if (be == NULL) {
            char names[64] = "auto";
            for (size_t i = 0; i < ARRAY_LEN(BACKENDS); i++) {
                const size_t used = strlen(names);
                snprintf(names + used, sizeof names - used, " %s", BACKENDS[i]->name);
            }
            err_printf("mtvanity: unknown --backend; choose one of: %s\n", names);
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

    /* Never search with arithmetic that has not just been checked. */
    if (self_test(0, be) != 0)
        die("the start-up self-test failed; refusing to generate keys");

    /* from here on Ctrl-C (or a polite kill) ends the search and still gets its summary */
    signal(SIGINT, on_interrupt);
    signal(SIGTERM, on_interrupt);

    if (bench) {
        run_benchmark(be, nthreads, pin, bench_seconds);
        return 0;
    }

    targets_clear();                        /* the self-test used the target list */
    for (int k = 0; k < specs; k++)
        if (targets_add(spec[k], ignore_case) != NULL)
            die("internal error: a target no longer parses");
    g_run.mode = json ? OUT_JSON : OUT_TEXT;
    g_run.hits_wanted = targets_want((uint64_t)hits_wanted);
    if (!quiet)
        print_plan(nthreads, hits_wanted);

    res = engine_run(be, nthreads, pin, 0.0, key_limit, !quiet);

    if (!quiet)
        print_outcome(&res);
    if (atomic_load(&g_run.faults) != 0)
        err_printf("mtvanity: warning: %llu candidate(s) failed verification and were "
                   "discarded; please report this\n", atomic_load(&g_run.faults));
    if (g_run.output_failed) {
        err_printf("mtvanity: a key could not be written to stdout (disk full? pipe closed?); "
                   "that key is lost and the search was stopped\n");
        return 1;
    }
    return g_run.hits >= g_run.hits_wanted ? 0 : 3;
}