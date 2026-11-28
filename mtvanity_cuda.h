/*
 * mtvanity_cuda.h - optional CUDA backend for mtvanity.c
 *
 * Authored by Claude, assisted by a human.
 *
 * Build mtvanity.c with -DMTV_CUDA and this file next to it.  Nothing from
 * the CUDA toolkit is needed, to build or to run: the NVIDIA driver library
 * (libcuda.so.1 or nvcuda.dll) is loaded at run time, and the GPU kernel is
 * PTX text that the code below writes and the driver compiles.  Without a
 * driver or a usable GPU the program says why on stderr and searches on the
 * CPU, as it does without this file.  A GPU that fails during a search hands
 * the search back to the CPU.
 *
 * HOW IT WORKS (CUDA.md and CUDA_PLAN.md have the measurements)
 *   - One walk ("lane") per GPU thread: the affine Montgomery walk of the
 *     CPU kernels, both keys M + iS and M - iS from one inverse.
 *   - Field elements are eight 32-bit limbs below 2^256, multiplied with
 *     PTX carry chains (mad.lo.cc / madc.hi.cc).  Only a published key is
 *     made canonical.
 *   - A window is 2 * CUDA_W + 1 keys.  The running product of the
 *     denominators is saved every CUDA_CH table points only; on the way back
 *     each chunk is recomputed into a small stack that stays in the L2
 *     cache.  That is one multiplication more per pair and far less memory
 *     traffic, which measured as the better trade.
 *   - CRC-32 by four tables in shared memory, skipped when no target reads
 *     the node ID; then the CPU's filter, group by group.
 *   - A key that passes is written to a record buffer.  The host derives
 *     the scalar, and the reference code of mtvanity.c rechecks every hit.
 *
 * SECURITY
 *   Scalars never leave the host.  Every lane has its own base scalar from
 *   the operating system's random generator.  The GPU sees public points.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef MTVANITY_CUDA_H
#define MTVANITY_CUDA_H

#if defined(__ANDROID__) || !(defined(_WIN32) || defined(__linux__))
#  error "MTV_CUDA is supported on Linux and Windows only"
#endif
#ifndef _WIN32
#  include <dlfcn.h>
#endif
#ifndef MTV_CUDA_LIB                             /* the driver library; a test may name another */
#  ifdef _WIN32
#    define MTV_CUDA_LIB "nvcuda.dll"
#  else
#    define MTV_CUDA_LIB "libcuda.so.1"
#  endif
#endif

#ifndef CUDA_W
#  define CUDA_W 2048u           /* table points per window */
#endif
#define CUDA_GW       (2u * CUDA_W + 1u)        /* keys per window                      */
#ifndef CUDA_CH
#  define CUDA_CH 16u            /* table points per checkpoint */
#endif
#ifndef CUDA_NT
#  define CUDA_NT 256u           /* threads per block */
#endif
#define CUDA_CAP      1024u                     /* candidate records per launch         */
#define CUDA_REC      48u                       /* bytes per record                     */
#define CUDA_STACK    (CUDA_CH * 32u * CUDA_NT)             /* bytes per block          */
#define CUDA_CKPT     ((CUDA_W / CUDA_CH) * 32u * CUDA_NT)  /* bytes per block          */
#define CUDA_SHARED   6144u                     /* CRC tables, then 64 filter groups    */
#define CUDA_BM_BITS  17                        /* bits of a group's GPU bitmap: 2^17   */
#define CUDA_BM_OFF   (CUDA_SHARED + 64u * 8u)  /* ... then the member values, 64 x 8   */
#define CUDA_FPAR     (CUDA_BM_OFF + 64u * (1u << (CUDA_BM_BITS - 3)))      /* ... bitmaps */
#define CUDA_DUMP_WIN 2u                        /* windows a dump launch may record     */
#define CUDA_NREG     360
#ifndef CUDA_BPS
#  define CUDA_BPS 2u                   /* blocks per multiprocessor */
#endif
#ifndef CUDA_MAXNREG
#  define CUDA_MAXNREG 0                /* .maxnreg for the kernels; 0: the driver decides */
#endif
#ifndef CUDA_FAST
#  define CUDA_FAST 0                   /* 1: test a single target group from registers; measured, no gain */
#endif
#ifndef CUDA_LAUNCH_S
#  define CUDA_LAUNCH_S 0.050           /* seconds a launch may take: well under 100 ms */
#endif
/*
 * Under auto the CPU threads stop only for a GPU this many times as fast as
 * they were: a small GPU beside a big CPU must not make the search slower.
 * The GPU's figure comes from its first launch, and a launch that starts
 * cold ran up to 30% faster than the GPU settled at on the test laptop.
 */
#ifndef CUDA_PARK_MIN
#  define CUDA_PARK_MIN 1.5
#endif
/*
 * Under auto the driver starts beside the searching CPU threads.  On Linux
 * (measured under WSL2) the GPU was ready sooner when two CPUs were left free
 * until its start-up was over, so in a search long enough for the GPU to
 * matter (CUDA_START_KEYS expected keys or more) that many CPU threads start
 * late.  A shorter search starts every thread at once, as a build without the
 * GPU does.  On Windows, where the search threads give way between windows,
 * free CPUs gained nothing, so none are kept.
 */
#ifndef CUDA_START_FREE
#  ifdef _WIN32
#    define CUDA_START_FREE 0
#  else
#    define CUDA_START_FREE 2
#  endif
#endif
#ifndef CUDA_START_KEYS
#  define CUDA_START_KEYS 1073741824.0          /* 2^30 */
#endif
#define CUDA_PTX_VERSION "5.0"                  /* PTX ISA 5.0 came with CUDA 8.0       */
#define CUDA_PTX_TARGET  "sm_50"

/* ---- the driver, bound at run time ---- */

typedef void *cu_ptr;
typedef unsigned long long cu_dptr;
#define CUDA_API(X) \
    X(int, cuInit, (unsigned)) \
    X(int, cuDeviceGet, (int *, int)) \
    X(int, cuDeviceGetName, (char *, int, int)) \
    X(int, cuDeviceGetAttribute, (int *, int, int)) \
    X(int, cuCtxCreate_v2, (cu_ptr *, unsigned, int)) \
    X(int, cuCtxSetCurrent, (cu_ptr)) \
    X(int, cuCtxSynchronize, (void)) \
    X(int, cuModuleLoadDataEx, (cu_ptr *, const void *, unsigned, int *, void **)) \
    X(int, cuModuleGetFunction, (cu_ptr *, cu_ptr, const char *)) \
    X(int, cuFuncGetAttribute, (int *, int, cu_ptr)) \
    X(int, cuMemAlloc_v2, (cu_dptr *, size_t)) \
    X(int, cuMemFree_v2, (cu_dptr)) \
    X(int, cuMemAllocHost_v2, (void **, size_t)) \
    X(int, cuMemcpyHtoD_v2, (cu_dptr, const void *, size_t)) \
    X(int, cuMemcpyDtoH_v2, (void *, cu_dptr, size_t)) \
    X(int, cuMemcpyHtoDAsync_v2, (cu_dptr, const void *, size_t, cu_ptr)) \
    X(int, cuMemcpyDtoHAsync_v2, (void *, cu_dptr, size_t, cu_ptr)) \
    X(int, cuMemsetD32_v2, (cu_dptr, unsigned, size_t)) \
    X(int, cuMemsetD32Async, (cu_dptr, unsigned, size_t, cu_ptr)) \
    X(int, cuStreamCreate, (cu_ptr *, unsigned)) \
    X(int, cuEventCreate, (cu_ptr *, unsigned)) \
    X(int, cuEventRecord, (cu_ptr, cu_ptr)) \
    X(int, cuEventSynchronize, (cu_ptr)) \
    X(int, cuLaunchKernel, (cu_ptr, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, \
                            unsigned, cu_ptr, void **, void **))
#define X(ret, name, args) static ret (*name) args;
CUDA_API(X)
#undef X
static int (*cuGetErrorName_opt)(int, const char **);   /* optional: names for error codes */

static int cuda_bind_once(void)
{
#ifdef _WIN32
    /* from the system directory only: a DLL planted next to the program is not picked up */
    HMODULE h = LoadLibraryExA(MTV_CUDA_LIB, NULL, 0x00000800 /* LOAD_LIBRARY_SEARCH_SYSTEM32 */);
    if (h == NULL)
        return -1;
#  define X(ret, name, args) \
    name = (ret (*) args)(void *)GetProcAddress(h, #name); if (name == NULL) return -1;
    CUDA_API(X)
    cuGetErrorName_opt = (int (*)(int, const char **))(void *)GetProcAddress(h, "cuGetErrorName");
#else
    void *h = dlopen(MTV_CUDA_LIB, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL)
        return -1;
#  define X(ret, name, args) name = (ret (*) args)dlsym(h, #name); if (name == NULL) return -1;
    CUDA_API(X)
    cuGetErrorName_opt = (int (*)(int, const char **))dlsym(h, "cuGetErrorName");
#endif
#undef X
    return 0;
}

/* "CUDA_ERROR_LAUNCH_TIMEOUT (702)", or the number alone from a driver that cannot name it */
static const char *cuda_errname(int rc, char *buf, size_t n)
{
    const char *name = NULL;

    if (cuGetErrorName_opt == NULL || cuGetErrorName_opt(rc, &name) != 0 || name == NULL)
        snprintf(buf, n, "CUDA error %d", rc);
    else
        snprintf(buf, n, "%s (%d)", name, rc);
    return buf;
}

static int cuda_bind(void)
{
    static int state;                               /* 0: not tried, 1: bound, -1: no driver library */

    if (state == 0)
        state = cuda_bind_once() == 0 ? 1 : -1;
    return state > 0 ? 0 : -1;
}

/* ---- the PTX emitter ---- */

/*
 * Deliberate faults, for testing the tests: with -DMTV_CUDA_FAULTS the hidden
 * option --cuda-fault N makes the emitter get one thing wrong.  A normal
 * build has no such switch.
 */
#define CUDA_FAULTS 36
#ifdef MTV_CUDA_FAULTS
#  define FT(k) (p->fault == (k) || (p->fault == 100 + (k) && !p->dump))
#else
#  define FT(k) 0
#endif

typedef struct {
    char *s;
    size_t n, cap;
    int nr, nq, lab, fault, dump;
    int t, s8, zero, c38, c19, x, sqc;              /* shared temporaries (u32 registers)     */
    int cv, ci, crc, idv, jr, g, src, sing, ml, mh, vl, vh, xl, xh;
    int gid, win, nwin, i, nid, ngr, dsl, fast, fsrc, fml, fmh, fvl, fvh;
    int qa, qb, qc, sb, gp, kq, qL, qT, qS, qC, qF, qD, dw;    /* u64 registers             */
    int rare[64][2], nrare;                         /* pending exact-canonical blocks         */
    int hit[8][3], nhit;                            /* pending record blocks: label, key, kind */
} ptx_t;

/* One line of PTX.  $ is a 32-bit register number, # a 64-bit one; %d %u %s as usual. */
static void E_(ptx_t *p, const char *f, ...)
{
    va_list ap;
    size_t len = strlen(f);

    if (p->n + len + 256 > p->cap) {
        p->cap = 2 * p->cap + 65536;
        p->s = (char *)realloc(p->s, p->cap);
        if (p->s == NULL)
            die("out of memory");
    }
    if (len == 0 || f[len - 1] != ':')
        p->n += (size_t)sprintf(p->s + p->n, "    ");
    va_start(ap, f);
    for (; *f; f++) {
        if (*f == '$')
            p->n += (size_t)sprintf(p->s + p->n, "%%r%d", va_arg(ap, int));
        else if (*f == '#')
            p->n += (size_t)sprintf(p->s + p->n, "%%q%d", va_arg(ap, int));
        else if (*f == '%' && f[1] == 'd')
            p->n += (size_t)sprintf(p->s + p->n, "%d", va_arg(ap, int)), f++;
        else if (*f == '%' && f[1] == 'u')
            p->n += (size_t)sprintf(p->s + p->n, "%u", va_arg(ap, unsigned)), f++;
        else if (*f == '%' && f[1] == 's')
            p->n += (size_t)sprintf(p->s + p->n, "%s", va_arg(ap, const char *)), f++;
        else
            p->s[p->n++] = *f;
    }
    va_end(ap);
    p->s[p->n++] = '\n';
    p->s[p->n] = '\0';
}
#define E(...) E_(p, __VA_ARGS__)

static int c_reg(ptx_t *p, int count)
{
    const int r = p->nr;
    p->nr += count;
    return r;
}

/* r = a op b over eight limbs, one carry chain, no reduction */
static void c_chain(ptx_t *p, const char *op, int r, int a, int b)
{
    E("%s.cc.u32 $, $, $;", op, r, a, b);
    for (int j = 1; j < 7; j++)
        E("%sc.cc.u32 $, $, $;", op, r + j, a + j, b + j);
    E("%sc.u32 $, $, $;", op, r + 7, a + 7, b + 7);
}

static void c_mov8(ptx_t *p, int r, int a)
{
    for (int j = 0; j < 8; j++)
        E("mov.u32 $, $;", r + j, a + j);
}

static void c_ld8(ptx_t *p, int r, int q, unsigned off, unsigned step)
{
    for (unsigned j = 0; j < 8; j++)
        E("ld.global.u32 $, [#+%u];", r + (int)j, q, off + j * step);
}

static void c_st8(ptx_t *p, int r, int q, unsigned off, unsigned step)
{
    for (unsigned j = 0; j < 8; j++)
        E("st.global.u32 [#+%u], $;", q, off + j * step, r + (int)j);
}

/* t[0..7] += 38 * t[8..15]; the carries end up in t[16] */
static void c_fold38(ptx_t *p)
{
    const int t = p->t, z = p->zero, cc = t + 16, k = FT(3) ? p->c19 : p->c38;

    E("mad.lo.cc.u32 $, $, $, $;", t, t + 8, k, t);
    for (int j = 1; j < 8; j++)
        E("madc.lo.cc.u32 $, $, $, $;", t + j, t + 8 + j, k, t + j);
    E("addc.u32 $, $, $;", cc, z, z);
    E("mad.hi.cc.u32 $, $, $, $;", t + 1, t + 8, k, t + 1);
    for (int j = 1; j < 7; j++)
        E("madc.hi.cc.u32 $, $, $, $;", t + j + 1, t + 8 + j, k, t + j + 1);
    E("madc.hi.u32 $, $, $, $;", cc, t + 15, k, cc);
}

/* r = t[0..15] mod p, below 2^256 */
static void c_reduce(ptx_t *p, int r)
{
    const int t = p->t, z = p->zero, cc = t + 16, x = t + 17;

    c_fold38(p);
    E("mul.lo.u32 $, $, $;", x, cc, p->c38);
    E("add.cc.u32 $, $, $;", t, t, x);
    for (int j = 1; j < 8; j++)
        E("addc.cc.u32 $, $, $;", t + j, t + j, z);
    E("addc.u32 $, $, $;", cc, z, z);
    E("mul.lo.u32 $, $, $;", x, cc, p->c38);
    E("add.u32 $, $, $;", r, t, x);
    for (int j = 1; j < 8; j++)
        E("mov.u32 $, $;", r + j, t + j);
}

/* r = a * b; r may be a or b */
static void c_mul(ptx_t *p, int r, int a, int b)
{
    const int t = p->t, z = p->zero;

    for (int j = 0; j < 8; j++)
        E("mul.lo.u32 $, $, $;", t + j, a + j, b);
    E("mad.hi.cc.u32 $, $, $, $;", t + 1, a + (FT(1) ? 1 : 0), b, t + 1);
    for (int j = 1; j < 7; j++)
        E("madc.hi.cc.u32 $, $, $, $;", t + j + 1, a + j, b, t + j + 1);
    E("madc.hi.u32 $, $, $, $;", t + 8, a + 7, b, z);
    for (int i = 1; i < 8; i++) {
        E("mad.lo.cc.u32 $, $, $, $;", t + i, a, b + i, t + i);
        for (int j = 1; j < 8; j++)
            E("madc.lo.cc.u32 $, $, $, $;", t + i + j, a + j, b + i, t + i + j);
        if (FT(2) && i == 3)
            E("mov.u32 $, 0;", t + i + 8);
        else
            E("addc.u32 $, $, $;", t + i + 8, z, z);
        E("mad.hi.cc.u32 $, $, $, $;", t + i + 1, a, b + i, t + i + 1);
        for (int j = 1; j < 7; j++)
            E("madc.hi.cc.u32 $, $, $, $;", t + i + j + 1, a + j, b + i, t + i + j + 1);
        E("madc.hi.u32 $, $, $, $;", t + i + 8, a + 7, b + i, t + i + 8);
    }
    c_reduce(p, r);
}

/* t[0..15] = a^2: the 28 cross products once, doubled, then the 8 squares */
static void c_sqcore(ptx_t *p, int a)
{
    const int t = p->t, z = p->zero;

    for (int j = 0; j < 16; j++)
        E("mov.u32 $, 0;", t + j);
    for (int i = 0; i < 7; i++) {
        E("mad.lo.cc.u32 $, $, $, $;", t + 2 * i + 1, a + i, a + i + 1, t + 2 * i + 1);
        for (int j = i + 2; j < 8; j++)
            E("madc.lo.cc.u32 $, $, $, $;", t + i + j, a + i, a + j, t + i + j);
        E("addc.u32 $, $, $;", t + i + 8, t + i + 8, z);
        if (i < 6) {
            E("mad.hi.cc.u32 $, $, $, $;", t + 2 * i + 2, a + i, a + i + 1, t + 2 * i + 2);
            for (int j = i + 2; j < 7; j++)
                E("madc.hi.cc.u32 $, $, $, $;", t + i + j + 1, a + i, a + j, t + i + j + 1);
            E("madc.hi.u32 $, $, $, $;", t + i + 8, a + i, a + 7, t + i + 8);
        } else {
            E("mad.hi.u32 $, $, $, $;", t + 14, a + 6, a + 7, t + 14);
        }
    }
    if (!FT(4)) {
        E("add.cc.u32 $, $, $;", t + 1, t + 1, t + 1);
        for (int j = 2; j < 15; j++)
            E("addc.cc.u32 $, $, $;", t + j, t + j, t + j);
        E("addc.u32 $, $, $;", t + 15, z, z);
    }
    E("mad.lo.cc.u32 $, $, $, $;", t, a, a, z);
    for (int i = 0; i < 8; i++) {
        const int d = a + (FT(5) && i == 3 ? 4 : i);
        if (i > 0)
            E("madc.lo.cc.u32 $, $, $, $;", t + 2 * i, d, a + i, t + 2 * i);
        E("madc.hi%s.u32 $, $, $, $;", i == 7 ? "" : ".cc", t + 2 * i + 1, d, a + i, t + 2 * i + 1);
    }
}

static void c_sq(ptx_t *p, int r, int a)
{
    c_sqcore(p, a);
    c_reduce(p, r);
}

/* r < 2^256  ->  the canonical value, exactly */
static void c_canon(ptx_t *p, int r)
{
    const int x = p->x, s = p->s8, z = p->zero;

    E("shr.u32 $, $, 31;", x, r + 7);
    E("and.b32 $, $, 0x7fffffff;", r + 7, r + 7);
    E("mul.lo.u32 $, $, 19;", x, x);
    E("add.cc.u32 $, $, $;", r, r, x);
    for (int j = 1; j < 7; j++)
        E("addc.cc.u32 $, $, $;", r + j, r + j, z);
    E("addc.u32 $, $, $;", r + 7, r + 7, z);
    E("add.cc.u32 $, $, 19;", s, r);
    for (int j = 1; j < 7; j++)
        E("addc.cc.u32 $, $, $;", s + j, r + j, z);
    E("addc.u32 $, $, $;", s + 7, r + 7, z);
    E("setp.gt.u32 %p1, $, 0x7fffffff;", s + 7);
    E("and.b32 $, $, 0x7fffffff;", s + 7, s + 7);
    if (!FT(9))
        for (int j = 0; j < 8; j++)
            E("selp.u32 $, $, $, %p1;", r + j, s + j, r + j);
}

/*
 * r = canonical(a^2 + e), the last step of every key.  The addition and the
 * fold of bit 255 ride on the square's own reduction; what comes out is below
 * 2^255 + 1600 and canonical unless its top word is 0x7fffffff or more, which
 * happens about once in 2^31 keys and is then finished exactly, out of line.
 */
static void c_sqadd(ptx_t *p, int r, int a, int e)
{
    const int t = p->t, z = p->zero, cc = t + 16, x = t + 17, L = p->lab++;

    c_sqcore(p, a);
    c_fold38(p);
    if (!FT(6)) {
        E("add.cc.u32 $, $, $;", t, t, e);
        for (int j = 1; j < 8; j++)
            E("addc.cc.u32 $, $, $;", t + j, t + j, e + j);
        E("addc.u32 $, $, $;", cc, cc, z);
    }
    E("shf.l.clamp.b32 $, $, $, 1;", x, t + 7, cc);
    E("and.b32 $, $, 0x7fffffff;", t + 7, t + 7);
    E("mul.lo.u32 $, $, %d;", x, x, FT(7) ? 0 : 19);
    E("add.cc.u32 $, $, $;", r, t, x);
    for (int j = 1; j < 7; j++)
        E("addc.cc.u32 $, $, $;", r + j, t + j, z);
    E("addc.u32 $, $, $;", r + 7, t + 7, z);
    if (!FT(8)) {
        E("setp.ge.u32 %p0, $, 0x7fffffff;", r + 7);
        E("@%p0 bra Lr%d;", L);
    }
    E("Lb%d:", L);
    p->rare[p->nrare][0] = L;
    p->rare[p->nrare++][1] = r;
}

static void c_sqn(ptx_t *p, int r, int n)
{
    const int L = p->lab++;

    if (n <= 2) {
        while (n-- > 0)
            c_sq(p, r, r);
        return;
    }
    E("mov.u32 $, %d;", p->sqc, n);
    E("Ls%d:", L);
    c_sq(p, r, r);
    E("sub.u32 $, $, 1;", p->sqc, p->sqc);
    E("setp.ne.u32 %p0, $, 0;", p->sqc);
    E("@%p0 bra Ls%d;", L);
}

/* r = z^(p-2): 254 squarings and 11 multiplications.  T: four spare elements. */
static void c_invert(ptx_t *p, int r, int z, const int *T)
{
    const int z11 = T[0], a = T[1], b = T[2], t = T[3];

    c_mov8(p, t, z); c_sqn(p, t, 1); c_mov8(p, z11, t);
    c_sqn(p, t, 2); c_mul(p, t, t, z);
    c_mul(p, z11, z11, t);
    c_mov8(p, a, z11); c_sqn(p, a, 1); c_mul(p, a, a, t);
    c_mov8(p, t, a); c_sqn(p, t, 5); c_mul(p, a, t, a);
    c_mov8(p, t, a); c_sqn(p, t, 10); c_mul(p, b, t, a);
    c_mov8(p, t, b); c_sqn(p, t, 20); c_mul(p, t, t, b);
    c_sqn(p, t, 10); c_mul(p, a, t, a);
    c_mov8(p, t, a); c_sqn(p, t, FT(10) ? 49 : 50); c_mul(p, b, t, a);
    c_mov8(p, t, b); c_sqn(p, t, 100); c_mul(p, t, t, b);
    c_sqn(p, t, 50); c_mul(p, t, t, a);
    c_sqn(p, t, 5); c_mul(p, r, t, z11);
}

/* p->crc = CRC-32 state of the 32-byte key before the final inversion (slicing by four) */
static void c_crc(ptx_t *p, int key)
{
    const int x = p->x, v = p->cv, ii = p->ci, crc = p->crc;

    for (int j = 0; j < 8; j++) {
        if (j == 0)
            E("not.b32 $, $;", x, key);
        else
            E("xor.b32 $, $, $;", x, crc, key + j);
        E("shl.b32 $, $, 2;", ii, x);
        E("shr.u32 $, $, %d;", ii + 1, x, FT(20) ? 7 : 6);
        E("shr.u32 $, $, 14;", ii + 2, x);
        E("shr.u32 $, $, 22;", ii + 3, x);
        for (int q = 0; q < 4; q++) {
            E("and.b32 $, $, 0x3fc;", ii + q, ii + q);
            E("cvt.u64.u32 #, $;", p->qa, ii + q);
            E("add.u64 #, #, #;", p->qa, p->qa, p->sb);
            E("ld.shared.u32 $, [#+%d];", v + q, p->qa, 1024 * (3 - q));
        }
        E("lop3.b32 $, $, $, $, 0x96;", crc, v, v + 1, v + 2);
        E("xor.b32 $, $, $;", crc, crc, v + 3);
    }
}

/* candidate j of the window into p->jr: kind 0 the centre, +1 M + iS, -1 M - iS */
static void c_jr(ptx_t *p, int kind)
{
    E("mov.u32 $, %u;", p->jr, CUDA_W);
    if (FT(14))
        kind = -kind;
    if (kind > 0)
        E("add.u32 $, $, $;", p->jr, p->jr, p->i);
    if (kind < 0)
        E("sub.u32 $, $, $;", p->jr, p->jr, p->i);
    if (FT(30))
        E("add.u32 $, $, 1;", p->jr, p->jr);
}

/*
 * What happens to a finished key (eight canonical words): its CRC-32 if a
 * target reads the node ID, in a dump launch a copy for the self-test, then
 * the filter of mtvanity.c, group by group.  A pass leaves the hot path.
 */
static void c_post(ptx_t *p, int key, int kind)
{
    const int L = p->lab++, xl = p->xl, xh = p->xh;

    E("mov.u32 $, 0;", p->idv);
    if (!p->dump) {
        E("setp.%s.u32 %p0, $, 0;", FT(28) ? "ne" : "eq", p->nid);
        E("@%p0 bra Ln%d;", L);
    }
    if (!(FT(21) && kind < 0))
        c_crc(p, key);
    E("not.b32 $, $;", p->idv, p->crc);
    E("Ln%d:", L);
    if (p->dump) {
        E("setp.eq.u32 %p2, $, 0;", p->dsl);
        E("@%p2 bra Ld%d;", L);
        c_jr(p, kind);
        E("mul.wide.u32 #, $, 36;", p->qa, p->jr);
        E("add.u64 #, #, #;", p->qa, p->qa, p->dw);
        c_st8(p, key, p->qa, 0, 4);
        E("st.global.u32 [#+32], $;", p->qa, p->idv);
        E("Ld%d:", L);
    }
    if (CUDA_FAST) {                                /* one group of one target, the usual case */
        E("setp.eq.u32 %p0, $, 0;", p->fast);
        E("@%p0 bra Lq%d;", L);
        E("setp.eq.u32 %p0, $, 4;", p->fsrc);
        E("@%p0 bra Lqi%d;", L);
        for (int s = 0; s < 3; s++) {
            E("setp.eq.u32 %p0, $, %d;", p->fsrc, s);
            E("@%p0 bra Lq%d_%d;", s, L);
        }
        E("mov.u32 $, $;", xl, key + 6);
        E("mov.u32 $, $;", xh, key + 7);
        E("bra Lqt%d;", L);
        for (int s = 0; s < 3; s++) {
            const int w = FT(24) && s == 1 ? 0 : 2 * s;
            E("Lq%d_%d:", s, L);
            E("mov.u32 $, $;", xl, key + w);
            E("mov.u32 $, $;", xh, key + w + 1);
            E("bra Lqt%d;", L);
        }
        E("Lqi%d:", L);
        E("mov.u32 $, $;", xl, p->idv);
        E("mov.u32 $, 0;", xh);
        E("Lqt%d:", L);                             /* (x ^ value) & mask, or without the mask as a fault */
        E("lop3.b32 $, $, $, $, %d;", xl, xl, p->fvl, p->fml, FT(22) ? 0x3c : 0x28);
        E("lop3.b32 $, $, $, $, %d;", xh, xh, p->fvh, p->fmh, FT(22) ? 0x3c : 0x28);
        E("or.b32 $, $, $;", xl, xl, xh);
        E("setp.eq.u32 %p3, $, 0;", xl);
        E("@%p3 bra Lh%d;", L);
        E("bra Lx%d;", L);
        E("Lq%d:", L);
    }
    E("mov.u32 $, 0;", p->g);
    E("add.u64 #, #, 4096;", p->gp, p->sb);
    E("Lg%d:", L);
    E("ld.shared.u32 $, [#+0];", p->src, p->gp);
    E("ld.shared.u32 $, [#+4];", p->sing, p->gp);
    E("ld.shared.u32 $, [#+8];", p->ml, p->gp);
    E("ld.shared.u32 $, [#+12];", p->mh, p->gp);
    E("ld.shared.u32 $, [#+16];", p->vl, p->gp);
    E("ld.shared.u32 $, [#+20];", p->vh, p->gp);
    E("setp.eq.u32 %p0, $, 4;", p->src);
    E("@%p0 bra Lgi%d;", L);
    for (int s = 0; s < 3; s++) {
        E("setp.eq.u32 %p0, $, %d;", p->src, s);
        E("@%p0 bra Lg%d_%d;", s, L);
    }
    E("mov.u32 $, $;", xl, key + 6);
    E("mov.u32 $, $;", xh, key + 7);
    E("bra Lgj%d;", L);
    for (int s = 0; s < 3; s++) {
        const int w = FT(24) && s == 1 ? 0 : 2 * s;
        E("Lg%d_%d:", s, L);
        E("mov.u32 $, $;", xl, key + w);
        E("mov.u32 $, $;", xh, key + w + 1);
        E("bra Lgj%d;", L);
    }
    E("Lgi%d:", L);
    E("mov.u32 $, $;", xl, p->idv);
    E("mov.u32 $, 0;", xh);
    E("Lgj%d:", L);
    if (!FT(22)) {
        E("and.b32 $, $, $;", xl, xl, p->ml);
        E("and.b32 $, $, $;", xh, xh, p->mh);
    }
    E("setp.eq.u32 %p0, $, 0;", p->sing);
    E("@%p0 bra Lgb%d;", L);
    E("xor.b32 $, $, $;", xl, xl, p->vl);
    E("xor.b32 $, $, $;", xh, xh, p->vh);
    E("or.b32 $, $, $;", xl, xl, xh);
    E("setp.eq.u32 %p3, $, 0;", xl);
    E("@%p3 bra Lh%d;", L);
    E("bra Lgn%d;", L);
    /*
     * Several targets of one shape.  The CPU's 4096-bit bitmap would let
     * through far more keys than there are records, so the GPU has a larger
     * one of its own and then compares with the members' values exactly.
     */
    E("Lgb%d:", L);
    E("mov.u32 $, $;", p->vl, xl);
    E("mov.u32 $, $;", p->vh, xh);
    E("cvt.u64.u32 #, $;", p->qa, xl);
    E("cvt.u64.u32 #, $;", p->qb, xh);
    E("shl.b64 #, #, 32;", p->qb, p->qb);
    E("or.b64 #, #, #;", p->qa, p->qa, p->qb);
    E("mul.lo.u64 #, #, #;", p->qa, p->qa, p->kq);
    E("shr.u64 #, #, %d;", p->qa, p->qa, 64 - CUDA_BM_BITS);
    E("cvt.u32.u64 $, #;", xl, p->qa);
    E("shr.u32 $, $, 5;", xh, xl);
    E("shl.b32 $, $, 2;", xh, xh);
    E("cvt.u64.u32 #, $;", p->qb, xh);
    E("cvt.u64.u32 #, $;", p->qa, p->g);
    E("shl.b64 #, #, %d;", p->qa, p->qa, CUDA_BM_BITS - 3);
    E("add.u64 #, #, #;", p->qa, p->qa, p->qb);
    E("add.u64 #, #, #;", p->qa, p->qa, p->qF);
    E("ld.global.u32 $, [#+%u];", xh, p->qa, CUDA_BM_OFF);
    E("and.b32 $, $, %d;", xl, xl, FT(23) ? 15 : 31);
    E("shr.u32 $, $, $;", xh, xh, xl);
    E("and.b32 $, $, 1;", xh, xh);
    E("setp.eq.u32 %p3, $, 0;", xh);
    E("@%p3 bra Lgn%d;", L);
    E("ld.shared.u32 $, [#+24];", xl, p->gp);               /* first member, how many */
    E("ld.shared.u32 $, [#+28];", xh, p->gp);
    E("cvt.u64.u32 #, $;", p->qa, xl);
    E("shl.b64 #, #, 3;", p->qa, p->qa);
    E("add.u64 #, #, #;", p->qa, p->qa, p->qF);
    E("Lgm%d:", L);
    E("ld.global.u32 $, [#+%u];", p->src, p->qa, CUDA_SHARED);
    E("ld.global.u32 $, [#+%u];", p->sing, p->qa, CUDA_SHARED + 4);
    E("xor.b32 $, $, $;", p->src, p->src, p->vl);
    E("xor.b32 $, $, $;", p->sing, p->sing, p->vh);
    E("or.b32 $, $, $;", p->src, p->src, p->sing);
    E("setp.eq.u32 %p3, $, 0;", p->src);
    E("@%p3 bra Lh%d;", L);
    E("add.u64 #, #, 8;", p->qa, p->qa);
    E("sub.u32 $, $, 1;", xh, xh);
    E("setp.ne.u32 %p0, $, 0;", xh);
    E("@%p0 bra Lgm%d;", L);
    E("Lgn%d:", L);
    E("add.u32 $, $, 1;", p->g, p->g);
    E("add.u64 #, #, 32;", p->gp, p->gp);
    E("setp.lt.u32 %p0, $, $;", p->g, p->ngr);
    E("@%p0 bra Lg%d;", L);
    E("Lx%d:", L);
    p->hit[p->nhit][0] = L;
    p->hit[p->nhit][1] = key;
    p->hit[p->nhit++][2] = kind;
}

/* claim a record; falls through with its address in qa, or jumps to `full` */
static void c_claim(ptx_t *p, const char *full, int L)
{
    E("atom.global.add.u32 $, [#], 1;", p->xl, p->qC);
    E("setp.ge.u32 %p0, $, %u;", p->xl, CUDA_CAP);
    E("@%p0 bra %s%d;", full, L);
    E("mul.wide.u32 #, $, %u;", p->qa, p->xl, CUDA_REC);
    E("add.u64 #, #, #;", p->qa, p->qa, p->qC);
    if (FT(29))
        E("add.u32 $, $, 1;", p->xh, p->gid);
    E("st.global.u32 [#+16], $;", p->qa, FT(29) ? p->xh : p->gid);
    E("st.global.u32 [#+20], $;", p->qa, FT(31) ? p->nwin : p->win);
}

/* the out-of-line blocks collected so far */
static void c_flush(ptx_t *p)
{
    for (int n = 0; n < p->nhit; n++) {
        const int L = p->hit[n][0], key = p->hit[n][1];
        E("Lh%d:", L);
        c_claim(p, "Lx", L);
        c_jr(p, p->hit[n][2]);
        E("st.global.u32 [#+24], $;", p->qa, p->jr);
        E("st.global.u32 [#+28], $;", p->qa, p->idv);
        c_st8(p, key, p->qa, 32, 4);
        E("bra Lx%d;", L);
    }
    for (int n = 0; n < p->nrare; n++) {
        E("Lr%d:", p->rare[n][0]);
        c_canon(p, p->rare[n][1]);
        E("bra Lb%d;", p->rare[n][0]);
    }
    p->nhit = p->nrare = 0;
}

/* entry point, registers, parameters, the shared-memory copy of CRC tables and groups */
static void c_begin(ptx_t *p, const char *name)
{
    int tid, cta;

    p->nr = p->nq = 0;
    E_(p, ".visible .entry %s(.param .u64 p_lanes, .param .u64 p_tab, .param .u64 p_scr, "
          ".param .u64 p_cand, .param .u64 p_fpar, .param .u64 p_dump, .param .u32 p_nwin, "
          ".param .u32 p_flags)", name);
    if (CUDA_MAXNREG != 0)
        E(".maxnreg %d", CUDA_MAXNREG);
    E("{:");
    p->n -= 2, p->s[p->n++] = '\n', p->s[p->n] = '\0';      /* "{" alone on its line */
    E(".reg .u32 %r<%d>;", CUDA_NREG);
    E(".reg .u64 %q<20>;");
    E(".reg .pred %p<4>;");
    E(".shared .align 4 .b8 shm[%u];", CUDA_SHARED);
    p->t = c_reg(p, 18); p->s8 = c_reg(p, 8); p->cv = c_reg(p, 4); p->ci = c_reg(p, 4);
    p->zero = c_reg(p, 1); p->c38 = c_reg(p, 1); p->c19 = c_reg(p, 1); p->x = c_reg(p, 1);
    p->sqc = c_reg(p, 1); p->crc = c_reg(p, 1); p->idv = c_reg(p, 1); p->jr = c_reg(p, 1);
    p->g = c_reg(p, 1); p->src = c_reg(p, 1); p->sing = c_reg(p, 1); p->ml = c_reg(p, 1);
    p->mh = c_reg(p, 1); p->vl = c_reg(p, 1); p->vh = c_reg(p, 1); p->xl = c_reg(p, 1);
    p->xh = c_reg(p, 1); p->gid = c_reg(p, 1); p->win = c_reg(p, 1); p->nwin = c_reg(p, 1);
    p->i = c_reg(p, 1); p->nid = c_reg(p, 1); p->ngr = c_reg(p, 1); p->dsl = c_reg(p, 1);
    tid = c_reg(p, 1); cta = c_reg(p, 1);
    p->fast = c_reg(p, 1); p->fsrc = c_reg(p, 1); p->fml = c_reg(p, 1); p->fmh = c_reg(p, 1);
    p->fvl = c_reg(p, 1); p->fvh = c_reg(p, 1);
    p->qa = 0; p->qb = 1; p->qc = 2; p->sb = 3; p->gp = 4; p->kq = 5; p->qL = 6; p->qT = 7;
    p->qS = 8; p->qC = 9; p->qF = 10; p->qD = 11; p->dw = 12; p->nq = 13;
    E("ld.param.u64 #, [p_lanes];", p->qL);
    E("ld.param.u64 #, [p_tab];", p->qT);
    E("ld.param.u64 #, [p_scr];", p->qS);
    E("ld.param.u64 #, [p_cand];", p->qC);
    E("ld.param.u64 #, [p_fpar];", p->qF);
    E("ld.param.u64 #, [p_dump];", p->qD);
    E("ld.param.u32 $, [p_nwin];", p->nwin);
    E("ld.param.u32 $, [p_flags];", p->x);
    E("and.b32 $, $, 0xff;", p->ngr, p->x);
    E("shr.u32 $, $, 8;", p->nid, p->x);
    E("mov.u32 $, %tid.x;", tid);
    E("mov.u32 $, %ctaid.x;", cta);
    E("mad.lo.u32 $, $, %u, $;", p->gid, cta, CUDA_NT, tid);
    E("mov.u32 $, 0;", p->zero);
    E("mov.u32 $, 38;", p->c38);
    E("mov.u32 $, 19;", p->c19);
    E("mov.u64 #, 0x9e3779b97f4a7c15;", p->kq);
    E("mov.u64 #, shm;", p->sb);
    E("cvt.u64.u32 #, $;", p->qc, tid);                     /* qc = 4 * tid, kept */
    E("shl.b64 #, #, 2;", p->qc, p->qc);
    E("add.u64 #, #, #;", p->qa, p->qc, p->qF);
    E("add.u64 #, #, #;", p->qb, p->qc, p->sb);
    for (unsigned q = 0; q < CUDA_SHARED / (4 * CUDA_NT); q++) {
        E("ld.global.u32 $, [#+%u];", p->x, p->qa, 4 * CUDA_NT * q);
        E("st.shared.u32 [#+%u], $;", p->qb, 4 * CUDA_NT * q, p->x);
    }
    E("bar.sync 0;");
    E("add.u64 #, #, 4096;", p->qa, p->sb);                  /* the first group's record, kept in registers */
    E("ld.shared.u32 $, [#+0];", p->fsrc, p->qa);
    E("ld.shared.u32 $, [#+4];", p->x, p->qa);
    E("ld.shared.u32 $, [#+8];", p->fml, p->qa);
    E("ld.shared.u32 $, [#+12];", p->fmh, p->qa);
    E("ld.shared.u32 $, [#+16];", p->fvl, p->qa);
    E("ld.shared.u32 $, [#+20];", p->fvh, p->qa);
    E("setp.eq.u32 %p0, $, 1;", p->ngr);
    E("selp.u32 $, $, $, %p0;", p->fast, p->x, p->zero);     /* one group of one target: the fast test */
    E("cvt.u64.u32 #, $;", p->gp, cta);                     /* gp = block number, for c_walk */
}

static void c_end(ptx_t *p)
{
    E("ret;");
    c_flush(p);
    E("}:");
    p->n -= 2, p->s[p->n++] = '\n', p->s[p->n] = '\0';
    if (p->nr > CUDA_NREG || p->nq > 20)
        die("internal error: the PTX emitter ran out of registers");
}

/* The self-test's window into the field code: r = op(a, b) for one pair per thread. */
static void c_test_kernel(ptx_t *p)
{
    int a, b, r, T[4];

    c_begin(p, "mtv_test");
    a = c_reg(p, 8); b = c_reg(p, 8); r = c_reg(p, 8);
    for (int k = 0; k < 4; k++)
        T[k] = c_reg(p, 8);
    E("mul.wide.u32 #, $, 96;", p->qa, p->gid);
    E("add.u64 #, #, #;", p->qL, p->qL, p->qa);
    c_ld8(p, a, p->qL, 0, 4);
    c_ld8(p, b, p->qL, 32, 4);
    for (int op = 0; op < 5; op++) {
        E("setp.eq.u32 %p0, $, %d;", p->nwin, op);
        E("@%p0 bra Lop%d;", op);
    }
    c_crc(p, a);                                            /* 5: CRC-32 of a */
    E("not.b32 $, $;", r, p->crc);
    E("bra Lend;");
    E("Lop0:"); c_mul(p, r, a, b); E("bra Lend;");
    E("Lop1:"); c_sq(p, r, a); E("bra Lend;");
    E("Lop2:"); c_sqadd(p, r, a, b); E("bra Lend;");
    E("Lop3:"); c_mov8(p, r, a); c_canon(p, r); E("bra Lend;");
    E("Lop4:"); c_invert(p, r, a, T);
    E("Lend:");
    c_st8(p, r, p->qL, 64, 4);
    c_end(p);
}

/* The search kernel: p_nwin windows for every lane.  `dump` adds the self-test's copy of every key. */
static void c_walk_kernel(ptx_t *p, int dump)
{
    const unsigned LS = 4 * CUDA_NT, ROW = 32 * CUDA_NT;
    int uM, vM, bias, nvm, P, acc, inv, d, lamp, lamm, ev, k0, k1, pw, cj, T[4];
    const int lp = 13, tp = 14, pq = 15, cq = 16;          /* u64: lane record, table, stack, checkpoint */

    p->dump = dump;
    c_begin(p, dump ? "mtv_dump" : "mtv_walk");
    p->nq = 17;
    uM = c_reg(p, 8); vM = c_reg(p, 8); bias = c_reg(p, 8); nvm = c_reg(p, 8); P = c_reg(p, 8);
    acc = c_reg(p, 8); inv = c_reg(p, 8); d = c_reg(p, 8); lamp = c_reg(p, 8); lamm = c_reg(p, 8);
    ev = c_reg(p, 8); k0 = c_reg(p, 8); k1 = c_reg(p, 8); pw = c_reg(p, 8); cj = c_reg(p, 1);
    for (int k = 0; k < 4; k++)
        T[k] = c_reg(p, 8);

    /* this lane's centre; p = 2^255 - 19 in pw */
    E("cvt.u64.u32 #, $;", p->qa, p->gid);
    if (FT(32))
        E("add.u64 #, #, 1;", p->qa, p->qa);
    E("shl.b64 #, #, 6;", p->qa, p->qa);
    E("add.u64 #, #, #;", lp, p->qa, p->qL);
    c_ld8(p, uM, lp, 0, 4);
    c_ld8(p, vM, lp, 32, 4);
    E("mov.u32 $, 0xffffffed;", pw);
    for (int j = 1; j < 7; j++)
        E("mov.u32 $, 0xffffffff;", pw + j);
    E("mov.u32 $, 0x7fffffff;", pw + 7);
    /* qS = this thread's stack column; qD = its checkpoint column (the stacks of all blocks come first) */
    E("mov.u32 $, %nctaid.x;", p->x);
    E("cvt.u64.u32 #, $;", p->qa, p->x);
    E("mul.lo.u64 #, #, %u;", p->qb, p->qa, CUDA_STACK);
    E("add.u64 #, #, #;", p->qb, p->qb, p->qS);             /* start of the checkpoints */
    if (dump) {
        E("mul.lo.u64 #, #, %u;", p->qa, p->qa, 4 * CUDA_NT);       /* the dump map: a word per lane */
        E("add.u64 #, #, #;", p->dw, p->qa, p->qD);
        E("mul.wide.u32 #, $, 4;", p->qa, p->gid);
        E("add.u64 #, #, #;", p->qa, p->qa, p->qD);
        E("ld.global.u32 $, [#];", p->dsl, p->qa);
        E("sub.u32 $, $, 1;", p->x, p->dsl);
        E("mul.wide.u32 #, $, %u;", p->qa, p->x, CUDA_DUMP_WIN * CUDA_GW * 36);
        E("add.u64 #, #, #;", p->dw, p->dw, p->qa);         /* this lane's dump, window 0 */
    }
    E("mul.lo.u64 #, #, %u;", p->qa, p->gp, CUDA_CKPT);
    E("add.u64 #, #, #;", p->qD, p->qb, p->qa);
    E("add.u64 #, #, #;", p->qD, p->qD, p->qc);
    E("mul.lo.u64 #, #, %u;", p->qa, p->gp, CUDA_STACK);
    E("add.u64 #, #, #;", p->qS, p->qS, p->qa);
    E("add.u64 #, #, #;", p->qS, p->qS, p->qc);
    E("mov.u32 $, 0;", p->win);

    E("Lwin:");
    if (!FT(33))
        c_post(p, uM, 0);                                   /* the centre itself */
    c_chain(p, "sub", bias, pw, uM);                        /* bias = -u_M, nvm = -v_M: sums stay below 2p */
    c_chain(p, "sub", nvm, pw, vM);
    c_ld8(p, d, p->qT, 0, 4);
    c_chain(p, "add", P, d, bias);                          /* u_Q - u_M starts the product */
    E("add.u64 #, #, 96;", tp, p->qT);
    E("mov.u64 #, #;", cq, p->qD);
    E("mov.u32 $, 0;", p->i);

    /* pass 1: the product of all u_i - u_M, saved before every chunk */
    E("Lf2:");
    c_st8(p, FT(15) ? d : P, cq, 0, LS);
    E("add.u64 #, #, %u;", cq, cq, ROW);
    E("mov.u32 $, 0;", cj);
    E("Lf:");
    c_ld8(p, d, tp, 0, 4);
    c_chain(p, "add", d, d, bias);
    c_mul(p, P, P, d);
    E("add.u64 #, #, 96;", tp, tp);
    E("add.u32 $, $, 1;", p->i, p->i);
    E("add.u32 $, $, 1;", cj, cj);
    E("setp.lt.u32 %p0, $, %u;", cj, CUDA_CH);
    E("@%p0 bra Lf;");
    E("setp.lt.u32 %p0, $, %u;", p->i, CUDA_W);
    E("@%p0 bra Lf2;");

    /* a zero product means the centre sits on a table point: tell the host */
    c_mov8(p, T[0], P);
    c_canon(p, T[0]);
    E("or.b32 $, $, $;", p->x, T[0], T[0] + 1);
    for (int j = 2; j < 8; j++)
        E("or.b32 $, $, $;", p->x, p->x, T[0] + j);
    if (!FT(27)) {
        E("setp.eq.u32 %p0, $, 0;", p->x);
        E("@%p0 bra Lzero;");
    }
    E("Lzb:");
    c_invert(p, acc, P, T);

    /* backward, chunk by chunk: recompute the chunk's products into the stack, then use them */
    E("Lb2:");
    E("sub.u64 #, #, %u;", cq, cq, ROW);
    for (unsigned j = 0; j < 8; j++)
        E("ld.global.u32 $, [#+%u];", P + (int)(FT(16) ? (j + 1) % 8 : j), cq, j * LS);
    E("sub.u64 #, #, %u;", tp, tp, 96 * CUDA_CH);
    E("mov.u64 #, #;", pq, p->qS);
    E("mov.u32 $, 0;", cj);
    E("Lre:");
    c_st8(p, P, pq, 0, LS);
    c_ld8(p, d, tp, 0, 4);
    c_chain(p, "add", d, d, bias);
    c_mul(p, P, P, d);
    E("add.u64 #, #, 96;", tp, tp);
    E("add.u64 #, #, %u;", pq, pq, ROW);
    E("add.u32 $, $, 1;", cj, cj);
    E("setp.lt.u32 %p0, $, %u;", cj, CUDA_CH);
    E("@%p0 bra Lre;");
    E("Lb:");
    E("sub.u64 #, #, 96;", tp, tp);
    E("sub.u64 #, #, %u;", pq, pq, ROW);
    c_ld8(p, d, pq, FT(17) ? ROW : 0, LS);
    c_mul(p, inv, acc, d);                                  /* 1 / (u_i - u_M) */
    c_ld8(p, d, tp, FT(11) ? 32 : 0, 4);
    c_chain(p, "add", d, d, bias);
    c_mul(p, acc, acc, d);
    c_ld8(p, d, tp, 32, 4);
    c_chain(p, "add", lamp, d, FT(13) ? bias : nvm);        /* v_i - v_M */
    c_chain(p, "add", lamm, d, vM);                         /* v_i + v_M */
    c_mul(p, lamp, inv, lamp);
    c_mul(p, lamm, inv, lamm);
    c_ld8(p, d, tp, FT(12) ? 160 : 64, 4);
    c_chain(p, "add", ev, d, bias);                         /* -A - u_i - u_M */
    c_sqadd(p, k0, lamp, ev);
    c_sqadd(p, k1, lamm, ev);
    c_post(p, k0, 1);
    c_post(p, k1, -1);
    E("sub.u32 $, $, 1;", p->i, p->i);
    E("sub.u32 $, $, 1;", cj, cj);
    E("setp.ne.u32 %p0, $, 0;", cj);
    E("@%p0 bra Lb;");
    E("setp.ne.u32 %p0, $, 0;", p->i);
    E("@%p0 bra Lb2;");

    /* move the centre: M' = M + Q, with acc = 1 / (u_Q - u_M) */
    c_ld8(p, d, p->qT, 32, 4);
    c_chain(p, "add", d, d, nvm);
    c_mul(p, lamp, acc, d);
    c_ld8(p, d, p->qT, 64, 4);
    c_chain(p, "add", ev, d, bias);
    c_sqadd(p, T[0], lamp, ev);                             /* u' */
    c_chain(p, "sub", d, pw, T[0]);
    c_chain(p, "add", d, d, uM);
    c_mul(p, lamm, lamp, d);                                /* lambda * (u_M - u') */
    c_canon(p, lamm);
    c_chain(p, "add", lamm, lamm, FT(18) ? vM : nvm);
    c_canon(p, lamm);
    c_mov8(p, uM, T[0]);
    c_mov8(p, vM, lamm);
    if (dump)
        E("add.u64 #, #, %u;", p->dw, p->dw, CUDA_GW * 36);
    E("add.u32 $, $, 1;", p->win, p->win);
    E("setp.lt.u32 %p0, $, $;", p->win, p->nwin);
    E("@%p0 bra Lwin;");
    if (!FT(26)) {
        c_st8(p, uM, lp, 0, 4);
        c_st8(p, vM, lp, 32, 4);
    }
    E("ret;");
    E("Lzero:");
    c_claim(p, "Lz", 0);
    E("mov.u32 $, 0xffffffff;", p->x);
    E("st.global.u32 [#+24], $;", p->qa, p->x);
    E("Lz0:");
    E("bra Lzb;");
    c_end(p);
}

/* The whole module.  The caller frees it. */
static char *cuda_ptx(int fault)
{
    ptx_t e, *p = &e;

    memset(p, 0, sizeof *p);
    p->fault = fault;
    E_(p, ".version " CUDA_PTX_VERSION ":");
    E_(p, ".target " CUDA_PTX_TARGET ":");
    E_(p, ".address_size 64:");
    for (char *c = p->s; *c; c++)
        if (*c == ':')
            *c = ' ';
    c_test_kernel(p);
    c_walk_kernel(p, 0);
    c_walk_kernel(p, 1);
    return p->s;
}

/* ---- the host side ---- */

static struct {
    int      state;                 /* 0: not tried yet, 1: usable, -1: not usable (see why) */
    char     why[240], name[100];
    int      fault;                 /* --cuda-fault, in builds that have it                  */
    int      auto_on;               /* --backend auto, and there is a driver library to try  */
    int      forced;                /* --backend cuda                                        */
    int      cpu_join;              /* CPU threads searching alongside it                    */
    int      async;                 /* this run brings the GPU up beside a CPU search        */
    atomic_int ran;                 /* the GPU has finished a launch of this run             */
    atomic_int park;                /* the CPU threads that only bridged the start-up stop   */
    atomic_int up;                  /* the GPU worker is back from starting the driver       */
    int      orphan;                /* the search ended first: nobody waits for that worker  */
    int      note;                  /* auto without a GPU: the plan says so, and why         */
    int      say;                   /* this run may tell stderr what the GPU does            */
    atomic_int lost;                /* the GPU failed during this run: the CPU takes over    */
    atomic_int none;                /* this run's GPU turned out not to be usable            */
    atomic_int retime;              /* goes up when the CPU threads must time their kernels  */
                                    /* again: the GPU's start-up is over and they search on  */
    char     lost_why[200];
    double   rate;                  /* keys per second of this run's first launch            */
    int      err_rc;                /* the first driver call that failed, and its error      */
    const char *err_call;
    unsigned fail_at;               /* --cuda-fail-at, in builds that have it                */
    double   ph[8];                 /* seconds: driver, context, kernel, buffers, tables,    */
                                    /* seeding, start-up check                               */
    cu_ptr   ctx, mod, f_walk, f_dump, f_test, stream, ev[2];
    unsigned blocks, lanes;
    int      regs;                  /* registers per thread of the search kernel             */
    cu_dptr  d_lanes, d_tab, d_scr, d_cand[2], d_fpar[2], d_dump;
    uint32_t *h_cand[2];            /* pinned: record count, then the records                */
    uint8_t  *h_fpar[2];            /* pinned: filter parameters as the kernel reads them    */
    mpoint   *comb;                 /* [64][15]: d * 16^k * B, for seeding                   */
} g_cuda;

/* The search state of the GPU worker.  The lanes hold the secrets. */
typedef struct {
    lane_t   *lane;                 /* [g_cuda.lanes]                                        */
    uint8_t  *gone;                 /* lane retired or void; its records are dropped         */
    unsigned *reseed, nreseed;      /* lanes waiting for a fresh base                        */
    uint32_t (*centre)[16];         /* scratch for seeding                                   */
    unsigned  nwin;                 /* windows in the next launch                            */
    int       head, inflight;       /* the oldest launch in flight and how many there are    */
    unsigned  slot_nwin[2];
    const filter_t *slot_f[2];      /* the filter each slot's parameters were made from      */
    double    t_last;
    unsigned long long last_keys, overflow;
    unsigned  launches;             /* finished in this search (for --cuda-fail-at)          */
    int       dead;
    int       raw;                  /* a test reads the records itself                       */
} cuda_walk_t;

PRINTF_LIKE(1, 2)
static int cuda_fail(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(g_cuda.why, sizeof g_cuda.why, fmt, ap);
    va_end(ap);
    g_cuda.state = -1;
    return -1;
}

/* Note the first driver call of a search that fails, for the message that hands it to the CPU. */
static int cuda_ck(int rc, const char *call)
{
    if (rc != 0 && g_cuda.err_rc == 0) {
        g_cuda.err_rc = rc;
        g_cuda.err_call = call;
    }
    return rc;
}
#define CUDA_TRY(fn, ...) cuda_ck(fn(__VA_ARGS__), #fn)

/* The GPU gave wrong answers in a check, or a driver call failed on the way: it is not used. */
static void cuda_fail_check(void)
{
    char e[80];

    if (g_cuda.err_rc != 0)
        (void)cuda_fail("the GPU failed its start-up check (%s: %s)", g_cuda.err_call,
                        cuda_errname(g_cuda.err_rc, e, sizeof e));
    else
        (void)cuda_fail("the GPU failed its start-up check");
}

/* A line on stderr from the GPU worker, clear of the status line. */
PRINTF_LIKE(1, 2)
static void cuda_say(const char *fmt, ...)
{
    char text[400];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(text, sizeof text, fmt, ap);
    va_end(ap);
    output_lock();
    progress_clear();
    err_printf("%s\n", text);
    output_unlock();
}

static void cuda_words(uint32_t w[8], const fe *a)
{
    uint64_t q[4];

    fe_to_words(q, a);
    memcpy(w, q, 32);
}

/* table entry: u, v and m = -u - A, canonical, so that the kernel's sums need no reduction */
static void cuda_entry(uint32_t *e, const mpoint *pt)
{
    const fe zero = {{0, 0, 0, 0, 0}};
    fe m;

    fe_add(&m, &pt->u, &FE_A);
    fe_sub(&m, &zero, &m);
    cuda_words(e, &pt->u);
    cuda_words(e + 8, &pt->v);
    cuda_words(e + 16, &m);
}

/* tab[0] = the window step (2W+1) S, tab[i] = i S for i = 1..W, S = 8B; one spare entry */
static int cuda_table(uint32_t *tab)
{
    mpoint pt = TAB[1], t, q;

    for (unsigned i = 1; i <= CUDA_W; i++) {
        if (i == 2 && mont_double(&pt, &TAB[1]))
            return -1;
        if (i > 2 && mont_add(&pt, &pt, &TAB[1]))
            return -1;
        cuda_entry(tab + 24 * i, &pt);
    }
    if (mont_double(&t, &pt) || mont_add(&q, &t, &TAB[1]))
        return -1;
    cuda_entry(tab, &q);
    memset(tab + 24 * (CUDA_W + 1), 0, 96);
    return 0;
}

static int cuda_comb(void)
{
    mpoint b = BASE;

    if (g_cuda.comb != NULL)
        return 0;
    g_cuda.comb = (mpoint *)malloc(64 * 15 * sizeof(mpoint));
    if (g_cuda.comb == NULL)
        return -1;
    for (int k = 0; k < 64; k++) {
        mpoint *row = g_cuda.comb + 15 * k;
        row[0] = b;
        if (mont_double(&row[1], &b))
            return -1;
        for (int d = 3; d <= 15; d++)
            if (mont_add(&row[d - 1], &row[d - 2], &b))
                return -1;
        for (int n = 0; n < 4; n++)
            if (mont_double(&b, &b))
                return -1;
    }
    return 0;
}

static void cuda_draw_base(uint8_t base[32])
{
    for (;;) {
        uint8_t top = 0xff;

        os_random(base, 32);
        base[0] &= 248;
        base[31] &= 127;
        base[31] |= 64;
        for (int i = 9; i < 31; i++)
            top &= base[i];
        if (!(top == 0xff && base[31] == 0x7f))     /* as lane_seed: stay clear of 2^255 */
            return;
    }
}

/*
 * Fresh bases for the lanes idx[0..n) (idx == NULL: lanes 0..n-1) and their
 * first window centres, as the kernel reads them.  Every lane gets its own
 * scalar from the operating system; nothing is derived from another lane.
 * The centres (base + 8W) B are summed from a table of d * 16^k * B, all
 * lanes in step, so that one inversion serves every lane's addition.
 */
static void cuda_seed_lanes(lane_t *lane, const unsigned *idx, unsigned n, uint32_t (*out)[16])
{
    mpoint *acc = (mpoint *)malloc(n * sizeof *acc);
    fe *pre = (fe *)malloc(n * sizeof *pre);
    uint8_t (*k)[32] = (uint8_t (*)[32])malloc(n * 32), *bad = (uint8_t *)calloc(n, 1);

    if (acc == NULL || pre == NULL || k == NULL || bad == NULL)
        die("out of memory");
    for (unsigned i = 0; i < n; i++) {
        lane_t *l = &lane[idx ? idx[i] : i];
        cuda_draw_base(l->base);
        l->win = 0;
        scalar_offset(k[i], l->base, CUDA_W);
        acc[i] = g_cuda.comb[15 * 63 + (k[i][31] >> 4) - 1];    /* bit 254 is set: never zero */
    }
    for (int w = 62; w >= 0; w--) {
        const mpoint *row = g_cuda.comb + 15 * w;
        fe run = FE_ONE, den, inv, lam, t, u3;

        for (unsigned i = 0; i < n; i++) {
            const unsigned dg = (k[i][w >> 1] >> ((w & 1) * 4)) & 15;
            if (dg == 0 || bad[i])
                continue;
            fe_sub(&den, &row[dg - 1].u, &acc[i].u);
            if (fe_is_zero(&den)) {
                bad[i] = 1;                         /* never by chance; redone below */
                continue;
            }
            pre[i] = run;
            fe_mul(&run, &run, &den);
        }
        fe_invert(&run, &run);
        for (unsigned i = n; i-- > 0;) {
            const unsigned dg = (k[i][w >> 1] >> ((w & 1) * 4)) & 15;
            const mpoint *q = &row[dg - 1];
            if (dg == 0 || bad[i])
                continue;
            fe_sub(&den, &q->u, &acc[i].u);
            fe_mul(&inv, &run, &pre[i]);
            fe_mul(&run, &run, &den);
            fe_sub(&t, &q->v, &acc[i].v);
            fe_mul(&lam, &t, &inv);
            fe_sq(&u3, &lam);
            fe_sub(&u3, &u3, &FE_A);
            fe_sub(&u3, &u3, &acc[i].u);
            fe_sub(&u3, &u3, &q->u);
            fe_sub(&t, &acc[i].u, &u3);
            fe_mul(&t, &lam, &t);
            fe_sub(&acc[i].v, &t, &acc[i].v);
            acc[i].u = u3;
        }
    }
    for (unsigned i = 0; i < n; i++) {
        lane_t *l = &lane[idx ? idx[i] : i];
        while (bad[i]) {                            /* the slow way, one lane */
            cuda_draw_base(l->base);
            scalar_offset(k[i], l->base, CUDA_W);
            bad[i] = mont_scalarmult_base(&acc[i], k[i]) != 0;
        }
        cuda_words(out[i], &acc[i].u);
        cuda_words(out[i] + 8, &acc[i].v);
    }
    secure_zero(k, (size_t)n * 32);
    secure_zero(acc, n * sizeof *acc);
    secure_zero(pre, n * sizeof *pre);
    free(acc); free(pre); free(k); free(bad);
}

static void cuda_phase(int k, double *t)
{
    const double now = now_seconds();

    g_cuda.ph[k] = now - *t;
    *t = now;
}

/* Bring the GPU up once: driver, device, kernel, buffers, tables.  0 if it can be used. */
static int cuda_start(void)
{
    static char log[4096];
    int opt[2] = {5, 6}, dev = 0, sms = 0, rc;      /* CU_JIT_ERROR_LOG_BUFFER and its size */
    void *val[2] = {log, (void *)(size_t)sizeof log};
    uint32_t *tab;
    char *ptx, e[80];
    double t = now_seconds();

    if (g_cuda.state != 0)
        return g_cuda.state > 0 ? 0 : -1;
    if (cuda_bind() != 0)
        return cuda_fail("no NVIDIA driver library (%s) was found", MTV_CUDA_LIB);
    if ((rc = cuInit(0)) != 0 || (rc = cuDeviceGet(&dev, 0)) != 0)
        return cuda_fail("the NVIDIA driver found no usable GPU (%s)", cuda_errname(rc, e, sizeof e));
    cuda_phase(0, &t);
    if (cuDeviceGetName(g_cuda.name, (int)sizeof g_cuda.name, dev) != 0)
        snprintf(g_cuda.name, sizeof g_cuda.name, "GPU");
    if ((rc = cuDeviceGetAttribute(&sms, 16, dev)) != 0)        /* multiprocessor count */
        return cuda_fail("the GPU did not say how large it is (%s)", cuda_errname(rc, e, sizeof e));
    if (sms < 1)
        return cuda_fail("the GPU did not say how large it is");
    if ((rc = cuCtxCreate_v2(&g_cuda.ctx, 4, dev)) != 0)        /* waits block, not spin */
        return cuda_fail("no CUDA context on %s (%s)", g_cuda.name, cuda_errname(rc, e, sizeof e));
    cuda_phase(1, &t);
    ptx = cuda_ptx(g_cuda.fault);
    log[0] = '\0';
    if ((rc = cuModuleLoadDataEx(&g_cuda.mod, ptx, 2, opt, val)) != 0) {
        free(ptx);
        log[100] = '\0';
        for (char *c = log; *c; c++)                /* one line */
            if (*c == '\n' || *c == '\r')
                *c = ' ';
        return cuda_fail("the driver refused the kernel (PTX " CUDA_PTX_VERSION ", %s)%s%s",
                         cuda_errname(rc, e, sizeof e), log[0] ? ": " : "", log);
    }
    free(ptx);
    if ((rc = cuModuleGetFunction(&g_cuda.f_walk, g_cuda.mod, "mtv_walk")) != 0 ||
        (rc = cuModuleGetFunction(&g_cuda.f_dump, g_cuda.mod, "mtv_dump")) != 0 ||
        (rc = cuModuleGetFunction(&g_cuda.f_test, g_cuda.mod, "mtv_test")) != 0)
        return cuda_fail("the compiled kernel is incomplete (%s)", cuda_errname(rc, e, sizeof e));
    (void)cuFuncGetAttribute(&g_cuda.regs, 4, g_cuda.f_walk);

    cuda_phase(2, &t);
    /* two blocks for every multiprocessor; fewer if memory is short */
    for (g_cuda.blocks = CUDA_BPS * (unsigned)sms;; g_cuda.blocks /= 2) {
        if (g_cuda.blocks == 0)
            return cuda_fail("not enough free memory on %s (%s)", g_cuda.name, cuda_errname(rc, e, sizeof e));
        if ((rc = cuMemAlloc_v2(&g_cuda.d_scr, (size_t)g_cuda.blocks * (CUDA_STACK + CUDA_CKPT))) == 0)
            break;
    }
    g_cuda.lanes = g_cuda.blocks * CUDA_NT;
    if ((rc = cuMemAlloc_v2(&g_cuda.d_tab, (CUDA_W + 2) * 96)) != 0 ||
        (rc = cuMemAlloc_v2(&g_cuda.d_lanes, (size_t)g_cuda.lanes * 64 + 64)) != 0 ||
        (rc = cuStreamCreate(&g_cuda.stream, 0)) != 0)
        return cuda_fail("not enough free memory on %s (%s)", g_cuda.name, cuda_errname(rc, e, sizeof e));
    for (int s = 0; s < 2; s++)
        if ((rc = cuMemAlloc_v2(&g_cuda.d_cand[s], 16 + CUDA_CAP * CUDA_REC)) != 0 ||
            (rc = cuMemAlloc_v2(&g_cuda.d_fpar[s], CUDA_FPAR)) != 0 ||
            (rc = cuMemAllocHost_v2((void **)&g_cuda.h_cand[s], 16 + CUDA_CAP * CUDA_REC)) != 0 ||
            (rc = cuMemAllocHost_v2((void **)&g_cuda.h_fpar[s], CUDA_FPAR)) != 0 ||
            (rc = cuEventCreate(&g_cuda.ev[s], 1)) != 0)        /* waiting blocks */
            return cuda_fail("not enough free memory on %s (%s)", g_cuda.name, cuda_errname(rc, e, sizeof e));
    g_cuda.d_dump = g_cuda.d_tab;                               /* any valid address */

    cuda_phase(3, &t);
    tab = (uint32_t *)malloc((CUDA_W + 2) * 96);
    if (tab == NULL || cuda_table(tab) != 0 || cuda_comb() != 0 ||
        cuMemcpyHtoD_v2(g_cuda.d_tab, tab, (CUDA_W + 2) * 96) != 0) {
        free(tab);
        return cuda_fail("the table of multiples could not be set up");
    }
    free(tab);
    cuda_phase(4, &t);
    g_cuda.state = 1;
    return 0;
}

static int cuda_available(void)
{
    return cuda_start() == 0;
}

/* the kernel's view of a filter: CRC tables, a record per group, member values, the groups' bitmaps */
static void cuda_fpar(uint8_t *buf, const filter_t *f)
{
    uint32_t *w = (uint32_t *)(void *)buf;

    memset(buf, 0, CUDA_FPAR);
    for (int q = 0; q < 4; q++)
        memcpy(w + 256 * q, CRC_TAB[q], 1024);
    memcpy(buf + CUDA_SHARED, f->member_value, 64 * 8);
    for (unsigned g = 0; g < f->groups; g++) {
        const fgroup_t *fg = &f->group[g];
        uint32_t *rec = w + 1024 + 8 * g;

        rec[0] = fg->source;
        rec[1] = fg->single;
        rec[2] = (uint32_t)fg->mask;
        rec[3] = (uint32_t)(fg->mask >> 32);
        rec[4] = (uint32_t)fg->value;
        rec[5] = (uint32_t)(fg->value >> 32);
        rec[6] = fg->first;
        rec[7] = fg->count;
        for (unsigned m = fg->first; !fg->single && m < (unsigned)fg->first + fg->count; m++) {
            const uint32_t slot = (uint32_t)((f->member_value[m] * 0x9e3779b97f4a7c15ULL) >> (64 - CUDA_BM_BITS));
            buf[CUDA_BM_OFF + ((size_t)g << (CUDA_BM_BITS - 3)) + (slot >> 3)] |= (uint8_t)(1u << (slot & 7));
        }
    }
}

/* Put one launch of nwin windows on the stream: parameters, kernel, read-back, event. */
static int cuda_queue(cuda_walk_t *w, int s, unsigned nwin, int dump, const filter_t *f)
{
    unsigned flags;
    void *args[8] = {&g_cuda.d_lanes, &g_cuda.d_tab, &g_cuda.d_scr, &g_cuda.d_cand[s],
                     &g_cuda.d_fpar[s], &g_cuda.d_dump, &nwin, &flags};

    if (f == NULL)
        f = filter_current();
    flags = f->groups | (f->need_id ? 256u : 0u);
    if (w->slot_f[s] != f) {
        cuda_fpar(g_cuda.h_fpar[s], f);
        if (CUDA_TRY(cuMemcpyHtoDAsync_v2, g_cuda.d_fpar[s], g_cuda.h_fpar[s], CUDA_FPAR, g_cuda.stream) != 0)
            return -1;
        w->slot_f[s] = f;
    }
    w->slot_nwin[s] = nwin;
    return CUDA_TRY(cuMemsetD32Async, g_cuda.d_cand[s], 0, 1, g_cuda.stream) != 0 ||
           CUDA_TRY(cuLaunchKernel, dump ? g_cuda.f_dump : g_cuda.f_walk, g_cuda.blocks, 1, 1,
                    CUDA_NT, 1, 1, 0, g_cuda.stream, args, NULL) != 0 ||
           CUDA_TRY(cuMemcpyDtoHAsync_v2, g_cuda.h_cand[s], g_cuda.d_cand[s],
                    16 + CUDA_CAP * CUDA_REC, g_cuda.stream) != 0 ||
           CUDA_TRY(cuEventRecord, g_cuda.ev[s], g_cuda.stream) != 0 ? -1 : 0;
}

static void cuda_retire(cuda_walk_t *w, unsigned lane)
{
    w->gone[lane] = 1;
    w->reseed[w->nreseed++] = lane;
}

/*
 * Wait for the launch in slot s and act on its records.  A record names a
 * lane, a window of the launch and a candidate; the host forms the scalar
 * and hands the key to the same exact check and reference recomputation as
 * the CPU kernels.  A lane that has given a key is retired.
 */
static int cuda_finish(cuda_walk_t *w, int s)
{
    const uint32_t *c = g_cuda.h_cand[s];
    const filter_t *f = w->slot_f[s];
    const unsigned nwin = w->slot_nwin[s];
    unsigned n;

    if (CUDA_TRY(cuEventSynchronize, g_cuda.ev[s]) != 0)
        return -1;
    n = c[0] < CUDA_CAP ? c[0] : CUDA_CAP;
    w->overflow += c[0] > CUDA_CAP;
    for (unsigned r = 0; r < n && !w->raw; r++) {
        const uint32_t *rec = c + 4 + 12 * r;
        const unsigned l = rec[0];
        uint64_t words[4];
        lane_t at;

        if (atomic_load_explicit(&g_run.stop, memory_order_relaxed) && rec[2] != 0xffffffffu)
            continue;                               /* every key is found: the rest is not wanted */
        if (l >= g_cuda.lanes || rec[1] >= nwin || (rec[2] >= CUDA_GW && rec[2] != 0xffffffffu)) {
            atomic_fetch_add(&g_run.faults, 1);     /* not a record this kernel can write */
            continue;
        }
        if (w->gone[l])
            continue;
        if (rec[2] == 0xffffffffu) {                /* the centre met a table point */
            cuda_retire(w, l);
            continue;
        }
        scalar_offset(at.base, w->lane[l].base, (w->lane[l].win + rec[1]) * (uint64_t)CUDA_GW);
        at.win = 0;
        memcpy(words, rec + 4, 32);
        if (engine_candidate(f, &at, rec[2], words, f->need_id, rec[3]))
            cuda_retire(w, l);
        secure_zero(&at, sizeof at);
    }
    for (unsigned l = 0; l < g_cuda.lanes; l++)
        w->lane[l].win += nwin;
    w->last_keys += (unsigned long long)g_cuda.lanes * CUDA_GW * nwin;
    w->inflight--;
    w->head ^= 1;
    return 0;
}

/* Give the retired lanes new bases.  No launch may be in flight. */
static int cuda_reseed(cuda_walk_t *w)
{
    cuda_seed_lanes(w->lane, w->reseed, w->nreseed, w->centre);
    for (unsigned i = 0; i < w->nreseed; i++) {
        const unsigned l = w->reseed[i];
        if (CUDA_TRY(cuMemcpyHtoD_v2, g_cuda.d_lanes + 64ull * l, w->centre[i], 64) != 0)
            return -1;
        w->gone[l] = 0;
    }
    w->nreseed = 0;
    return 0;
}

/*
 * The GPU failed during a search.  Keys already printed stand, and records
 * still on the GPU are lost, never printed wrong.  engine_run sees g_cuda.lost,
 * says so, and has the CPU carry the search on.  No later run tries this GPU.
 */
static void cuda_lost(cuda_walk_t *w)
{
    char e[80];

    w->dead = 1;
    if (atomic_load(&g_cuda.lost))
        return;
    if (g_cuda.err_rc != 0)
        snprintf(g_cuda.lost_why, sizeof g_cuda.lost_why, "%s: %s", g_cuda.err_call,
                 cuda_errname(g_cuda.err_rc, e, sizeof e));
    else
        snprintf(g_cuda.lost_why, sizeof g_cuda.lost_why, "a driver call failed");
    snprintf(g_cuda.why, sizeof g_cuda.why, "it stopped working during a search (%s)", g_cuda.lost_why);
    g_cuda.state = -1;
    atomic_store(&g_cuda.lost, 1);
}

/*
 * How many of a search's CPU threads wait until the GPU's start-up is over
 * (see CUDA_START_FREE).  CPUs the search leaves idle anyway count as free,
 * and at most one thread in eight waits.  With threads bound to CPUs it is
 * the first CPUs that stay free: see engine_run.
 */
static int cuda_spare(int cpu_threads)
{
    int spare = CUDA_START_FREE - (os_cpu_count() - cpu_threads);

    if (spare > cpu_threads / 8)
        spare = cpu_threads / 8;
    if (spare < 0 || g_targets == 0 || targets_expected_keys() < CUDA_START_KEYS)
        spare = 0;
    return spare;
}

typedef struct {
    thread_t thread;
    lane_t  *lane;
    unsigned n;
    uint32_t (*out)[16];
} cuda_seed_job;

static void cuda_seed_job_run(void *arg)
{
    cuda_seed_job *j = (cuda_seed_job *)arg;

    cuda_seed_lanes(j->lane, NULL, j->n, j->out);
}

/* cuda_seed_lanes for lanes 0..n-1, shared out among the CPUs */
static void cuda_seed_all(lane_t *lane, unsigned n, uint32_t (*out)[16])
{
    enum { MOST = 64 };
    cuda_seed_job job[MOST];
    int started[MOST], jobs = os_cpu_count();

    if (jobs > MOST)
        jobs = MOST;
    if ((unsigned)jobs > n / 256)
        jobs = (int)(n / 256);
    if (jobs < 1)
        jobs = 1;
    for (int k = 0; k < jobs; k++) {
        const unsigned from = (unsigned)((uint64_t)n * (unsigned)k / (unsigned)jobs);
        const unsigned to = (unsigned)((uint64_t)n * (unsigned)(k + 1) / (unsigned)jobs);

        job[k].lane = lane + from;
        job[k].n = to - from;
        job[k].out = out + from;
        job[k].thread.fn = cuda_seed_job_run;
        job[k].thread.arg = &job[k];
        started[k] = k + 1 < jobs && thread_start(&job[k].thread) == 0;
        if (!started[k])
            cuda_seed_job_run(&job[k]);
    }
    for (int k = 0; k < jobs; k++)
        if (started[k])
            thread_join(&job[k].thread);
}

static int cuda_quick_check(cuda_walk_t *w);

/*
 * A search begins: bring the GPU up if this is the first one, then give every
 * lane a fresh base.  Under --backend auto all of this runs beside the CPU
 * threads, which are already searching; the GPU joins when it has passed its
 * start-up check.  A worker without a usable GPU idles until the search ends.
 */
static void cuda_seed(void *state)
{
    cuda_walk_t *w = (cuda_walk_t *)state;
    unsigned n;
    double t;

    memset(w, 0, sizeof *w);
    w->dead = 1;
    g_cuda.err_rc = 0;
    t = cuda_start() != 0 ? -1.0 : 0.0;
    atomic_store(&g_cuda.up, 1);
    if (t < 0.0) {                                  /* the CPU threads are searching already */
        if (g_cuda.async && g_cuda.say && !atomic_load(&g_run.stop))
            cuda_say("no usable GPU: %s; searching on the CPU", g_cuda.why);
        atomic_store(&g_cuda.none, 1);
        return;
    }
    if (atomic_load(&g_run.stop))
        return;
    n = g_cuda.lanes;
    (void)cuCtxSetCurrent(g_cuda.ctx);
    w->lane = (lane_t *)calloc(n, sizeof(lane_t));
    w->gone = (uint8_t *)calloc(n, 1);
    w->reseed = (unsigned *)calloc(n, sizeof(unsigned));
    w->centre = (uint32_t (*)[16])calloc(n, 64);
    if (w->lane == NULL || w->gone == NULL || w->reseed == NULL || w->centre == NULL)
        die("out of memory");
    t = now_seconds();
    cuda_seed_all(w->lane, n, w->centre);
    cuda_phase(5, &t);
    w->nwin = 1;
    if (CUDA_TRY(cuMemcpyHtoD_v2, g_cuda.d_lanes, w->centre, (size_t)n * 64) != 0) {
        cuda_lost(w);
        return;
    }
    if (g_cuda.async && !atomic_load(&g_run.stop)) {    /* never search with a kernel that was not just checked */
        const int bad = cuda_quick_check(w);
        cuda_phase(6, &t);
        if (bad) {
            cuda_fail_check();
            cuda_say("mtvanity: warning: %s, so it is not used; searching on the CPU", g_cuda.why);
            atomic_store(&g_cuda.none, 1);
            return;
        }
    }
    w->dead = 0;
}

/*
 * One step of the search: keep two launches queued, wait for the older one
 * and act on its records.  With two in flight the GPU never waits for this
 * thread to wake up.
 */
static void cuda_window(void *state, const walk_dump *dump)
{
    cuda_walk_t *w = (cuda_walk_t *)state;
    double now;
    unsigned ran;

    (void)dump;
    w->last_keys = 0;
    if (w->dead) {
        sleep_ms(10);
        return;
    }
    (void)cuCtxSetCurrent(g_cuda.ctx);
    if (w->inflight == 0)
        w->t_last = now_seconds();
    while (w->inflight < 2) {
        if (cuda_queue(w, (w->head + w->inflight) & 1, w->nwin, 0, NULL) != 0) {
            cuda_lost(w);
            return;
        }
        w->inflight++;
    }
    ran = w->slot_nwin[w->head];
    if (cuda_finish(w, w->head) != 0) {
        cuda_lost(w);
        return;
    }
    now = now_seconds();
    if (!atomic_load(&g_cuda.ran) && now > w->t_last)  /* the first launch: is the GPU worth the CPU's stop? */
        g_cuda.rate = (double)g_cuda.lanes * CUDA_GW * ran / (now - w->t_last);
    atomic_store(&g_cuda.ran, 1);
#ifdef MTV_CUDA_FAULTS
    if (g_cuda.fail_at != 0 && ++w->launches == g_cuda.fail_at) {     /* as if the GPU were lost here */
        (void)cuda_ck(999, "--cuda-fail-at");
        cuda_lost(w);
        return;
    }
#endif
    if (now > w->t_last) {                          /* whole windows, at most about 50 ms a launch */
        const double per_window = (now - w->t_last) / ran;
        const double want = CUDA_LAUNCH_S / per_window;
        w->nwin = want < 1.0 ? 1 : want > 16.0 ? 16 : (unsigned)want;      /* rounded down */
    }
    w->t_last = now;
    if (w->nreseed != 0) {                          /* rare: empty the pipeline first */
        if ((w->inflight != 0 && cuda_finish(w, w->head) != 0) || cuda_reseed(w) != 0)
            cuda_lost(w);
        w->t_last = now_seconds();
    }
}

/* End of a search: let the GPU finish what is queued, then wipe the secrets. */
static void cuda_drain(void *state)
{
    cuda_walk_t *w = (cuda_walk_t *)state;

    if (g_cuda.ctx != NULL) {                       /* also after a failure: wait, then wipe */
        (void)cuCtxSetCurrent(g_cuda.ctx);
        (void)cuCtxSynchronize();
    }
    for (int s = 0; s < 2; s++)
        if (g_cuda.h_cand[s] != NULL)
            secure_zero(g_cuda.h_cand[s], 16 + CUDA_CAP * CUDA_REC);
    if (w->lane != NULL)
        secure_zero(w->lane, g_cuda.lanes * sizeof(lane_t));
    if (w->centre != NULL)
        secure_zero(w->centre, (size_t)g_cuda.lanes * 64);
    free(w->lane); free(w->gone); free(w->reseed); free(w->centre);
    memset(w, 0, sizeof *w);
}

static void cuda_set_centre(void *state, unsigned lane, const mpoint *c)
{
    uint32_t rec[16];
    mpoint pt = *c;

    (void)state;
    mpoint_canonical(&pt);
    cuda_words(rec, &pt.u);
    cuda_words(rec + 8, &pt.v);
    (void)cuCtxSetCurrent(g_cuda.ctx);
    (void)cuMemcpyHtoD_v2(g_cuda.d_lanes + 64ull * lane, rec, 64);
}

static const backend_t BACKEND_CUDA = {
    "cuda", 1, sizeof(cuda_walk_t), cuda_available, cuda_seed, cuda_window, cuda_set_centre, 1,
};

/* ---- self-test, in the style of mtvanity.c section 13 ---- */

static void t_check(int ok, const char *fmt, ...);
static void test_search(const backend_t *be);

/* eight words below 2^256 as a field element */
static void cuda_fe(fe *r, const uint32_t w[8])
{
    uint64_t q[4];
    fe top = {{0, 0, 0, 0, 0}};

    memcpy(q, w, 32);
    top.v[0] = 19 * (q[3] >> 63);
    fe_from_words(r, q);
    fe_add(r, r, &top);
}

/* one launch of nwin windows, waited for, its records acted on */
static int cuda_run1(cuda_walk_t *w, unsigned nwin, int dump, const filter_t *f)
{
    w->slot_f[0] = NULL;
    w->head = 0;
    w->inflight = 1;
    if (cuda_queue(w, 0, nwin, dump, f) != 0 || cuda_finish(w, 0) != 0)
        return -1;
    w->head = 0;
    return w->nreseed != 0 ? cuda_reseed(w) : 0;
}

/* The kernel's field code, inversion and CRC against the scalar code, value by value. */
static int cuda_field_core(unsigned bad[6])
{
    enum { N = 16 * CUDA_NT };
    static const char *const edge[] = {
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "0000000000000000000000000000000000000000000000000000000000000080",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000",
    };
    uint32_t (*io)[24] = (uint32_t (*)[24])malloc(N * 96);
    unsigned flags = 257;
    cu_dptr d_io = 0;
    int ok;

    if (io == NULL)
        die("out of memory");
    memset(bad, 0, 6 * sizeof bad[0]);
    (void)cuCtxSetCurrent(g_cuda.ctx);
    cuda_fpar(g_cuda.h_fpar[0], &FILTER_NONE);
    ok = cuMemAlloc_v2(&d_io, N * 96) == 0 &&
         cuMemcpyHtoD_v2(g_cuda.d_fpar[0], g_cuda.h_fpar[0], CUDA_FPAR) == 0;
    for (unsigned op = 0; op < 6 && ok; op++) {
        void *args[8] = {&d_io, &g_cuda.d_tab, &g_cuda.d_scr, &g_cuda.d_cand[0], &g_cuda.d_fpar[0],
                         &g_cuda.d_dump, &op, &flags};

        os_random(io, N * 96);
        for (unsigned i = 0; i < 81; i++) {         /* every pair of the special values */
            hex_decode((uint8_t *)io[i], edge[i / 9], 32);
            hex_decode((uint8_t *)(io[i] + 8), edge[i % 9], 32);
        }
        if (op == 2)                                /* squares that end at or next to p */
            for (unsigned i = 81; i < N; i += 2) {
                fe a, t, s = {{i % 80, 0, 0, 0, 0}};
                cuda_fe(&a, io[i]);
                fe_sq(&t, &a);
                fe_sub(&t, &s, &t);
                cuda_words(io[i] + 8, &t);
                if (i % 4 == 1 && io[i][15] < 0x7fffff00u) {    /* the same value, plus p */
                    uint64_t carry = 0;
                    for (int k = 0; k < 8; k++) {
                        carry += (uint64_t)io[i][8 + k] + (k == 0 ? 0xffffffedu : k == 7 ? 0x7fffffffu : 0xffffffffu);
                        io[i][8 + k] = (uint32_t)carry;
                        carry >>= 32;
                    }
                }
            }
        ok = cuMemcpyHtoD_v2(d_io, io, N * 96) == 0 &&
             cuLaunchKernel(g_cuda.f_test, 16, 1, 1, CUDA_NT, 1, 1, 0, NULL, args, NULL) == 0 &&
             cuMemcpyDtoH_v2(io, d_io, N * 96) == 0;
        for (unsigned i = 0; i < N && ok; i++) {
            const uint32_t *r = io[i] + 16;
            uint32_t want[8];
            fe a, b, t, got;

            cuda_fe(&a, io[i]);
            cuda_fe(&b, io[i] + 8);
            cuda_fe(&got, r);
            if (op == 0) {
                fe_mul(&t, &a, &b);
            } else if (op == 1) {
                fe_sq(&t, &a);
            } else if (op == 2) {
                fe_sq(&t, &a);
                fe_add(&t, &t, &b);
            } else if (op == 3) {
                t = a;
            } else if (op == 4) {
                fe_invert(&t, &a);
            } else {
                bad[5] += r[0] != crc32_ref((const uint8_t *)io[i], 32);
                continue;
            }
            cuda_words(want, &t);
            bad[op] += op == 2 || op == 3 ? memcmp(want, r, 32) != 0 : !fe_equal(&t, &got);
        }
    }
    if (d_io != 0)
        (void)cuMemFree_v2(d_io);
    free(io);
    return ok;
}

static void cuda_test_field(void)
{
    unsigned bad[6];
    const int ok = cuda_field_core(bad), n = 16 * CUDA_NT;

    t_check(ok && bad[0] + bad[1] + bad[4] == 0,
            "cuda: multiplication, squaring and inversion equal the scalar code (%d values each)", n);
    t_check(ok && bad[2] + bad[3] == 0, "cuda: the canonical forms are exact, at the extremes too");
    t_check(ok && bad[5] == 0, "cuda: CRC-32 of %d keys equals the reference", n);
}

/*
 * Plant the key at candidate j of lane l as a target of some shape; 0 if it
 * is found and ends its base.  A launch tries over a hundred million keys, so
 * every planted target fixes at least 48 bits: a 32-bit node ID alone would
 * be matched by another key of the same launch once in 35 launches.
 */
static int cuda_plant(cuda_walk_t *w, unsigned l, unsigned j, int shape)
{
    static const char dots[] = "................................................";
    const out_mode_t saved_mode = g_run.mode;
    const lane_t before = w->lane[l];
    hit_t hits[4];
    uint8_t sk[32], pk[32];
    char spec[160], hex[65];
    int bad = 0;

    scalar_offset(sk, before.base, before.win * (uint64_t)CUDA_GW + j);
    x25519_ref(pk, sk, BASE_U);
    hex_encode(hex, pk, 32);
    if (shape == 0)                                 /* both patterns; the filter reads the key */
        snprintf(spec, sizeof spec, "id:%08" PRIx32 "+keyhex:%.8s", crc32_ref(pk, 32), hex);
    else if (shape <= 3)                            /* a pattern inside key word 1, 2 or 3 */
        snprintf(spec, sizeof spec, "keyhex:%.*s%.12s", 16 * shape, dots, hex + 16 * shape);
    else                                            /* read by its node ID: alone (5) or among three (4) */
        snprintf(spec, sizeof spec, "id:%08" PRIx32 "+keyhex:%.6s", crc32_ref(pk, 32), hex);
    targets_clear();
    if (targets_add(spec, 0) == NULL) {             /* refused: a reserved node ID, five in 2^32 */
        if (shape == 4)
            bad += targets_add("id:12345678+keyhex:000000", 0) != NULL ||
                   targets_add("id:9abcdef0+keyhex:ffffff", 0) != NULL;
        g_run.mode = OUT_CAPTURE;
        g_run.capture = hits;
        g_run.hits = 0;
        g_run.hits_wanted = targets_want(1);
        atomic_store(&g_run.stop, 0);
        filter_reset();
        filter_publish();
        bad += cuda_run1(w, 1, 0, NULL) != 0;
        bad += g_run.hits != 1 || memcmp(hits[0].secret, sk, 32) != 0;
        bad += memcmp(before.base, w->lane[l].base, 32) == 0 || w->lane[l].win != 0;
        filter_reset();
    }
    targets_clear();
    g_run.mode = saved_mode;
    g_run.hits_wanted = 0;
    g_run.capture = NULL;
    secure_zero(hits, sizeof hits);
    secure_zero(sk, sizeof sk);
    return bad;
}

static void cuda_test_walk(int full)
{
    const unsigned n = g_cuda.lanes, pick[8] = {0, 1, 31, 32, 255, 256, n - 257, n - 1};
    static const unsigned place[] = {CUDA_W, CUDA_W + 17, 0, CUDA_GW - 1, CUDA_W + 16, CUDA_W - 16, CUDA_W - 15,
                                     CUDA_W + 1, CUDA_W - 1, 1, CUDA_GW - 16, CUDA_GW - 17};
    static const int shape_of[] = {5, 1, 0, 2, 3, 4, 5, 0, 4, 1, 2, 3};
    const size_t slot = (size_t)CUDA_DUMP_WIN * CUDA_GW * 36;
    cuda_walk_t *w = (cuda_walk_t *)xalloc_aligned(sizeof *w);
    uint32_t *map = (uint32_t *)calloc(n, 4), *dump = (uint32_t *)malloc(8 * slot);
    lane_t before[8], l5, l6, lz;
    unsigned bad = 0, checked = 0;
    int ok;

    if (map == NULL || dump == NULL)
        die("out of memory");
    targets_clear();
    filter_reset();
    atomic_store(&g_run.stop, 0);
    cuda_seed(w);

    for (int k = 0; k < 4; k++) {                   /* the batched seeding against the slow way */
        const unsigned l = pick[2 * k + 1];
        uint32_t want[16];
        uint8_t sk[32];
        mpoint c;

        scalar_offset(sk, w->lane[l].base, CUDA_W);
        bad += mont_scalarmult_base(&c, sk) != 0;
        mpoint_canonical(&c);
        cuda_words(want, &c.u);
        cuda_words(want + 8, &c.v);
        bad += memcmp(want, w->centre[l], 64) != 0;
        secure_zero(sk, sizeof sk);
    }
    t_check(bad == 0, "cuda: every lane has its own base, and the batched centres equal the slow way");

    /* three windows with every key of eight lanes copied out */
    for (unsigned k = 0; k < 8; k++) {
        map[pick[k]] = k + 1;
        before[k] = w->lane[pick[k]];
    }
    bad = 0;
    ok = cuMemAlloc_v2(&g_cuda.d_dump, (size_t)n * 4 + 8 * slot) == 0 &&
         cuMemsetD32_v2(g_cuda.d_dump, 0, n) == 0 &&
         cuMemcpyHtoD_v2(g_cuda.d_dump, map, (size_t)n * 4) == 0 &&
         cuda_run1(w, CUDA_DUMP_WIN, 1, NULL) == 0 &&
         cuMemcpyDtoH_v2(dump, g_cuda.d_dump + (size_t)n * 4, 8 * slot) == 0;
    for (unsigned k = 0; k < 8 && ok; k++) {
        bad += w->lane[pick[k]].win != CUDA_DUMP_WIN;
        for (unsigned win = 0; win < CUDA_DUMP_WIN; win++)
            for (unsigned j = 0; j < CUDA_GW; j++) {
                const unsigned i = j > CUDA_W ? j - CUDA_W : CUDA_W - j;
                const int edge = i <= 1 || i == CUDA_W;
                const int joint = i % CUDA_CH <= 1 && (i < 3 * CUDA_CH || i > CUDA_W - 2 * CUDA_CH);
                const uint32_t *rec = dump + ((size_t)(k * CUDA_DUMP_WIN + win) * CUDA_GW + j) * 9;
                uint8_t sk[32], want[32];

                if (!(edge || (win == 0 && ((full && (k == 0 || k == 7)) || (joint && k < 4)))))
                    continue;
                scalar_offset(sk, before[k].base, (uint64_t)win * CUDA_GW + j);
                x25519_ref(want, sk, BASE_U);
                bad += memcmp(want, rec, 32) != 0 || rec[8] != crc32_ref(want, 32);
                checked++;
                secure_zero(sk, sizeof sk);
            }
    }
    t_check(ok && bad == 0, "backend cuda: %u keys equal the reference ladder and CRC", checked);
    if (g_cuda.d_dump != g_cuda.d_tab)
        (void)cuMemFree_v2(g_cuda.d_dump);
    g_cuda.d_dump = g_cuda.d_tab;

    bad = 0;
    for (size_t r = 0; r < (full ? ARRAY_LEN(place) : 2); r++)
        bad += (unsigned)cuda_plant(w, pick[(7 * r + 7) % 8], place[r], shape_of[r]);
    t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
            "backend cuda: a key planted in a window is found, and one key ends its base");

    if (full) {
        hit_t hit;

        l5 = w->lane[5]; l6 = w->lane[6]; lz = w->lane[n - 2];
        cuda_set_centre(w, 5, &TAB[7]);
        cuda_set_centre(w, n - 2, &TAB[HALF_WINDOW]);
        atomic_store(&g_run.stop, 0);
        bad = cuda_run1(w, 1, 0, NULL) != 0;
        bad += memcmp(l5.base, w->lane[5].base, 32) == 0 || w->lane[5].win != 0;
        bad += memcmp(lz.base, w->lane[n - 2].base, 32) == 0 || w->lane[n - 2].win != 0;
        bad += memcmp(l6.base, w->lane[6].base, 32) != 0 || w->lane[6].win != l6.win + 1;
        bad += (unsigned)cuda_plant(w, 5, CUDA_W + 3, 5) + (unsigned)cuda_plant(w, 6, 7, 5);
        t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
                "backend cuda: a centre on a table point is abandoned cleanly");

        targets_clear();                            /* one hex digit: far more passes than records */
        bad = targets_add("0", 0) != NULL;
        g_run.mode = OUT_CAPTURE;
        g_run.capture = &hit;
        g_run.hits = 0;
        g_run.hits_wanted = targets_want(1);
        filter_reset();
        filter_publish();
        atomic_store(&g_run.stop, 0);
        bad += cuda_run1(w, 1, 0, NULL) != 0 || g_run.hits != 1 || w->overflow == 0;
        t_check(bad == 0 && atomic_load(&g_run.faults) == 0,
                "backend cuda: a filter that lets through more keys than there are records is survived");
        filter_reset();
        targets_clear();
        g_run.hits_wanted = 0;
        g_run.capture = NULL;
        secure_zero(&hit, sizeof hit);
        secure_zero(&l5, sizeof l5); secure_zero(&l6, sizeof l6); secure_zero(&lz, sizeof lz);
    }
    secure_zero(before, sizeof before);
    cuda_drain(w);
    xfree_aligned(w);
    free(map);
    free(dump);
}

/*
 * The check a search under --backend auto makes while the CPU threads are
 * already searching: the field code against the scalar code, keys of four
 * lanes against the reference ladder and CRC, and a key planted through a
 * filter of its own, whose record must come back.  Unlike the self-test it
 * does not touch the target list, which the running search is using.
 * Returns 0 if the GPU may be used.
 */
static int cuda_quick_check(cuda_walk_t *w)
{
    static const unsigned at[] = {0, 1, CUDA_W - CUDA_CH, CUDA_W - 1, CUDA_W, CUDA_W + 1,
                                  CUDA_W + CUDA_CH + 1, CUDA_GW - 2, CUDA_GW - 1};
    const unsigned n = g_cuda.lanes, pick[4] = {0, CUDA_NT - 1, CUDA_NT, n - 1};
    const unsigned pl = n - 2, pj = CUDA_W + 2 * CUDA_CH + 1;
    const size_t slot = (size_t)CUDA_DUMP_WIN * CUDA_GW * 36;
    uint32_t *map = (uint32_t *)calloc(n, 4), *dump = (uint32_t *)malloc((size_t)CUDA_GW * 36);
    filter_t *f = (filter_t *)xalloc_aligned(sizeof *f);
    const uint32_t *c = g_cuda.h_cand[0];
    lane_t before[4];
    unsigned bad[6], wrong = 0;
    uint8_t sk[32], pk[32];
    uint32_t id;
    int ok, found = 0;

    if (map == NULL || dump == NULL)
        die("out of memory");
    wrong += !cuda_field_core(bad);
    for (int k = 0; k < 6; k++)
        wrong += bad[k];
    for (unsigned k = 0; k < 4; k++) {
        map[pick[k]] = k + 1;
        before[k] = w->lane[pick[k]];
    }
    ok = cuMemAlloc_v2(&g_cuda.d_dump, (size_t)n * 4 + 4 * slot) == 0 &&
         cuMemsetD32_v2(g_cuda.d_dump, 0, n) == 0 &&
         cuMemcpyHtoD_v2(g_cuda.d_dump, map, (size_t)n * 4) == 0 &&
         cuda_run1(w, 1, 1, NULL) == 0;
    for (unsigned k = 0; k < 4 && ok; k++) {
        ok = cuMemcpyDtoH_v2(dump, g_cuda.d_dump + (size_t)n * 4 + k * slot, (size_t)CUDA_GW * 36) == 0;
        for (size_t i = 0; i < ARRAY_LEN(at) && ok; i++) {
            const uint32_t *rec = dump + (size_t)at[i] * 9;

            scalar_offset(sk, before[k].base, at[i]);
            x25519_ref(pk, sk, BASE_U);
            wrong += memcmp(pk, rec, 32) != 0 || rec[8] != crc32_ref(pk, 32);
        }
    }
    if (g_cuda.d_dump != g_cuda.d_tab)
        (void)cuMemFree_v2(g_cuda.d_dump);
    g_cuda.d_dump = g_cuda.d_tab;

    scalar_offset(sk, w->lane[pl].base, w->lane[pl].win * (uint64_t)CUDA_GW + pj);
    x25519_ref(pk, sk, BASE_U);
    id = crc32_ref(pk, 32);
    memset(f, 0, sizeof *f);
    f->groups = 1;
    f->need_id = 1;
    f->group[0].mask = 0xffffffffu;
    f->group[0].value = id;
    f->group[0].source = SRC_ID;
    f->group[0].single = 1;
    f->group[0].count = 1;
    f->member_value[0] = id;
    w->raw = 1;
    ok = ok && cuda_run1(w, 1, 0, f) == 0;
    w->raw = 0;
    for (unsigned r = 0; ok && r < (c[0] < CUDA_CAP ? c[0] : CUDA_CAP); r++) {
        const uint32_t *rec = c + 4 + 12 * r;
        found |= rec[0] == pl && rec[1] == 0 && rec[2] == pj && rec[3] == id && memcmp(rec + 4, pk, 32) == 0;
    }
    w->slot_f[0] = w->slot_f[1] = NULL;             /* f is about to go */
    secure_zero(sk, sizeof sk);
    secure_zero(before, sizeof before);
    xfree_aligned(f);
    free(map);
    free(dump);
    return ok && wrong == 0 && found ? 0 : -1;
}

/* What can be checked without a GPU: the tables, the batched seeding, the kernel text. */
static void cuda_test_host(void)
{
    enum { N = 2048 };
    lane_t *lane = (lane_t *)calloc(N, sizeof *lane);
    uint32_t (*centre)[16] = (uint32_t (*)[16])calloc(N, 64);
    uint32_t *tab = (uint32_t *)malloc((CUDA_W + 2) * 96), want[16];
    char *ptx = cuda_ptx(0);
    const size_t len = strlen(ptx);
    unsigned bad = 0;

    if (lane == NULL || centre == NULL || tab == NULL)
        die("out of memory");
    bad += cuda_comb() != 0 || cuda_table(tab) != 0;
    if (bad == 0)
        cuda_seed_all(lane, N, centre);
    for (unsigned i = 0; i < N && bad == 0; i++) {
        uint8_t sk[32];
        mpoint c;

        bad += i > 0 && memcmp(lane[i].base, lane[i - 1].base, 32) == 0;
        if (i % 409 != 0)
            continue;
        scalar_offset(sk, lane[i].base, CUDA_W);
        bad += mont_scalarmult_base(&c, sk) != 0;
        mpoint_canonical(&c);
        cuda_words(want, &c.u);
        cuda_words(want + 8, &c.v);
        bad += memcmp(want, centre[i], 64) != 0;
        secure_zero(sk, sizeof sk);
    }
    for (unsigned i = 0; i <= CUDA_W && bad == 0; i += i < 4 ? 1 : 341) {
        mpoint pt;
        fe m;

        cuda_fe(&pt.u, tab + 24 * i);
        cuda_fe(&pt.v, tab + 24 * i + 8);
        cuda_fe(&m, tab + 24 * i + 16);
        fe_add(&m, &m, &pt.u);
        fe_add(&m, &m, &FE_A);
        bad += !mont_on_curve(&pt) || !fe_is_zero(&m);
    }
    cuda_words(want, &TAB[HALF_WINDOW].u);
    bad += memcmp(want, tab + 24 * HALF_WINDOW, 32) != 0;
    bad += len < 100000 || strstr(ptx, ".entry mtv_walk(") == NULL || strstr(ptx, ".entry mtv_dump(") == NULL ||
           ptx[len - 2] != '}';
    t_check(bad == 0, "cuda: tables, batched seeding and the kernel text are right (no GPU needed)");
    secure_zero(lane, N * sizeof *lane);
    free(lane);
    free(centre);
    free(tab);
    free(ptx);
}

static void cuda_self_test(int full)
{
    const out_mode_t saved_mode = g_run.mode;

    cuda_test_field();
    cuda_test_walk(full);
    g_run.mode = saved_mode;
    if (full)
        test_search(&BACKEND_CUDA);
}

static void cuda_usage(int to)
{
    static const char text[] =
        "\n"
        "This build can search on an NVIDIA GPU (driver only, no CUDA toolkit):\n"
        "  --backend cuda     the GPU alone.  auto starts on the CPU at once and hands\n"
        "                     over to the GPU when it is ready, a second or two later,\n"
        "                     if the GPU is clearly the faster\n"
        "  --threads N        with a GPU under auto: N CPU threads, and they keep\n"
        "                     searching beside the GPU (default: every CPU, until it is ready)\n"
        "  --cuda-ptx         print the GPU kernel (PTX text) and exit\n"
        "Without a usable GPU, or if it fails during a search, a line on stderr says\n"
        "why and the search runs on the CPU.\n";

    (void)os_write(to, text, sizeof text - 1);
}

#endif /* MTVANITY_CUDA_H */
