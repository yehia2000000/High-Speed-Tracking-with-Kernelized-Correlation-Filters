// Dual-core histogram benchmark
//   build: nexus-am bare-metal (AM_ARCH=riscv64-xs is fine, see "Harts" below)
//   run:   XS-GEM5 Ruby CHI build: build/RISCV_CHI/gem5.opt kmhv3.py -n 2 --ruby
//          --raw-cpt (the classic cache hierarchy is single-core only).
//          The CMake helpers pick that up from BENCHMARK_NCPU_image_hist_bench.
//
// Three measurements, all on the same image:
//   1) scalar reference  - hart 0 alone
//   2) RVV, 1 core       - hart 0 alone
//   3) RVV, 2 cores      - each hart does half the image, hart 0 reduces
//
// Harts: both harts boot through nexus-am's start.S, which already gives each
// hart its own stack (sp = _stack_pointer + mhartid * 32KB). Hart 1 is held in
// reset until hart 0 releases it with _mpe_wakeup(1). nexus-am's _barrier() is
// NOT used: it keeps its per-hart sense in a __thread variable, and only the
// riscv64-xs-dual runtime sets up tp/TLS, so with riscv64-xs that variable sits
// at address 0. barrier() below keeps the per-hart sense in a plain array.

#include "benchmark.hpp"     // -> riscv_port.hpp -> klib.h (memset, printf, assert)
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include <stdint.h>
#include <stddef.h>
// No <string.h>: in bare-metal builds it resolves to the Linux glibc header,
// whose C++ overloads of strchr/strrchr/strstr clash with klib.h.
#include "image1.hpp"

#ifndef RISCV_BAREMETAL
#error "Dual-core version needs the nexus-am bare-metal build"
#endif
#include <am.h>       // _cpu, _mpe_init, _halt
#include <xsextra.h>  // _mpe_wakeup


inline uint8_t* make_constant(uint8_t* out, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i) out[i] = value;
    return out;
}

inline uint8_t* make_uniform(uint8_t* out, size_t n,
                             uint8_t a = 0, uint8_t b = 255, uint32_t seed = 12345) {
    uint32_t x = seed;
    uint32_t range = uint32_t(b) - a + 1;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        out[i] = uint8_t(a + (uint64_t(x) * range >> 32));
    }
    return out;
}

inline uint8_t* make_skewed(uint8_t* out, size_t n,
                            uint8_t a = 0, uint8_t b = 255, uint32_t seed = 12345,
                            uint32_t hot_pct = 90, uint32_t hot_bins = 8) {
    assert(a <= b);
    uint32_t range = uint32_t(b) - a + 1;

    // Clamp parameters to something meaningful.
    if (hot_pct  > 100)   hot_pct  = 100;
    if (hot_bins == 0)    hot_bins = 1;
    if (hot_bins > range) hot_bins = range;
    uint32_t cold_bins = range - hot_bins;          // 0 when the hot set covers everything

    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        // Draw 1: biased coin. Map x to [0, 100) using the HIGH bits, compare to hot_pct.
        x = x * 1664525u + 1013904223u;
        bool hot = (uint64_t(x) * 100 >> 32) < hot_pct || cold_bins == 0;

        // Draw 2: pick a value inside the chosen region.
        x = x * 1664525u + 1013904223u;
        uint32_t idx = hot ? uint32_t(uint64_t(x) * hot_bins  >> 32)
                           : hot_bins + uint32_t(uint64_t(x) * cold_bins >> 32);

        out[i] = uint8_t(a + idx);
    }
    return out;
}


// ---------------------------------------------------------------- config
static constexpr int    NCPU   = 2;          // nexus-am MAX_CPU is 2 on riscv
static constexpr size_t N      = 640 * 960;
static constexpr int    WARMUP = 1;          // iterations run but not counted
static constexpr int    ITERS  = 3;          // iterations that are averaged

// ---------------------------------------------------------------- shared state
// Globals live in .bss, so both harts see the same memory.
// Stack variables are private: each hart has its own stack (see "Harts" above).
alignas(64) static uint32_t hist_ref[256];           // hart 0 only
alignas(64) static uint32_t hist_vec1[256];          // hart 0 only (RVV, 1 core)
alignas(64) static uint32_t hist_part[NCPU][256];    // one 1KB row per hart -> no shared cache line
alignas(64) static uint32_t hist_dual[256];          // hart 0 reduces the rows into this

struct alignas(64) HartStats {                       // one cache line per hart -> no false sharing
    uint64_t cycles;
    uint64_t instrs;
};
static HartStats part_stats[NCPU];

// ---------------------------------------------------------------- barrier
// Sense-reversing barrier, same algorithm as nexus-am's _barrier() but with
// the per-hart sense in a global array indexed by hart id instead of TLS.
struct alignas(64) HartSense {                       // one cache line per hart
    int64_t value;
};
static HartSense      bar_local_sense[NCPU];
static volatile int64_t bar_count = 0;
static volatile int64_t bar_sense = 0;

static void barrier(int id) {
    asm volatile("fence" ::: "memory");              // publish this hart's earlier stores

    const int64_t s = !bar_local_sense[id].value;
    bar_local_sense[id].value = s;

    if (__atomic_fetch_add(&bar_count, 1, __ATOMIC_ACQ_REL) == NCPU - 1) {
        bar_count = 0;                               // last one in resets before releasing
        __atomic_store_n(&bar_sense, s, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&bar_sense, __ATOMIC_ACQUIRE) != s)
            ;
    }

    asm volatile("fence" ::: "memory");              // see the other hart's stores after this
}


static bool hist_equal(const uint32_t* a, const uint32_t* b) {
    uint64_t total = 0;
    for (int k = 0; k < 256; ++k) {
        if (a[k] != b[k]) return false;
        total += a[k];
    }
    return total == N;                               // every pixel counted exactly once
}

// Hart 1 parks here forever. It must never return from main, because
// _halt() (called on return) ends the whole simulation for both harts.
// The asm keeps the compiler from deleting the empty infinite loop.
[[noreturn]] static void park() {
    while (1) asm volatile("" ::: "memory");
}


static void worker() {
    const int id = _cpu();                           // csrr mhartid
    if (id == 0) _mpe_wakeup(1);                     // HartCtrl: write 0 releases hart 1 (held in reset at boot)

    // This hart's half of the image
    const size_t   chunk = N / NCPU;
    const size_t   begin = size_t(id) * chunk;
    const size_t   len   = (id == NCPU - 1) ? N - begin : chunk;
    const uint8_t* my_in = image1_data + begin;

    // Accumulators (only hart 0's copies are used)
    uint64_t ref_cycles  = 0, ref_instrs  = 0;
    uint64_t vec1_cycles = 0, vec1_instrs = 0;
    uint64_t dual_cycles = 0, dual_instrs = 0;
    uint64_t part_cycles_sum[NCPU] = {};
    bool all_correct = true;

    for (int it = 0; it < WARMUP + ITERS; ++it) {
        const bool counted = (it >= WARMUP);

        // ===== Phase A: single core. Hart 0 runs, hart 1 waits at barrier (1).
        if (id == 0) {
            memset(hist_ref,  0, sizeof hist_ref);
            memset(hist_vec1, 0, sizeof hist_vec1);
            Timer t_ref, t_vec;

            t_ref.Start();
            ref::hist256(image1_data, N, hist_ref);
            t_ref.Stop();

            t_vec.Start();
            vec::image_hist_rvv(image1_data, N, hist_vec1);
            t_vec.Stop();

            if (counted) {
                ref_cycles  += t_ref.ElapsedCycles();
                ref_instrs  += t_ref.ElapsedInstructions();
                vec1_cycles += t_vec.ElapsedCycles();
                vec1_instrs += t_vec.ElapsedInstructions();
            }
            all_correct &= hist_equal(hist_ref, hist_vec1);
        }
        barrier(id);                                 // (1) phase A done

        // ===== Phase B: two cores, each on its own half.
        memset(hist_part[id], 0, sizeof hist_part[id]);   // prep, not timed
        barrier(id);                                 // (2) common start line

        Timer t_wall, t_part;
        if (id == 0) t_wall.Start();                 // wall time measured on hart 0
        t_part.Start();
        vec::image_hist_rvv(my_in, len, hist_part[id]);
        t_part.Stop();
        part_stats[id].cycles = t_part.ElapsedCycles();
        part_stats[id].instrs = t_part.ElapsedInstructions();

        barrier(id);                                 // (3) both halves done; its fences make
                                                     //     hart 1's stores visible to hart 0
        if (id == 0) {
            for (int k = 0; k < 256; ++k) {          // reduce
                uint32_t s = 0;
                for (int c = 0; c < NCPU; ++c) s += hist_part[c][k];
                hist_dual[k] = s;
            }
            t_wall.Stop();

            if (counted) {
                dual_cycles += t_wall.ElapsedCycles();
                for (int c = 0; c < NCPU; ++c) {
                    dual_instrs        += part_stats[c].instrs;  // total work of both harts
                    part_cycles_sum[c] += part_stats[c].cycles;
                }
            }
            all_correct &= hist_equal(hist_ref, hist_dual);
        }
        // Hart 1 loops straight back to barrier (1) and waits there while
        // hart 0 reduces and runs the next phase A. That barrier is what stops
        // hart 1 from clearing hist_part[1] while hart 0 is still reading it.
    }

    if (id != 0) park();

    // ----- only hart 0 reaches here
    PrintTime("Histogram RVV 1-core vs scalar", N,
              vec1_cycles / ITERS, vec1_instrs / ITERS,
              ref_cycles  / ITERS, ref_instrs  / ITERS);

    PrintTime("Histogram RVV 2-core vs RVV 1-core", N,
              dual_cycles / ITERS, dual_instrs / ITERS,
              vec1_cycles / ITERS, vec1_instrs / ITERS);

    for (int c = 0; c < NCPU; ++c)
        printf("  hart %d compute: %llu cycles\n",
               c, (unsigned long long)(part_cycles_sum[c] / ITERS));

    printf(all_correct ? "Output is correct.\n" : "Output is wrong!\n");
    _halt(all_correct ? 0 : 1);                      // ends the simulation for both harts
}


int main()
{
    // Runs on EACH hart: hart 1 boots through start.S -> _trm_init -> main too.
    riscv_enable_vector();   // mstatus.VS is a per-hart CSR
    _mpe_init(worker);       // init_tls() (a no-op on riscv64-xs), then worker()
    return 0;                // never reached
}