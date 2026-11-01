/* Per-kernel counters for the scalar tracker: cycles and retired instructions from the
   RISC-V CSRs (riscv_timer.hpp), summed over every call of each region.
   Built in when KCF_PROFILE is defined (CMake option KCF_PROFILE, ON by default); otherwise
   PROF() is empty and nothing here is compiled.

   gem5:      the cycle / instret counters of the simulated core (the real numbers).
   qemu-user: qemu returns host ticks for both, so only the proportions mean anything.

   The table is a plain zero-initialized array: no static constructor and no guard
   variable, because nexus-am does not run C++ static initialization. */
#pragma once
#ifdef KCF_PROFILE
#include <cstdint>
#include "riscv_timer.hpp"

namespace kprof {

enum Region { K1, K2, K3, K4, LAYOUT, K5_FFT, K5_DETECT, K5_TRAIN, K6, N_REGIONS };

inline const char *name(int r) {
    static const char *const n[] = {"K1_subwindow", "K2_gradient", "K3_histogram", "K4_norm_pca_hann",
                                    "layout_to_planar", "K5_spectra_31fft", "K5_detect", "K5_train",
                                    "K6_peak"};
    return n[r];
}

struct Slot { uint64_t cycles, instret, calls; };   /* POD: lives in .bss, starts at 0 */
inline Slot g_table[N_REGIONS];

inline Slot *table() { return g_table; }
inline void reset() {
    for (int r = 0; r < N_REGIONS; r++) g_table[r] = Slot{0, 0, 0};
}

/* RAII: the constructor samples the counters, the destructor (end of the { } block) adds the difference. */
struct Scope {
    int r;
    CycleSample s0;
    explicit Scope(int r_) : r(r_), s0(Sample()) {}
    ~Scope() {
        CycleSample d = Delta(Sample(), s0);
        g_table[r].cycles  += d.cycles;
        g_table[r].instret += d.instructions;
        g_table[r].calls++;
    }
};

}  /* namespace kprof */

#define PROF(r) kprof::Scope kprof_scope_##r(kprof::r)
#else
#define PROF(r)
#endif
