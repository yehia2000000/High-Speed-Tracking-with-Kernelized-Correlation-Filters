/*
 * gem5 m5ops for the bare-metal build: per-region statistics in stats.txt.
 *
 * gem5 decodes opcode 0x7b (custom-3) as an m5op; bits 31:25 select the operation, a0 and a1
 * are its arguments (delay, period; 0, 0 = now, once). These are the same instructions as
 * m5_reset_stats / m5_dump_stats in gem5's util/m5 library, written inline so the bare-metal
 * build needs no extra library.
 *
 *   m5_reset_stats()  zero every statistic
 *   m5_dump_stats()   append one "Begin Simulation Statistics" section to stats.txt
 *
 * An m5op executes only at the head of the ROB, so the pipeline drains around it: keep it
 * outside any Timer whose cycles you report.
 *
 * qemu and real hardware do not know the instruction (illegal instruction). Without
 * RISCV_BAREMETAL, or with -DGEM5_M5OPS=0, both functions are empty.
 */
#pragma once

#ifndef GEM5_M5OPS
#ifdef RISCV_BAREMETAL
#define GEM5_M5OPS 1
#else
#define GEM5_M5OPS 0
#endif
#endif

#if GEM5_M5OPS
static inline void m5_reset_stats()
{
    asm volatile("li a0, 0\n\t"
                 "li a1, 0\n\t"
                 ".word 0x8000007b"     /* m5op 0x40: reset stats */
                 ::: "a0", "a1", "memory");
}

static inline void m5_dump_stats()
{
    asm volatile("li a0, 0\n\t"
                 "li a1, 0\n\t"
                 ".word 0x8200007b"     /* m5op 0x41: dump stats */
                 ::: "a0", "a1", "memory");
}
#else
static inline void m5_reset_stats() {}
static inline void m5_dump_stats() {}
#endif
