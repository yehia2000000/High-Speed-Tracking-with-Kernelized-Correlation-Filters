# riscv_add_baremetal.cmake
#
# Helpers to build bare-metal RISC-V-CV images for XS-GEM5, Linux static ELFs
# for qemu-riscv64 user-mode, and to register run targets / ctest entries.
#
#   riscv_add_baremetal_image(<source.cpp>)
#       Builds an executable named after the source's basename (NAME_WE), with
#       OUTPUT_NAME <name>.elf, C++20, bare-metal compile flags, and a POST_BUILD
#       objcopy to <name>.bin. Does not link libraries — callers must link
#       ${PROJECT_NAME}, reference, and riscv_baremetal_runtime (user libs MUST
#       be listed before riscv_baremetal_runtime).
#
#   riscv_baremetal_runtime (INTERFACE library)
#       Carries nexus-am am/klib archives (--start-group), -lgcc, and bare-metal
#       link options (-nostdlib, linker script, etc.). Depends on nexus_am_archives.
#
#   riscv_add_gem5_run(<target> [NCPU <n>])
#       Adds a `run_<target>` custom target and, unless the target property
#       RISCV_GEM5_NO_CTEST is TRUE, a ctest entry named <target> that runs the
#       image under gem5. The .bin path comes from the RISCV_GEM5_BIN property
#       set by riscv_add_baremetal_image. Pass/fail regexes come from target
#       properties RISCV_GEM5_PASS_REGEX / RISCV_GEM5_FAIL_REGEX (gtest-lite
#       defaults if unset). gem5 does not propagate the guest exit code.
#
#       NCPU sets the number of gem5 cores (passed as -n). If NCPU is omitted,
#       the target property RISCV_GEM5_NCPU is used, and if that is unset too,
#       the default is 1. Multi-hart images (_mpe_wakeup) need NCPU 2:
#       with one core, XS-GEM5's HartCtrl device has no slot for hart 1 and
#       panics on the _mpe_wakeup(1) write.
#
#       NCPU > 1 runs on GEM5_MULTICORE_BINARY with GEM5_MULTICORE_ARGS added
#       (defaults: XS-GEM5's Ruby CHI build next to GEM5_BINARY, and
#       --ruby --mem-type=DDR4_2400_8x8). XS-GEM5's classic cache hierarchy is
#       single-core only: with -n 2 it aborts on CoherentXBar's express-snoop
#       assert. Ruby CHI is the multi-core setup XS-GEM5's own CI runs.
#
#       GEM5_EXTRA_ARGS is a CMake list (e.g.
#       "--disable-difftest;--enable-riscv-vector") appended to every gem5 run,
#       single- and multi-core.
#
#       The run target goes through `bash -o pipefail`, so a gem5 crash
#       (panic / fatal / abort) fails the build instead of being hidden behind
#       tee's exit status.
#
#   riscv_add_linux_executable(<source.cpp>)
#       Builds ${name}_qemu as a static Linux ELF (glibc) from the source.
#       Defines RISCV_QEMU (hosted PGM/PPM file I/O in Image). Callers link
#       RISCV_CV_qemu and reference_qemu.
#
#   riscv_add_qemu_run(<target>)
#       Adds `run_<target>` and ctest <target> for a Linux ELF built by
#       riscv_add_linux_executable (e.g. unit_tests_qemu). QEMU propagates the
#       guest exit code, so ctest uses COMMAND only (no PASS/FAIL regex).
#
# Requires the cache variables set by the top-level CMakeLists.txt:
#   NEXUS_AM_HOME, AM_ARCH, RISCV_MARCH, RISCV_MABI, RISCV_TOOLCHAIN_PREFIX,
#   GEM5_BINARY, GEM5_CONFIG, GEM5_EXTRA_ARGS, QEMU_BINARY, QEMU_CPU,
#   QEMU_EXTRA_ARGS.
# Defines (if not already set) GEM5_MULTICORE_BINARY and GEM5_MULTICORE_ARGS.

# ---- gem5 for multi-core runs -------------------------------------------------
#
# Default: <gem5>/build/RISCV_CHI/gem5.opt, the sibling of GEM5_BINARY's
# build/RISCV. Build it in the gem5 tree with: scons build/RISCV_CHI/gem5.opt
get_filename_component(_gem5_variant_dir "${GEM5_BINARY}" DIRECTORY)
get_filename_component(_gem5_build_dir "${_gem5_variant_dir}" DIRECTORY)
set(GEM5_MULTICORE_BINARY "${_gem5_build_dir}/RISCV_CHI/gem5.opt" CACHE FILEPATH
    "gem5 binary for NCPU > 1 runs (XS-GEM5 Ruby CHI build)")
set(GEM5_MULTICORE_ARGS "--ruby;--mem-type=DDR4_2400_8x8" CACHE STRING
    "Extra gem5 config arguments for NCPU > 1 runs (CMake list)")

# ---- build nexus-am's am/klib archives --------------------------------------
#
# These archives are produced by nexus-am's own Makefiles, not CMake. Build
# them once per configure as a custom target; every image depends on it so the
# archives always exist and match AM_ARCH / RISCV_MARCH.
set(RISCV_AM_ARCHIVE  "${NEXUS_AM_HOME}/am/build/am-${AM_ARCH}.a")
set(RISCV_KLIB_ARCHIVE "${NEXUS_AM_HOME}/libs/klib/build/klib-${AM_ARCH}.a")

# AM_HOME is passed on the make line: nexus-am's Makefiles include
# $(AM_HOME)/Makefile.lib, so without it the build depends on the shell
# happening to export AM_HOME.
add_custom_command(
  OUTPUT  "${RISCV_AM_ARCHIVE}" "${RISCV_KLIB_ARCHIVE}"
  COMMAND ${CMAKE_MAKE_PROGRAM} -C "${NEXUS_AM_HOME}/am"
          AM_HOME=${NEXUS_AM_HOME}
          ARCH=${AM_ARCH} LINUX_GNU_TOOLCHAIN=1 MARCH=${RISCV_MARCH}
  COMMAND ${CMAKE_MAKE_PROGRAM} -C "${NEXUS_AM_HOME}/libs/klib"
          AM_HOME=${NEXUS_AM_HOME}
          ARCH=${AM_ARCH} LINUX_GNU_TOOLCHAIN=1 MARCH=${RISCV_MARCH}
  COMMENT "Building nexus-am am/klib archives for ${AM_ARCH}"
  VERBATIM)

add_custom_target(nexus_am_archives
  DEPENDS "${RISCV_AM_ARCHIVE}" "${RISCV_KLIB_ARCHIVE}")

# ---- riscv_baremetal_runtime --------------------------------------------------
#
# Link order matters: the linker resolves left to right, so user object files
# and libraries must precede the am/klib archives that satisfy their undefined
# symbols. Callers MUST list ${PROJECT_NAME} and reference before
# riscv_baremetal_runtime. The --start-group wraps the circular am<->klib refs.
add_library(riscv_baremetal_runtime INTERFACE)
target_link_libraries(riscv_baremetal_runtime INTERFACE
  "-Wl,--start-group"
  "${RISCV_AM_ARCHIVE}" "${RISCV_KLIB_ARCHIVE}"
  "-Wl,--end-group"
  "-lgcc")
target_link_options(riscv_baremetal_runtime INTERFACE
  "-nostdlib" "-nostartfiles"
  "-Wl,--build-id=none"
  "-T" "${RISCV_LINKER_SCRIPT}"
  "-L" "${NEXUS_AM_HOME}/am/src/nemu/ldscript"
  "-Wl,--gc-sections")
add_dependencies(riscv_baremetal_runtime nexus_am_archives)

# ---- riscv_add_baremetal_image ------------------------------------------------
function(riscv_add_baremetal_image source)
  if(NOT source)
    message(FATAL_ERROR "riscv_add_baremetal_image: no source given")
  endif()
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "riscv_add_baremetal_image: source not found: ${source}")
  endif()

  get_filename_component(name ${source} NAME_WE)

  # Compile the source with the toolchain's flags plus the nexus-am defines
  # and include paths. The C++ runtime (operator new/delete,
  # __cxa_pure_virtual, riscv_run_static_ctors) reaches every image through
  # the riscv_cv library archive.
  add_executable(${name}_gem5 ${source})

  set_target_properties(${name}_gem5 PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "${name}.elf"
    SUFFIX "")

  target_compile_options(${name}_gem5 PRIVATE
    ${RISCV_AM_DEFINE_FLAGS}
    ${RISCV_BAREMETAL_COMPILE_FLAGS})

  # The archives are built by a custom target, so make sure they exist first.
  add_dependencies(${name}_gem5 nexus_am_archives)

  # Emit the gem5-consumable binary, matching nexus-am's `image` rule.
  set(riscv_gem5_bin "${CMAKE_CURRENT_BINARY_DIR}/${name}.bin")
  set_target_properties(${name}_gem5 PROPERTIES RISCV_GEM5_BIN "${riscv_gem5_bin}")

  add_custom_command(TARGET ${name}_gem5 POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -S --set-section-flags .bss=alloc,contents
            -O binary "$<TARGET_FILE:${name}_gem5>" "${riscv_gem5_bin}"
    COMMENT "objcopy ${name}.elf -> ${name}.bin"
    VERBATIM)

endfunction()

# ---- riscv_add_gem5_run -------------------------------------------------------
function(riscv_add_gem5_run target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "NCPU" "")
  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "riscv_add_gem5_run: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT TARGET ${target})
    message(FATAL_ERROR "riscv_add_gem5_run: target not found: ${target}")
  endif()

  get_target_property(bin ${target} RISCV_GEM5_BIN)
  if(NOT bin)
    message(FATAL_ERROR
      "riscv_add_gem5_run: target '${target}' has no RISCV_GEM5_BIN property "
      "(use riscv_add_baremetal_image first)")
  endif()

  # get_target_property sets an unset property to "<VAR>-NOTFOUND" (the
  # variable's name, not the property's), so match the suffix only.
  get_target_property(pass_regex ${target} RISCV_GEM5_PASS_REGEX)
  if(pass_regex MATCHES "-NOTFOUND$")
    set(pass_regex "\\[ PASSED")
  endif()

  get_target_property(fail_regex ${target} RISCV_GEM5_FAIL_REGEX)
  if(fail_regex MATCHES "-NOTFOUND$")
    set(fail_regex "\\[ FAILED")
  endif()

  get_target_property(no_ctest ${target} RISCV_GEM5_NO_CTEST)

  # Core count: NCPU argument > RISCV_GEM5_NCPU property > 1.
  if(DEFINED arg_NCPU)
    set(ncpu "${arg_NCPU}")
  else()
    get_target_property(ncpu ${target} RISCV_GEM5_NCPU)
    if(ncpu MATCHES "-NOTFOUND$")
      set(ncpu 1)
    endif()
  endif()
  if(NOT ncpu MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
      "riscv_add_gem5_run: NCPU for '${target}' must be a positive integer, got '${ncpu}'")
  endif()

  # Multi-core runs need the Ruby CHI build (see the header comment).
  if(ncpu GREATER 1)
    set(gem5_bin  "${GEM5_MULTICORE_BINARY}")
    set(gem5_args ${GEM5_MULTICORE_ARGS})
    if(NOT EXISTS "${gem5_bin}")
      message(WARNING
        "riscv_add_gem5_run: '${target}' runs on ${ncpu} cores and needs "
        "${gem5_bin}, which does not exist yet. Build it in the gem5 tree "
        "with: scons build/RISCV_CHI/gem5.opt  (or set GEM5_MULTICORE_BINARY)")
    endif()
  else()
    set(gem5_bin  "${GEM5_BINARY}")
    set(gem5_args "")
  endif()
  # The shell string needs spaces; a raw CMake list would put ';' in it.
  list(JOIN gem5_args " " gem5_args_str)
  # Same for GEM5_EXTRA_ARGS: inside the bash -c string a ';' would end the
  # gem5 command and bash would run the next argument as its own command.
  list(JOIN GEM5_EXTRA_ARGS " " gem5_extra_args_str)

  set(m5out "${CMAKE_CURRENT_BINARY_DIR}/${target}-m5out")
  set(log   "${CMAKE_CURRENT_BINARY_DIR}/${target}.log")

  # gem5 writes raw UART output with no trailing-newline flush, so pipe
  # through tee to land results in a log file as well as the console. The
  # shell form is used so the pipe works; the bin path is attached to
  # --generic-rv-cpt with no embedded quotes. pipefail makes a gem5 crash
  # fail the target; without it the pipeline returns tee's status (always 0).
  add_custom_target(run_${target}
    COMMAND bash -o pipefail -c
      "${gem5_bin} --remote-gdb-port=0 -d ${m5out} ${GEM5_CONFIG} -n ${ncpu} ${gem5_args_str} --raw-cpt --generic-rv-cpt=${bin} ${gem5_extra_args_str} 2>&1 | tee ${log}"
    DEPENDS ${target}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Running ${target} on gem5 with ${ncpu} core(s) (output -> ${log})"
    USES_TERMINAL
    VERBATIM)

  if(NOT no_ctest)
    add_test(NAME ${target}
      COMMAND ${gem5_bin} --remote-gdb-port=0 -d ${m5out} ${GEM5_CONFIG}
              -n ${ncpu} ${gem5_args} --raw-cpt --generic-rv-cpt=${bin} ${GEM5_EXTRA_ARGS})
    set_tests_properties(${target} PROPERTIES
      PASS_REGULAR_EXPRESSION "${pass_regex}"
      FAIL_REGULAR_EXPRESSION "${fail_regex}")
  endif()
endfunction()

# ---- riscv_add_linux_executable -----------------------------------------------
function(riscv_add_linux_executable source)
  if(NOT source)
    message(FATAL_ERROR "riscv_add_linux_executable: no source given")
  endif()
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "riscv_add_linux_executable: source not found: ${source}")
  endif()

  get_filename_component(name ${source} NAME_WE)

  add_executable(${name}_qemu ${source})

  set_target_properties(${name}_qemu PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF)

  # -static already in toolchain flags; ensure hosted libc (not freestanding).
  target_compile_options(${name}_qemu PRIVATE
    -fno-freestanding -fbuiltin)

  target_compile_definitions(${name}_qemu PRIVATE RISCV_QEMU)
endfunction()

# ---- riscv_add_qemu_run -------------------------------------------------------
function(riscv_add_qemu_run target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "riscv_add_qemu_run: target not found: ${target}")
  endif()

  set(log "${CMAKE_CURRENT_BINARY_DIR}/${target}.log")

  set(qemu_cmd_line "${QEMU_BINARY} -cpu ${QEMU_CPU}")
  if(QEMU_EXTRA_ARGS)
    set(qemu_cmd_line "${qemu_cmd_line} ${QEMU_EXTRA_ARGS}")
  endif()

  add_custom_target(run_${target}
    COMMAND /bin/sh -c
      "${qemu_cmd_line} \"$1\" 2>&1 | tee ${log}" _ $<TARGET_FILE:${target}>
    DEPENDS ${target}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Running ${target} on qemu-riscv64 (output -> ${log})"
    USES_TERMINAL
    VERBATIM)

  set(qemu_test_cmd ${QEMU_BINARY} -cpu ${QEMU_CPU})
  if(QEMU_EXTRA_ARGS)
    list(APPEND qemu_test_cmd ${QEMU_EXTRA_ARGS})
  endif()
  list(APPEND qemu_test_cmd $<TARGET_FILE:${target}>)

  add_test(NAME ${target} COMMAND ${qemu_test_cmd})
endfunction()