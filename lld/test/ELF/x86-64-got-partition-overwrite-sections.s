# This test checks that GOT partitioning also applies to output sections
# described by an OVERWRITE_SECTIONS command.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld --fatal-warnings -T %t/lds -pie --got-partition-threshold=0x4000 \
# RUN:   -z max-page-size=4096 %t/test.o -o %t.exe
# RUN: llvm-readelf -S %t.exe | FileCheck %s --check-prefix=SEC
# RUN: llvm-objdump -d %t.exe | FileCheck %s --check-prefix=DISASM

# SEC:      .ltext       PROGBITS
# SEC-NEXT: .got.ltext.0 PROGBITS
# SEC-NEXT: .ltext.1     PROGBITS

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM:      Disassembly of section .ltext.1:
# DISASM-EMPTY:
# DISASM-NEXT: <_start_2>:

#--- lds
SECTIONS {
  .ltext : { *(.ltext.01) }
}
OVERWRITE_SECTIONS {
  .ltext : { *(.ltext.*) }
}

#--- test.s
.section .ltext.01, "axl", @progbits
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax
  .space 7000

.section .ltext.02, "axl", @progbits
.globl _start_2
_start_2:
  movq foo@GOTPCREL(%rip), %rax
  .space 7000

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
