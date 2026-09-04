# This test checks the interaction between GOT partitioning and linker script
# section alignment (e.g., ALIGN(2MB) for huge pages).
# It verifies that:
# 1. When sections are aligned to large boundaries, the distance between them
#    might exceed the threshold, triggering partitioning even if the individual
#    sections are small.
# 2. Relocations from the far section (.text.01 at 0x1000) are routed through
#    a partition (.got.ltext.0).
# 3. Relocations from near sections (.text.02 and .text.03 at 0x200000+) are
#    relaxed to direct PC-relative references because they are close to the target.
# 4. The linker script alignment constraints (e.g., ALIGN(0x200000)) are respected.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie --got-partition-threshold=1048576 %t/test.o -o %t.exe -Map=%t.map
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck %s --check-prefix=DISASM
# RUN: FileCheck %s --check-prefix=MAP < %t.map

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT:   {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x1008

# DISASM:      Disassembly of section .text.2:
# DISASM-EMPTY:
# DISASM-NEXT: <_start_2>:
# DISASM-NEXT:   {{.*}}leaq{{.*}}(%rip), %rax{{.*}}# 0x2000a8 <foo>

# DISASM:      Disassembly of section .text.3:
# DISASM-EMPTY:
# DISASM-NEXT: <_start_3>:
# DISASM-NEXT:   {{.*}}leaq{{.*}}(%rip), %rax{{.*}}# 0x2000a8 <foo>

# MAP:         1000 1000 7 1 .ltext
# MAP:         1008 1008 8 8 .got.ltext.0
# MAP:         200000 200000 7 2097152 .text.2
# MAP:         200007 200007 7 1 .text.3
# MAP:         2000a0 2000a0 8 8 .got

#--- lds
SECTIONS {
  .ltext 0x1000 : { *(.text.01) }
  .text.2 : ALIGN(0x200000) { *(.text.02) }
  .text.3 : { *(.text.03) }
}

#--- test.s
.section .text.01, "axl", @progbits
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .text.02, "ax", @progbits
.globl _start_2
_start_2:
  movq foo@GOTPCREL(%rip), %rax

.section .text.03, "ax", @progbits
.globl _start_3
_start_3:
  movq foo@GOTPCREL(%rip), %rax

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
