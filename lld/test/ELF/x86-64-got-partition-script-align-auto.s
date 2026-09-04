# This test checks that when LLD automatically splits large sections,
# it correctly propagates the alignment constraints of the parent section
# to all the newly created split sections.
# It verifies that:
# 1. The parent section (.ltext) has a large alignment constraint (ALIGN(64KB)).
# 2. When .ltext is split into .ltext, .ltext.1, and .ltext.2 due to size,
#    each split section inherits and respects the 64KB alignment.
# 3. This is verified by checking that the start addresses of the split
#    sections in the map file are multiples of 64KB.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie --got-partition-threshold=262144 %t/test.o -o %t.exe -Map=%t.map
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck %s --check-prefix=DISASM
# RUN: FileCheck %s --check-prefix=MAP < %t.map

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT:   {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x40d48

# DISASM:      Disassembly of section .ltext.1:
# DISASM-EMPTY:
# DISASM-NEXT: <_start_2>:
# DISASM-NEXT:   {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x80d48

# DISASM:      Disassembly of section .ltext.2:
# DISASM-EMPTY:
# DISASM-NEXT: <_start_3>:
# DISASM-NEXT:   {{.*}}leaq{{.*}}(%rip), %rax{{.*}}# 0xc0de0 <foo>

# MAP:         10000 {{.*}} .ltext
# MAP:         40d48 {{.*}} .got.ltext.0
# MAP:         50000 {{.*}} .ltext.1
# MAP:         80d48 {{.*}} .got.ltext.1
# MAP:         90000 {{.*}} .ltext.2

#--- lds
SECTIONS {
  .ltext 0x10000 : ALIGN(0x10000) { *(.text .text.*) }
}

#--- test.s
.section .text.01, "axl", @progbits
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .text.02, "axl", @progbits
.globl _start_2
_start_2:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .text.03, "axl", @progbits
.globl _start_3
_start_3:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
