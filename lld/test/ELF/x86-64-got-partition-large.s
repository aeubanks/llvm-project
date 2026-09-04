# This test checks that LLD correctly splits large executable sections
# (or groups of sections) and inserts multiple GOT partitions when the
# accumulated size exceeds the threshold.
# It verifies that:
# 1. Sections (.ltext.01 to .ltext.04) are grouped and split when their
#    accumulated size exceeds the threshold (16KB).
# 2. Multiple GOT partitions (.got.ltext.0, .got.ltext.1, etc.) are created
#    and inserted between the splits.
# 3. Relocations in each split section are routed to the closest partition.
# 4. Multiple PT_GNU_RELRO segments are created (one for each partition + primary GOT).

# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie --got-partition-threshold=16384 -z max-page-size=4096 %t.o -o %t.exe -Map=%t.map
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck %s --check-prefix=DISASM
# RUN: FileCheck %s --check-prefix=MAP < %t.map
# RUN: llvm-readelf -l %t.exe | FileCheck %s --check-prefix=PHDR

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x4000
# DISASM:      <_start_2>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x7b68
# DISASM:      <_start_3>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0xb6d0
# DISASM:      <_start_4>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0xb6d0

# MAP:         .ltext
# MAP-NOT:     <internal>:(.got.ltext.0)
# MAP:         {{.*}}:(.ltext.01)
# MAP:         .got.ltext.0
# MAP:         {{.*}} 8 {{.*}} <internal>:(.got.ltext.0)
# MAP:         .ltext.1
# MAP:         {{.*}}:(.ltext.02)
# MAP:         .got.ltext.1
# MAP:         {{.*}} 8 {{.*}} <internal>:(.got.ltext.1)
# MAP:         .ltext.2
# MAP:         {{.*}}:(.ltext.03)
# MAP:         .got.ltext.2
# MAP:         {{.*}} 8 {{.*}} <internal>:(.got.ltext.2)
# MAP:         .ltext.3
# MAP:         {{.*}}:(.ltext.04)

# PHDR:      GNU_RELRO      {{.*}} 0x0000000000004000 0x0000000000004000 0x000008 0x000008 R   0x1
# PHDR-NEXT: GNU_RELRO      {{.*}} 0x0000000000007b68 0x0000000000007b68 0x000008 0x000008 R   0x1
# PHDR-NEXT: GNU_RELRO      {{.*}} 0x000000000000b6d0 0x000000000000b6d0 0x000008 0x000008 R   0x1
# PHDR-NEXT: GNU_RELRO      {{.*}} 0x0000000000012500 0x0000000000012500 0x000098 0x000b00 R   0x1

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

.section .ltext.03, "axl", @progbits
.globl _start_3
_start_3:
  movq foo@GOTPCREL(%rip), %rax
  .space 7000

.section .ltext.04, "axl", @progbits
.globl _start_4
_start_4:
  movq foo@GOTPCREL(%rip), %rax
  .space 20000

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
