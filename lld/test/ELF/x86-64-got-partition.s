# This test checks the basic functionality of x86_64 GOT partitioning.
# It verifies that:
# 1. When a relocation target (foo) is out of range (> 2GB) from the code (.text.01),
#    LLD creates a GOT partition (.got.ltext.0) and routes the relocation through it.
# 2. When the target is in range (from .text.02 to foo in .text.03), LLD relaxes
#    the GOTPCREL relocation to a direct PC-relative reference (leaq) and does
#    not use a partition.
# 3. The partition is correctly placed in the layout and mapped in the link map.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie %t/test.o -o %t.exe -Map=%t.map
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck %s --check-prefix=DISASM
# RUN: FileCheck %s --check-prefix=MAP < %t.map
# RUN: ld.lld -T %t/lds %t/test.o -o %t.nopie
# RUN: llvm-readelf -x .got.ltext.0 %t.nopie | FileCheck %s --check-prefix=HEX-NOPIE

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x10010
# DISASM-EMPTY:
# DISASM-NEXT: <_start_2>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax{{.*}}# 0x10010

# DISASM:      Disassembly of section .text.2:
# DISASM-EMPTY:
# DISASM-NEXT: <bar>:
# DISASM-NEXT: {{.*}}leaq{{.*}}(%rip), %rax{{.*}}<foo>

# MAP:         .ltext
# MAP:         {{.*}}test.o:(.text.01)
# MAP:         {{.*}} 8 {{.*}} <internal>:(.got.ltext.0)

# HEX-NOPIE: Hex dump of section '.got.ltext.0':
# HEX-NOPIE-NEXT: 0x00010010 20000180 00000000

#--- lds
SECTIONS {
  .ltext 0x10000 : { *(.text.01) }
  .text.2 0x80010000 : { *(.text.02) }
  .text.3 0x80010020 : { *(.text.03) }
}

#--- test.s
.section .text.01, "axl"
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax
.globl _start_2
_start_2:
  movq foo@GOTPCREL(%rip), %rax

.section .text.02, "ax"
.globl bar
bar:
  movq foo@GOTPCREL(%rip), %rax

.section .text.03, "ax"
.globl foo
foo:
  .quad 0
