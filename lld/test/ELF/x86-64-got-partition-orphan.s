# This test checks that when .ltext is placed as an orphan section (not explicitly
# specified in a SECTIONS command) and is far away (> 2GB) from the primary GOT,
# LLD creates a GOT partition (.got.ltext.0) and routes the relocation through it.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie %t/test.o -o %t.exe -Map=%t.map
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck %s --check-prefix=DISASM
# RUN: FileCheck %s --check-prefix=MAP < %t.map

# DISASM:      Disassembly of section .ltext:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT: {{.*}}movq{{.*}}(%rip), %rax

# MAP:         .ltext
# MAP:         {{.*}} 8 {{.*}} <internal>:(.got.ltext.0)

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text*) }
  .got 0x90000000 : { *(.got*) }
  .data 0x90001000 : { *(.data*) }
}

#--- test.s
.section .ltext, "axl", @progbits
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
