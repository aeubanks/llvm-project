# REQUIRES: x86
## With a SECTIONS command, check that a GOT partition does not share a page
## with the text before it.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie %t/test.o -o %t.exe
# RUN: llvm-readelf -lS %t.exe | FileCheck %s

# CHECK:      .ltext       PROGBITS 0000000000010000 002000 000007 00 AXl
# CHECK-NEXT: .got.ltext.0 PROGBITS 0000000000011008 002008 000008 00 WAl

# CHECK:      LOAD 0x002000 0x0000000000010000 0x0000000000010000 0x000007 0x000007 R E
# CHECK-NEXT: LOAD 0x002008 0x0000000000011008 0x0000000000011008 0x000008 0x000008 RW

#--- lds
SECTIONS {
  .ltext 0x10000 : { *(.text.01) }
  .far 0x90010000 : { *(.far) }
}

#--- test.s
.section .text.01, "axl"
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .far, "aw", @progbits
.globl foo
foo:
  .quad 0
