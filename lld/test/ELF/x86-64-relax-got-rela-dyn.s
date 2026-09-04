# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie %t/test.o -o %t.exe
# RUN: llvm-readobj -r %t.exe | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.dyn {
# CHECK-NEXT:     0x2000 R_X86_64_RELATIVE - 0x80002000
# CHECK-NEXT:   }
# CHECK-NEXT: ]

#--- lds
SECTIONS {
  .text 0x1000 : { *(.text) }
  .got 0x2000 : { *(.got) }
  .far 0x80002000 : { *(.far) }
}

#--- test.s
.text
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .far, "a", @progbits
.globl foo
foo:
  .quad 0
