# REQUIRES: x86
## An ifunc that also has an IPLT entry needs an IRELATIVE for its GOT partition
## entry.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-print-imm-hex %t > %t.txt
# RUN: llvm-readelf -r -x .got.ltext.0 %t >> %t.txt
# RUN: FileCheck %s < %t.txt

# CHECK:      <_start>:
# CHECK-NEXT:   movq {{.*}}(%rip), %rax # 0x[[#%x, GOT:]]
# CHECK-NEXT:   jmpq *%rax
# CHECK:      [[#%.16x, RESOLVER:]] <resolver>:

# CHECK:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# CHECK:      {{.*}} R_X86_64_IRELATIVE [[#%x, RESOLVER]]
# CHECK-NEXT: [[#%.16x, GOT]] {{.*}} R_X86_64_IRELATIVE [[#%x, RESOLVER]]

# CHECK:      Hex dump of section '.got.ltext.0':
# CHECK-NEXT: 0x[[#%.8x, GOT]] 00000000 00000000

.section .ltext, "axl"
.globl _start
.type _start, @function
_start:
  movq foo@GOTPCREL(%rip), %rax
  jmp *%rax

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.text
bar:
  call foo

.globl resolver
.type resolver, @function
resolver:
  ret
.globl foo
.type foo, @gnu_indirect_function
.set foo, resolver
