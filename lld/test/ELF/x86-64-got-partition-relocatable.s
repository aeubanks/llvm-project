# REQUIRES: x86
## Relocatable output is not partitioned.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld -r --got-partition-threshold=0x80 %t/a.o %t/b.o -o %t.o
# RUN: llvm-readelf -S %t.o | FileCheck %s

# CHECK:     .ltext
# CHECK-NOT: .ltext.1
# CHECK-NOT: .got.ltext

#--- a.s
.section .ltext, "axl"
.globl a
a:
  .zero 200

#--- b.s
.section .ltext, "axl"
.globl b
b:
  .zero 200
