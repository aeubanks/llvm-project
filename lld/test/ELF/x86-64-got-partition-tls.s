# REQUIRES: x86
## A GOT entry in a GOT partition for a TLS symbol holds the offset of the
## symbol from the thread pointer, just like the primary GOT entry.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared --got-partition-threshold=4096 %t.o -o %t.so
# RUN: llvm-objdump -d --no-print-imm-hex %t.so > %t.txt
# RUN: llvm-readelf -S -r %t.so >> %t.txt
# RUN: FileCheck %s < %t.txt

# CHECK:      <_start>:
# CHECK-NEXT:   movq {{.*}}(%rip), %rax # 0x[[#%x, GOT:]]
# CHECK-NEXT:   movq %fs:(%rax), %rax

# CHECK:      .got.ltext.0 PROGBITS [[#%.16x, GOT]]
# CHECK:      [[#%.16x, GOT]] {{.*}} R_X86_64_TPOFF64 {{.*}} foo + 0

.section .ltext, "axl"
.globl _start
_start:
  movq foo@GOTTPOFF(%rip), %rax
  movq %fs:(%rax), %rax

.section .ltext.pad, "axl", @nobits
.space 0x4000

.section .tbss, "awT", @nobits
.globl foo
foo:
  .quad 0
