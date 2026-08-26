# REQUIRES: x86
## Test a range extension thunk to a preemptible symbol with a non-zero addend.
## A GLOB_DAT relocation cannot express the addend, so the GOT entry the thunk
## loads needs a symbolic relocation instead.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -d --no-print-imm-hex %t.so | FileCheck %s
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=REL %s

# CHECK-LABEL: <__X86_64Thunk_bar>:
# CHECK-NEXT:  1440: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x2468
# CHECK-LABEL: <__X86_64Thunk_foo+0x8>:
# CHECK-NEXT:  1448: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x2460
# CHECK-LABEL: <high>:
# CHECK-NEXT:  144e: e8 {{.*}} callq 0x1448 <__X86_64Thunk_foo+0x8>
# CHECK-NEXT:  1453: e8 {{.*}} callq 0x1440 <__X86_64Thunk_bar>

# REL:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# REL:      0000000000002460 {{.*}} R_X86_64_64 {{.*}} foo + 8
# REL-NEXT: 0000000000002468 {{.*}} R_X86_64_GLOB_DAT {{.*}} bar + 0

.section .ltext, "axl"
.globl high
.type high, @function
high:
  call foo@PLT+8
  call bar@PLT
  ret

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.section .text, "ax"
.globl _start
.type _start, @function
_start:
  call high
