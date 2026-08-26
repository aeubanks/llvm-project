# REQUIRES: x86
## Test range extension thunks for branches to a section symbol with a non-zero
## addend, which is how the assembler encodes a branch to a temporary label in
## another section.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-print-imm-hex %t | FileCheck %s
# RUN: llvm-readelf -x .got.ltext.1 -x .got %t | FileCheck --check-prefix=HEX %s

# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-objdump -d --no-print-imm-hex %t.pie | FileCheck --check-prefix=PIE-ASM %s
# RUN: llvm-readelf -r %t.pie | FileCheck --check-prefix=PIE %s

## The branches must target the first byte of the thunk, and the GOT entry the
## thunk loads must hold the address of the temporary label (.text+0x7 and
## .ltext+0x2), not the address of the section.

# CHECK-LABEL: <ltext_start>:
# CHECK-NEXT:  201270: 90 nop
# CHECK-NEXT:  201271: 90 nop
# CHECK-NEXT:  201272: c3 retq
# CHECK-LABEL: <__X86_64Thunk_.text+0x7>:
# CHECK-NEXT:  204280: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x203278
# CHECK-LABEL: <trampoline>:
# CHECK-NEXT:  204286: e9 {{.*}} jmp 0x204280 <__X86_64Thunk_.text+0x7>
# CHECK-LABEL: <__X86_64Thunk_.ltext+0x2>:
# CHECK-NEXT:  8020628c: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x802072a0
# CHECK-LABEL: <_start>:
# CHECK-NEXT:  80206294: e9 {{.*}} jmp 0x8020628c <__X86_64Thunk_.ltext+0x2>
# CHECK-NEXT:  80206299: 90 nop
# CHECK-NEXT:  8020629a: 90 nop
# CHECK-NEXT:  8020629b: c3 retq

# HEX:      Hex dump of section '.got.ltext.1':
# HEX-NEXT: 0x00203278 9b622080 00000000
# HEX:      Hex dump of section '.got':
# HEX-NEXT: 0x802072a0 72122000 00000000

# PIE-ASM-LABEL: <__X86_64Thunk_.text+0x7>:
# PIE-ASM-NEXT:  4330: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x3328
# PIE-ASM-LABEL: <trampoline>:
# PIE-ASM-NEXT:  4336: e9 {{.*}} jmp 0x4330 <__X86_64Thunk_.text+0x7>
# PIE-ASM-LABEL: <__X86_64Thunk_.ltext+0x2>:
# PIE-ASM-NEXT:  8000633c: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x80007420
# PIE-ASM-LABEL: <_start>:
# PIE-ASM-NEXT:  80006344: e9 {{.*}} jmp 0x8000633c <__X86_64Thunk_.ltext+0x2>

# PIE:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# PIE:      0000000000003328 {{.*}} R_X86_64_RELATIVE 8000634b
# PIE-NEXT: 0000000080007420 {{.*}} R_X86_64_RELATIVE 1322

.section .ltext, "axl"
.globl ltext_start
ltext_start:
  nop
  nop
4:
  ret

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.section .text, "ax"
.globl _start
.type _start, @function
_start:
  jmp 4b
.pushsection .ltext.unlikely, "axl"
trampoline:
  jmp 3f
.popsection
  nop
  nop
3:
  ret
