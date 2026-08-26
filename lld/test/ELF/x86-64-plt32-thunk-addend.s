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
# CHECK-NEXT:  203280: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x202278
# CHECK-LABEL: <trampoline>:
# CHECK-NEXT:  203286: e9 {{.*}} jmp 0x203280 <__X86_64Thunk_.text+0x7>
# CHECK-LABEL: <__X86_64Thunk_.ltext+0x2>:
# CHECK-NEXT:  8020428c: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x802052a0
# CHECK-LABEL: <_start>:
# CHECK-NEXT:  80204294: e9 {{.*}} jmp 0x8020428c <__X86_64Thunk_.ltext+0x2>
# CHECK-NEXT:  80204299: 90 nop
# CHECK-NEXT:  8020429a: 90 nop
# CHECK-NEXT:  8020429b: c3 retq

# HEX:      Hex dump of section '.got.ltext.1':
# HEX-NEXT: 0x00202278 9b422080 00000000
# HEX:      Hex dump of section '.got':
# HEX-NEXT: 0x802052a0 72122000 00000000

# PIE-ASM-LABEL: <__X86_64Thunk_.text+0x7>:
# PIE-ASM-NEXT:  3330: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x2328
# PIE-ASM-LABEL: <trampoline>:
# PIE-ASM-NEXT:  3336: e9 {{.*}} jmp 0x3330 <__X86_64Thunk_.text+0x7>
# PIE-ASM-LABEL: <__X86_64Thunk_.ltext+0x2>:
# PIE-ASM-NEXT:  8000433c: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x80005420
# PIE-ASM-LABEL: <_start>:
# PIE-ASM-NEXT:  80004344: e9 {{.*}} jmp 0x8000433c <__X86_64Thunk_.ltext+0x2>

# PIE:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# PIE:      0000000000002328 {{.*}} R_X86_64_RELATIVE 8000434b
# PIE-NEXT: 0000000080005420 {{.*}} R_X86_64_RELATIVE 1322

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
