# REQUIRES: x86
## A GOT entry in a GOT partition for a non-preemptible ifunc holds the address
## computed by the resolver at startup, just like the .igot.plt entry, so it
## needs its own IRELATIVE relocation.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-print-imm-hex %t > %t.txt
# RUN: llvm-readelf -S -r -x .got.ltext.0 %t >> %t.txt
# RUN: FileCheck %s < %t.txt

# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-objdump -d --no-print-imm-hex %t.pie > %t.pie.txt
# RUN: llvm-readelf -S -r -x .got.ltext.0 %t.pie >> %t.pie.txt
# RUN: FileCheck %s < %t.pie.txt

# CHECK:      <__X86_64Thunk_bar>:
# CHECK-NEXT:   jmpq *{{.*}}(%rip) # 0x[[#%x, GOT_BAR:]]
# CHECK:      <_start>:
# CHECK-NEXT:   movq {{.*}}(%rip), %rax # 0x[[#%x, GOT:]]
# CHECK-NEXT:   callq 0x{{.*}} <__X86_64Thunk_bar>
# CHECK-NEXT:   jmpq *%rax
# CHECK:      [[#%.16x, RESOLVER:]] <resolver>:

## The GOT partition entries loaded by _start and __X86_64Thunk_bar. Primary .got is empty (size 0).
# CHECK:      .got.ltext.0 PROGBITS [[#%.16x, GOT]] {{.*}} 000010
# CHECK:      .got PROGBITS {{.*}} 000000

## IRELATIVE relocations for the GOT partition entries and .igot.plt with the resolver as the addend.
# CHECK:      Relocation section '.rela.dyn' {{.*}} contains 3 entries:
# CHECK:      {{.*}} R_X86_64_IRELATIVE [[#%x, RESOLVER]]
# CHECK-NEXT: [[#%.16x, GOT]] {{.*}} R_X86_64_IRELATIVE [[#%x, RESOLVER]]
# CHECK-NEXT: [[#%.16x, GOT_BAR]] {{.*}} R_X86_64_IRELATIVE [[#%x, RESOLVER]]

## The entries must not statically hold the address of the resolver.
# CHECK:      Hex dump of section '.got.ltext.0':
# CHECK-NEXT: 0x[[#%.8x, GOT]] 00000000 00000000 00000000 00000000

.section .ltext, "axl"
.globl _start
.type _start, @function
_start:
  movq foo@GOTPCREL(%rip), %rax
  call bar
  jmp *%rax

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.text
.globl resolver
.type resolver, @function
resolver:
  ret
.globl foo
.type foo, @gnu_indirect_function
.set foo, resolver
.globl bar
.type bar, @gnu_indirect_function
.set bar, resolver
