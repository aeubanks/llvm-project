# This test checks that when LLD automatically splits large sections,
# it correctly propagates the memory region assignments (VMA and LMA)
# of the parent section to all the newly created split sections.
# It verifies that:
# 1. The parent section (.ltext) is assigned to a specific memory region (> ALL)
#    and has a separate load memory region (AT > ROM).
# 2. When .ltext is split, the split sections (.ltext.1, etc.) inherit these
#    assignments.
# 3. This is verified by checking the VirtualAddress (VMA) and PhysicalAddress
#    (LMA) of the LOAD segments in the program headers, ensuring the LMA offset
#    is correctly calculated and propagated.

# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -T %t/lds -pie --got-partition-threshold=262144 %t/test.o -o %t.exe -Map=%t.map
# RUN: FileCheck %s --check-prefix=MAP < %t.map
# RUN: llvm-readobj -l %t.exe | FileCheck %s --check-prefix=PHDR

# MAP:         10000 {{.*}} 30d47 65536 .ltext
# MAP:         40d48 {{.*}} 8 8 .got.ltext.0
# MAP:         50000 {{.*}} 30d47 65536 .ltext.1
# MAP:         80d48 {{.*}} 8 8 .got.ltext.1
# MAP:         90000 {{.*}} 30d47 65536 .ltext.2

# PHDR:      ProgramHeaders [
# PHDR:        ProgramHeader {
# PHDR:          Type: PT_LOAD
# PHDR:          VirtualAddress: 0x10000
# PHDR:          PhysicalAddress: 0x10000000
# PHDR:          FileSize: 200007
# PHDR:          Flags [
# PHDR-NEXT:       PF_R
# PHDR-NEXT:       PF_X
# PHDR:        ProgramHeader {
# PHDR:          Type: PT_LOAD
# PHDR:          VirtualAddress: 0x40D48
# PHDR:          PhysicalAddress: 0x40D48
# PHDR:          Flags [
# PHDR-NEXT:       PF_R
# PHDR-NEXT:       PF_W
# PHDR:        ProgramHeader {
# PHDR:          Type: PT_LOAD
# PHDR:          VirtualAddress: 0x50000
# PHDR:          PhysicalAddress: 0x10040000
# PHDR:          FileSize: 200007
# PHDR:          Flags [
# PHDR-NEXT:       PF_R
# PHDR-NEXT:       PF_X
# PHDR:        ProgramHeader {
# PHDR:          Type: PT_LOAD
# PHDR:          VirtualAddress: 0x80D48
# PHDR:          PhysicalAddress: 0x80D48
# PHDR:          Flags [
# PHDR-NEXT:       PF_R
# PHDR-NEXT:       PF_W
# PHDR:        ProgramHeader {
# PHDR:          Type: PT_LOAD
# PHDR:          VirtualAddress: 0x90000
# PHDR:          PhysicalAddress: 0x10080000
# PHDR:          FileSize: 200007
# PHDR:          Flags [
# PHDR-NEXT:       PF_R
# PHDR-NEXT:       PF_X

#--- lds
MEMORY {
  ALL (rwx)  : ORIGIN = 0x10000, LENGTH = 0x10000000
  ROM (r)    : ORIGIN = 0x10000000, LENGTH = 0x10000000
}

SECTIONS {
  .ltext 0x10000 : ALIGN(0x10000) { *(.text .text.*) } > ALL AT > ROM
  .data : { *(.data) } > ALL
}

#--- test.s
.section .text.01, "axl", @progbits
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .text.02, "axl", @progbits
.globl _start_2
_start_2:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .text.03, "axl", @progbits
.globl _start_3
_start_3:
  movq foo@GOTPCREL(%rip), %rax
  .space 200000

.section .data, "aw", @progbits
.globl foo
foo:
  .quad 0
