# REQUIRES: x86
## A thunk in a section without a GOT partition uses the primary GOT. The
## destination's aux entry is allocated while scanning relocations, long before
## the thunk is created, so it is not necessarily the most recent one.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: ld.lld -shared -T %t/lds %t/test.o -o %t/test.so
# RUN: llvm-objdump -d --no-print-imm-hex %t/test.so > %t/out.txt
# RUN: llvm-readelf -r %t/test.so >> %t/out.txt
# RUN: FileCheck %s < %t/out.txt

# CHECK:      <__X86_64Thunk_foo>:
# CHECK-NEXT:   jmpq *{{.*}}(%rip) # 0x[[#%x, FOO:]]
# CHECK:      <__X86_64Thunk_bar>:
# CHECK-NEXT:   jmpq *{{.*}}(%rip) # 0x[[#%x, BAR:]]
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64Thunk_foo>
# CHECK-NEXT:   callq {{.*}} <__X86_64Thunk_bar>

## Each thunk gets the GOT entry of its own destination.
# CHECK:      [[#%.16x, FOO]] {{.*}} R_X86_64_GLOB_DAT {{.*}} foo + 0
# CHECK-NEXT: [[#%.16x, BAR]] {{.*}} R_X86_64_GLOB_DAT {{.*}} bar + 0

#--- lds
SECTIONS {
  .plt 0x1000 : { *(.plt) }
  .ltext 0x2000 : { *(.ltext) }
  .text 0x100000000 : { *(.text) }
  .got : { *(.got) }
  .got.plt : { *(.got.plt) }
}

#--- test.s
.text
.globl _start
.type _start, @function
_start:
  call foo@PLT
  call bar@PLT
  ret

## Large executable sections are what enables thunks.
.section .ltext, "axl"
.globl big
.type big, @function
big:
  ret
