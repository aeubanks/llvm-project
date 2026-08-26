# REQUIRES: x86
## Verify that TLS relocations (IE, GD, LD, TLSDESC) from partitioned text
## sections allocate their GOT entries in the corresponding GotPartitionSection
## rather than the primary .got, and support relaxation in executables.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/dso.s -o %t/dso.o

## Test unrelaxed TLS relocations in -shared mode.
# RUN: ld.lld -shared --got-partition-threshold=0x1000 %t/a.o %t/dso.o -o %t/shared.so
# RUN: llvm-objdump -d --no-print-imm-hex %t/shared.so > %t/shared.txt
# RUN: llvm-readelf -S -r %t/shared.so >> %t/shared.txt
# RUN: FileCheck %s --check-prefix=SHARED < %t/shared.txt

## Test TLS relaxation (GD/TLSDESC -> IE, IE/LD -> LE) in executable mode.
# RUN: ld.lld -shared %t/dso.o -o %t/dso.so
# RUN: ld.lld --got-partition-threshold=0x1000 %t/a.o %t/dso.so -o %t/exec
# RUN: llvm-objdump -d --no-print-imm-hex %t/exec > %t/exec.txt
# RUN: llvm-readelf -S -r %t/exec >> %t/exec.txt
# RUN: FileCheck %s --check-prefix=EXEC < %t/exec.txt

# SHARED:      <_start>:
# SHARED-NEXT:   movq {{.*}}(%rip), %rax # 0x[[#%x, GOT_FOO:]]
# SHARED-NEXT:   movq %fs:(%rax), %rax
# SHARED-NEXT:   leaq {{.*}}(%rip), %rdi # 0x[[#%x, GOT_BAR:]]
# SHARED-NEXT:   callq {{.*}} <__tls_get_addr@plt>
# SHARED-NEXT:   leaq {{.*}}(%rip), %rdi # 0x[[#%x, GOT_LD:]]
# SHARED-NEXT:   callq {{.*}} <__tls_get_addr@plt>
# SHARED-NEXT:   leaq {{.*}}(%rax), %rcx
# SHARED-NEXT:   leaq {{.*}}(%rip), %rax # 0x[[#%x, GOT_QUX:]]
# SHARED-NEXT:   callq *(%rax)
# SHARED-NEXT:   movl %fs:(%rax), %eax

# SHARED:      .got.ltext.0 PROGBITS [[#%.16x, GOT_FOO]] {{.*}} 000038
# SHARED:      .got PROGBITS {{.*}} 000000
# SHARED:      Relocation section '.rela.dyn' {{.*}} contains 5 entries:
# SHARED:      [[#%.16x, GOT_LD]]      {{.*}} R_X86_64_DTPMOD64 0
# SHARED-NEXT: [[#%.16x, GOT_FOO]]     {{.*}} R_X86_64_TPOFF64  {{.*}} foo + 0
# SHARED-NEXT: [[#%.16x, GOT_BAR]]     {{.*}} R_X86_64_DTPMOD64 {{.*}} bar + 0
# SHARED-NEXT: [[#%.16x, GOT_BAR + 8]] {{.*}} R_X86_64_DTPOFF64 {{.*}} bar + 0
# SHARED-NEXT: [[#%.16x, GOT_QUX]]     {{.*}} R_X86_64_TLSDESC  {{.*}} qux + 0

# EXEC:      <_start>:
## foo (IE) relaxed to LE:
# EXEC-NEXT:   movq $-16, %rax
# EXEC-NEXT:   movq %fs:(%rax), %rax
## bar (GD) relaxed to IE targeting .got.ltext.0:
# EXEC-NEXT:   movq %fs:0, %rax
# EXEC-NEXT:   addq {{.*}}(%rip), %rax # 0x[[#%x, GOT_BAR:]]
## baz (LD) relaxed to LE:
# EXEC-NEXT:   movq %fs:0, %rax
# EXEC-NEXT:   leaq -8(%rax), %rcx
## qux (TLSDESC) relaxed to IE targeting .got.ltext.0:
# EXEC-NEXT:   movq {{.*}}(%rip), %rax # 0x[[#%x, GOT_QUX:]]
# EXEC-NEXT:   nop
# EXEC-NEXT:   movl %fs:(%rax), %eax

# EXEC:      .got.ltext.0 PROGBITS [[#%.16x, GOT_BAR]] {{.*}} 000010
# EXEC:      .got PROGBITS {{.*}} 000000
# EXEC:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# EXEC:      [[#%.16x, GOT_BAR]] {{.*}} R_X86_64_TPOFF64 {{.*}} bar + 0
# EXEC-NEXT: [[#%.16x, GOT_QUX]] {{.*}} R_X86_64_TPOFF64 {{.*}} qux + 0

#--- a.s
.section .ltext, "axl"
.globl _start
_start:
  movq foo@GOTTPOFF(%rip), %rax
  movq %fs:(%rax), %rax

  .byte 0x66
  leaq bar@TLSGD(%rip), %rdi
  .word 0x6666
  rex64
  call __tls_get_addr@PLT

  leaq baz@TLSLD(%rip), %rdi
  call __tls_get_addr@PLT
  leaq baz@DTPOFF(%rax), %rcx

  leaq qux@TLSDESC(%rip), %rax
  call *qux@TLSCALL(%rax)
  movl %fs:(%rax), %eax

.section .ltext.pad, "axl", @nobits
.space 0x4000

.section .tbss, "awT", @nobits
.globl foo
foo:
  .quad 0
baz:
  .quad 0

#--- dso.s
.section .tbss, "awT", @nobits
.globl bar
bar:
  .quad 0
.globl qux
qux:
  .quad 0

.text
.globl __tls_get_addr
.type __tls_get_addr, @function
__tls_get_addr:
  ret
