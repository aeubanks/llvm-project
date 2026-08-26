# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s
# RUN: llvm-readobj -S %t | FileCheck --check-prefix=SEC %s

# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-objdump -d %t.pie | FileCheck %s
# RUN: llvm-readelf -r %t.pie | FileCheck --check-prefix=PIE %s
# RUN: llvm-readelf -d %t.pie | FileCheck --check-prefix=RELACOUNT %s

# RUN: ld.lld -pie --pack-dyn-relocs=relr %t.o -o %t.pie.relr
# RUN: llvm-objdump -d %t.pie.relr | FileCheck %s
# RUN: llvm-readelf -r %t.pie.relr | FileCheck --check-prefix=PIE-RELR %s

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -d %t.so | FileCheck --check-prefix=SHARED-ASM %s
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=SHARED-REL %s

# SEC:        Name: .got
# SEC-NEXT:   Type: SHT_PROGBITS
# SEC-NEXT:   Flags [
# SEC-NEXT:     SHF_ALLOC
# SEC-NEXT:     SHF_WRITE
# SEC-NEXT:   ]
# SEC-NEXT:   Address:
# SEC-NEXT:   Offset:
# SEC-NEXT:   Size: 8

# PIE:        Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# PIE:        R_X86_64_RELATIVE

# RELACOUNT:  0x000000006ffffff9 (RELACOUNT) 1

# PIE-RELR: Relocation section '.relr.dyn'
# PIE-RELR: {{[0-9a-f]+}} {{[0-9a-f]+}} _DYNAMIC + 0x{{[0-9a-f]+}}

# SHARED-REL:     Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# SHARED-REL:     {{[0-9a-f]+}} {{[0-9a-f]+}} R_X86_64_GLOB_DAT {{[0-9a-f]+}} high + 0

# CHECK-LABEL: <high>:
# CHECK-NEXT: {{.*}}: e8 {{.*}} callq {{.*}} <high>
# CHECK-NEXT: [[#%x, HIGH:]]: c3                            retq
# CHECK-LABEL: <__X86_64Thunk_high>:
# CHECK-NEXT: {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip)
# CHECK-LABEL: <_start>:
# CHECK-NEXT: {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>
# CHECK-NEXT: {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>

# SHARED-ASM-LABEL: <__X86_64Thunk_high>:
# SHARED-ASM-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip)
# SHARED-ASM-LABEL: <high>:
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>
# SHARED-ASM-LABEL: <_start>:
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <high@plt>
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <high@plt>

.section .ltext, "axl"
.globl high
.type high, @function
high:
  call high
  ret

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.section .text, "ax"
.globl _start
_start:
  call high
  call high
