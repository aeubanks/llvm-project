# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-print-imm-hex %t | FileCheck %s
# RUN: llvm-readobj -S %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-readelf -x .got.ltext.0 %t | FileCheck --check-prefix=HEX-GOT-LTEXT %s
# RUN: llvm-readelf -x .got %t | FileCheck --check-prefix=HEX-GOT %s

# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-objdump -d --no-print-imm-hex %t.pie | FileCheck %s --check-prefix=PIE-ASM
# RUN: llvm-readelf -r %t.pie | FileCheck --check-prefix=PIE %s
# RUN: llvm-readelf -d %t.pie | FileCheck --check-prefix=RELACOUNT %s

# RUN: ld.lld -pie --pack-dyn-relocs=relr %t.o -o %t.pie.relr
# RUN: llvm-objdump -d --no-print-imm-hex %t.pie.relr | FileCheck %s --check-prefix=PIE-ASM
# RUN: llvm-readelf -r %t.pie.relr | FileCheck --check-prefix=PIE-RELR %s

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -d --no-print-imm-hex %t.so | FileCheck --check-prefix=SHARED-ASM %s
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=SHARED-REL %s

# SEC:        Name: .got.ltext.0 (
# SEC-NEXT:   Type: SHT_PROGBITS
# SEC-NEXT:   Flags [
# SEC-NEXT:     SHF_ALLOC
# SEC-NEXT:     SHF_WRITE
# SEC-NEXT:     SHF_X86_64_LARGE
# SEC-NEXT:   ]
# SEC-NEXT:   Address: 0x202288
# SEC-NEXT:   Offset:
# SEC-NEXT:   Size: 8

# SEC:        Name: .got (
# SEC-NEXT:   Type: SHT_PROGBITS
# SEC-NEXT:   Flags [
# SEC-NEXT:     SHF_ALLOC
# SEC-NEXT:     SHF_WRITE
# SEC-NEXT:   ]
# SEC-NEXT:   Address: 0x802052A8
# SEC-NEXT:   Offset:
# SEC-NEXT:   Size: 8

# HEX-GOT-LTEXT: Hex dump of section '.got.ltext.0':
# HEX-GOT-LTEXT-NEXT: 0x00202288 98422080 00000000

# HEX-GOT: Hex dump of section '.got':
# HEX-GOT-NEXT: 0x802052a8 76122000 00000000

# PIE:        Relocation section '.rela.dyn' at offset {{.*}} contains 2 entries:
# PIE:        0000000000002338 {{.*}} R_X86_64_RELATIVE 80004348
# PIE-NEXT:   0000000080005428 {{.*}} R_X86_64_RELATIVE 1326

# RELACOUNT:  0x000000006ffffff9 (RELACOUNT) 2

# PIE-RELR: Relocation section '.relr.dyn'
# PIE-RELR: {{[0-9a-f]+}} {{[0-9a-f]+}} _DYNAMIC + 0x{{[0-9a-f]+}}

# SHARED-REL:     Relocation section '.rela.dyn' at offset {{.*}} contains 2 entries:
# SHARED-REL:     0000000000002400 {{.*}} R_X86_64_GLOB_DAT {{[0-9a-f]+}} high + 0
# SHARED-REL-NEXT: 0000000000002408 {{.*}} R_X86_64_GLOB_DAT {{[0-9a-f]+}} _start + 0

# CHECK-LABEL: <__X86_64Thunk__start>:
# CHECK-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x202288
# CHECK-LABEL: <high>:
# CHECK-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <high>
# CHECK-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk__start>
# CHECK-NEXT:  {{.*}}: c3                            retq
# CHECK-LABEL: <__X86_64Thunk_high>:
# CHECK-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x802052a8
# CHECK-LABEL: <_start>:
# CHECK-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>
# CHECK-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>

# PIE-ASM-LABEL: <__X86_64Thunk__start>:
# PIE-ASM-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x23{{[0-9a-f]+}}
# PIE-ASM-LABEL: <__X86_64Thunk_high>:
# PIE-ASM-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x80005428

# SHARED-ASM-LABEL: <__X86_64Thunk__start>:
# SHARED-ASM-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x2408
# SHARED-ASM-LABEL: <__X86_64Thunk_high>:
# SHARED-ASM-NEXT:  {{.*}}: ff 25 {{.*}} jmpq *{{.*}}(%rip) # 0x2400
# SHARED-ASM-LABEL: <high>:
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk_high>
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <__X86_64Thunk__start>
# SHARED-ASM-LABEL: <_start>:
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <high@plt>
# SHARED-ASM-NEXT:  {{.*}}: e8 {{.*}} callq {{.*}} <high@plt>

.section .ltext, "axl"
.globl high
.type high, @function
high:
  call high
  call _start
  ret

.section .ltext.pad, "axl", @nobits
.space 0x80000000

.section .text, "ax"
.globl _start
.type _start, @function
_start:
  call high
  call high
