# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o /dev/null 2>&1 | FileCheck %s
# RUN: ld.lld %t/a.o %t/b.o -o /dev/null --no-warn-different-section-flags 2>&1 | FileCheck %s --check-prefix=NO --allow-empty

# CHECK: warning: incompatible section flags for foo
# CHECK-NEXT: >>> {{.*}}a.o:(foo): 0x10000003
# CHECK-NEXT: >>> {{.*}}b.o:(foo): 0x3

# NO-NOT: incompatible section flags

#--- a.s
.section foo,"awl",@progbits

.type   hello, @object
.globl  hello
hello:
.long   1

#--- b.s
.section foo,"aw",@progbits

.type   hello2, @object
.globl  hello2
hello2:
.long   1
