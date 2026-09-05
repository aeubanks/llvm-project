; RUN: llvm-as %s -o %t0.o
; RUN: llvm-as < %p/Inputs/codemodel-3.ll > %t1.o
; RUN: llvm-lto2 run -save-temps -r %t0.o,_start,px -r %t1.o,bar,px -r %t0.o,_GLOBAL_OFFSET_TABLE_, \
; RUN:   %t0.o %t1.o -o %t2.s
; RUN: llvm-dis %t2.s.0.0.preopt.bc -o - | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 9, !"Code Model", i32 4}

@data = internal constant [0 x i32] []

define ptr @_start() nounwind readonly {
entry:
    ret ptr @data
}

; CHECK: define ptr @_start()
; CHECK: define void @bar(ptr %a, i8 %b, i32 %c) code_model "small"

