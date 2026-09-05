; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary 1.ll -o 1.bc
; RUN: opt -module-summary 2.ll -o 2.bc
; RUN: llvm-lto -thinlto -o 3 1.bc 2.bc
; RUN: opt -S -passes=function-import -summary-file 3.thinlto.bc 1.bc | FileCheck %s

; CHECK: @large_var = available_externally hidden global [250 x i8] zeroinitializer, code_model "large"
; CHECK: @small_var = available_externally hidden global [150 x i8] zeroinitializer, code_model "small"
; CHECK: define i32 @main()
; CHECK: define available_externally void @foo() code_model "large"
; CHECK: !{i32 9, !"Code Model", i32 3}
; CHECK: !{i32 9, !"Large Data Threshold", i32 100}

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@large_var = external global [250 x i8]
@small_var = external global [150 x i8]

define i32 @main() {
entry:
  call void @foo()
  %v1 = load i8, ptr @large_var
  %v2 = load i8, ptr @small_var
  %add = add i8 %v1, %v2
  %conv = zext i8 %add to i32
  ret i32 %conv
}

declare void @foo()

!llvm.module.flags = !{!0, !1}
!0 = !{i32 9, !"Code Model", i32 3}
!1 = !{i32 9, !"Large Data Threshold", i32 100}

;--- 2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@large_var = hidden global [250 x i8] zeroinitializer
@small_var = hidden global [150 x i8] zeroinitializer

define void @foo() code_model "large" {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 9, !"Code Model", i32 3}
!1 = !{i32 9, !"Large Data Threshold", i32 200}
