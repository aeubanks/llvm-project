; RUN: opt -S -passes=lowertypetests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck %s

; CHECK: @[[COMBINED:[0-9]+]] = private constant { [2048 x i8] } zeroinitializer, code_model "small"
; CHECK: @[[BYTEARRAY:[0-9]+]] = private constant [66 x i8] {{.*}}, code_model "small"
; CHECK: define private void @.cfi.jumptable() {{.*}} code_model "small"

@foo = constant [2048 x i8] zeroinitializer, !type !0, !type !1, !type !2, !type !3

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 130, !"typeid1"}
!2 = !{i32 4, !"typeid2"}
!3 = !{i32 1032, !"typeid2"}

define void @f() !type !4 {
  ret void
}

define void @g() !type !4 {
  ret void
}

define ptr @take_f() {
  ret ptr @f
}

define ptr @take_g() {
  ret ptr @g
}

!4 = !{i32 0, !"typeid3"}

define i1 @test_global(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

define i1 @test_func(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid3")
  ret i1 %x
}

declare i1 @llvm.type.test(ptr, metadata)

!llvm.module.flags = !{!5}
!5 = !{i32 1, !"Code Model", i32 4}
