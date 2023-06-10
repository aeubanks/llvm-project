; RUN: opt -passes='require<memoryssa>,gvn' -S < %s

; check this doesn't crash

@global = external constant i64

declare void @foo(i64)

define void @f() {
bb:
  br i1 true, label %bb2, label %bb1

bb1:
  %load = load i64, ptr @global
  br label %bb2

bb2:
  %load3 = load i64, ptr @global
  call void @foo(i64 %load3)
  ret void
}
