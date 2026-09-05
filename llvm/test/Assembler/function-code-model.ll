; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define void @f1() code_model "tiny" {
  ret void
}

define void @f2() code_model "small" {
  ret void
}

define void @f3() code_model "kernel" {
  ret void
}

define void @f4() code_model "medium" {
  ret void
}

define void @f5() code_model "large" {
  ret void
}

; CHECK: define void @f1() code_model "tiny" {
; CHECK: define void @f2() code_model "small" {
; CHECK: define void @f3() code_model "kernel" {
; CHECK: define void @f4() code_model "medium" {
; CHECK: define void @f5() code_model "large" {
