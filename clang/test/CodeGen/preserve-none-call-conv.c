// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-passes < %s | FileCheck %s --check-prefix=YES
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -disable-llvm-passes < %s | FileCheck %s --check-prefix=NO
// RUN: %clang_cc1 -triple arm64-unknown-unknown -emit-llvm -disable-llvm-passes < %s | FileCheck %s --check-prefix=NO

// Only supported for x86-64 (for now).
// YES: define dso_local ghccc void @foo()
// NO:  define dso_local void @foo()
void foo(void) __attribute__((preserve_none)) {
}

