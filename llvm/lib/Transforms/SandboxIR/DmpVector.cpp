//===- DmpVector.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/SandboxIR/DmpVector.h"
#include "llvm/Transforms/SandboxIR/SandboxIR.h"

using namespace llvm;

void DmpVector<SBValue *>::init(const DmpVector<Value *> &Vec,
                                const SBBasicBlock &SBBB) {
  reserve(Vec.size());
  for (Value *V : Vec) {
    SBValue *SBV = SBBB.getContext().getSBValue(V);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    push_back(SBV);
  }
}

DmpVector<Value *> DmpVector<SBValue *>::getLLVMValueVector() const {
  DmpVector<Value *> Vec(size());
  for (auto *N : *this)
    Vec.push_back(ValueAttorney::getValue(N));
  return Vec;
}

void DmpVector<SBInstruction *>::init(const DmpVector<Value *> &Vec,
                                      const SBBasicBlock &SBBB) {
  reserve(Vec.size());
  for (Instruction *I : Vec.instrRange()) {
    auto *SBV = SBBB.getContext().getSBValue(I);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    assert(isa<SBInstruction>(SBV) && "Not a SBInstruction!");
    push_back(cast<SBInstruction>(SBV));
  }
}

DmpVector<Value *>
DmpVector<Value *>::create(const DmpVector<SBValue *> &SBVec) {
  DmpVector<Value *> Vec;
  Vec.reserve(SBVec.size());
  for (const auto *SBV : SBVec)
    Vec.push_back(ValueAttorney::getValue(SBV));
  return Vec;
}
