//===- AcceptOrRevertPass.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H

#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"

namespace llvm {

class SBVecContext;

class AcceptOrRevert : public SBRegionPass {
  SBVecContext &Ctxt;

public:
  AcceptOrRevert(SBVecContext &Ctxt)
      : SBRegionPass("AcceptOrRevert", "accept-or-revert"), Ctxt(Ctxt) {}
  bool runOnRegion(SBRegion &Rgn) final;
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
