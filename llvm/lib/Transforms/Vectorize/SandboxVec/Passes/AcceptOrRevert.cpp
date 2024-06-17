//===- AcceptOrRevertPass.cpp - Check cost and accept/revert the region ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/AcceptOrRevert.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"

using namespace llvm;

extern llvm::cl::opt<int> CostThreshold;

bool AcceptOrRevert::runOnRegion(SBRegion &Rgn) {
  InstructionCost CostAfterMinusBefore = Rgn.getVectorMinusScalarCost();
  if (CostAfterMinusBefore < -CostThreshold) {
    Ctxt.getTracker().accept();
    return false;
  }
  auto &Sched = *Ctxt.getScheduler(Rgn.getBB());
  // Don't maintain the view when reverting.
  Sched.getDAG().resetView();
  // We don't maintain the scheduler when reverting.
  Sched.clearState();
  // Revert the IR.
  Ctxt.getTracker().revert();
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  Sched.getDAG().verify(/*CheckReadyCnt=*/false);
#endif
  SBRegionAttorney::dropMetadata(Rgn);
  return true;
}
