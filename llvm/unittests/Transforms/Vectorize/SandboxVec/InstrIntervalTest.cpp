//===- InstrIntervalTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("InstrIntervalTest", errs());
  return Mod;
}

TEST(InstrInterval, Basic) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  store i8 2, ptr %ptr
  store i8 3, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  SBContext Ctxt(C);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  {
    InstrInterval InstrInterval(I0, I1);
    EXPECT_TRUE(InstrInterval.from()->comesBefore(InstrInterval.to()));
    EXPECT_EQ(InstrInterval.from(), I0);
    EXPECT_EQ(InstrInterval.to(), I1);
  }
  {
    InstrInterval InstrInterval(I0, I0);
    EXPECT_EQ(InstrInterval.from(), InstrInterval.to());
    EXPECT_EQ(InstrInterval.from(), I0);
    EXPECT_EQ(InstrInterval.to(), I0);
  }
  {
    InstrInterval InstrInterval(I1, I0);
    EXPECT_TRUE(InstrInterval.from()->comesBefore(InstrInterval.to()));
    EXPECT_EQ(InstrInterval.from(), I0);
    EXPECT_EQ(InstrInterval.to(), I1);
  }

  // Check ArrayRef constructor.
  {
    InstrInterval InstrInterval({I0, I1, I4});
    EXPECT_TRUE(InstrInterval.from()->comesBefore(InstrInterval.to()));
    EXPECT_EQ(InstrInterval.from(), I0);
    EXPECT_EQ(InstrInterval.to(), I4);
  }
  {
    InstrInterval InstrInterval({I4, I0, I2, I2});
    EXPECT_TRUE(InstrInterval.from()->comesBefore(InstrInterval.to()));
    EXPECT_EQ(InstrInterval.from(), I0);
    EXPECT_EQ(InstrInterval.to(), I4);
  }

  // Check contains(Instruction).
  {
    InstrInterval InstrInterval(I1, I1);
    EXPECT_TRUE(InstrInterval.contains(I1));
    EXPECT_FALSE(InstrInterval.contains(I0));
    EXPECT_FALSE(InstrInterval.contains(I2));
    EXPECT_FALSE(InstrInterval.contains(I3));
    EXPECT_FALSE(InstrInterval.contains(I4));
  }
  {
    InstrInterval InstrInterval(I1, I4);
    EXPECT_TRUE(InstrInterval.contains(I1));
    EXPECT_TRUE(InstrInterval.contains(I2));
    EXPECT_TRUE(InstrInterval.contains(I3));
    EXPECT_TRUE(InstrInterval.contains(I4));
    EXPECT_FALSE(InstrInterval.contains(I0));
  }

  // Check contains(InstrInterval).
  {
    InstrInterval InstrInterval1(I0, I2);
    InstrInterval InstrInterval2(I0, I1);
    EXPECT_TRUE(InstrInterval1.contains(InstrInterval2));
    InstrInterval InstrInterval3(I0, I2);
    EXPECT_TRUE(InstrInterval1.contains(InstrInterval3));
    InstrInterval InstrInterval4(I1, I2);
    EXPECT_TRUE(InstrInterval1.contains(InstrInterval4));
    InstrInterval InstrInterval5(I2, I3);
    EXPECT_FALSE(InstrInterval1.contains(InstrInterval5));
    InstrInterval InstrInterval6(I3, I4);
    EXPECT_FALSE(InstrInterval1.contains(InstrInterval6));
  }

  // Check operator== and operator!=.
  {
    InstrInterval InstrIntervalEmpty;
    InstrInterval InstrInterval1(I0, I0);
    EXPECT_FALSE(InstrInterval1 == InstrIntervalEmpty);
    EXPECT_FALSE(InstrIntervalEmpty == InstrInterval1);
    EXPECT_TRUE(InstrInterval1 != InstrIntervalEmpty);
    EXPECT_TRUE(InstrIntervalEmpty != InstrInterval1);
    InstrInterval InstrInterval2(I0, I1);
    EXPECT_TRUE(InstrInterval1 != InstrInterval2);
    EXPECT_TRUE(InstrInterval2 != InstrInterval1);
    InstrInterval InstrIntervalEmpty2;
    EXPECT_TRUE(InstrIntervalEmpty == InstrIntervalEmpty2);
    EXPECT_TRUE(InstrIntervalEmpty2 == InstrIntervalEmpty);
    InstrInterval InstrInterval3(I0, I1);
    EXPECT_TRUE(InstrInterval2 == InstrInterval3);
    EXPECT_TRUE(InstrInterval3 == InstrInterval2);
  }

  // Check getUnion().
  {
    InstrInterval InstrIntervalEmpty;
    InstrInterval InstrInterval1(I0, I0);
    auto UnionE1 = InstrIntervalEmpty.getUnionSingleSpan(InstrInterval1);
    EXPECT_TRUE(UnionE1 == InstrInterval1);
    auto Union1E = InstrInterval1.getUnionSingleSpan(InstrInterval1);
    EXPECT_TRUE(Union1E == InstrInterval1);
    InstrInterval InstrInterval2(I1, I1);
    auto Union12 = InstrInterval1.getUnionSingleSpan(InstrInterval2);
    EXPECT_TRUE(Union12 == InstrInterval2.getUnionSingleSpan(InstrInterval1));
    EXPECT_EQ(Union12.from(), I0);
    EXPECT_EQ(Union12.to(), I1);

    InstrInterval InstrInterval3(I4, I4);
    auto Union13 = InstrInterval1.getUnionSingleSpan(InstrInterval3);
    EXPECT_EQ(Union13.from(), I0);
    EXPECT_EQ(Union13.to(), I4);
  }

  // Check getIntersection().
  {
    InstrInterval InstrIntervalEmpty;
    InstrInterval InstrInterval1(I0, I0);
    EXPECT_TRUE(InstrIntervalEmpty.getIntersection(InstrInterval1).empty());
    EXPECT_TRUE(InstrInterval1.getIntersection(InstrIntervalEmpty).empty());

    InstrInterval InstrInterval2(I1, I1);
    EXPECT_TRUE(InstrInterval1.getIntersection(InstrInterval2).empty());
    EXPECT_TRUE(InstrInterval2.getIntersection(InstrInterval1).empty());

    InstrInterval InstrInterval3(I0, I3);
    InstrInterval InstrInterval4(I1, I2);
    auto Intersection34 = InstrInterval3.getIntersection(InstrInterval4);
    EXPECT_TRUE(InstrInterval4.getIntersection(InstrInterval3) ==
                Intersection34);
    EXPECT_EQ(Intersection34.from(), I1);
    EXPECT_EQ(Intersection34.to(), I2);

    InstrInterval InstrInterval5(I2, I4);
    auto Intersection35 = InstrInterval3.getIntersection(InstrInterval5);
    EXPECT_TRUE(InstrInterval5.getIntersection(InstrInterval3) ==
                Intersection35);
    EXPECT_EQ(Intersection35.from(), I2);
    EXPECT_EQ(Intersection35.to(), I3);
  }

  // Check the difference operator-().
  {
    // Same FromI
    InstrInterval InstrInterval1(I0, I3);
    InstrInterval InstrInterval2(I0, I2);
    auto Diff12Vec = InstrInterval1 - InstrInterval2;
    EXPECT_EQ(Diff12Vec.size(), 1u);
    auto Diff12 = Diff12Vec.back();
    EXPECT_EQ(Diff12.from(), I3);
    EXPECT_EQ(Diff12.to(), I3);

    // Same ToI
    InstrInterval InstrInterval3(I2, I3);
    auto Diff13Vec = InstrInterval1 - InstrInterval3;
    EXPECT_EQ(Diff13Vec.size(), 1u);
    auto Diff13 = Diff13Vec.back();
    EXPECT_EQ(Diff13.from(), I0);
    EXPECT_EQ(Diff13.to(), I1);

    // Disjoint
    InstrInterval InstrInterval4(I4, I4);
    auto Diff14Vec = InstrInterval1 - InstrInterval4;
    EXPECT_EQ(Diff14Vec.size(), 1u);
    EXPECT_TRUE(Diff14Vec.back() == InstrInterval1);

    // Overlap
    InstrInterval InstrInterval5(I2, I4);
    auto Diff15Vec = InstrInterval1 - InstrInterval5;
    EXPECT_EQ(Diff15Vec.size(), 1u);
    EXPECT_TRUE(Diff15Vec.back() == InstrInterval(I0, I1));

    // 2 results
    InstrInterval InstrInterval6(I2, I2);
    auto Diff16Vec = InstrInterval1 - InstrInterval6;
    EXPECT_EQ(Diff16Vec.size(), 2u);
    EXPECT_TRUE(Diff16Vec[0] == InstrInterval(I0, I1));
    EXPECT_TRUE(Diff16Vec[1] == InstrInterval(I3, I3));

    // A - A
    auto Diff11Vec = InstrInterval1 - InstrInterval1;
    EXPECT_EQ(Diff11Vec.size(), 1u);
    EXPECT_TRUE(Diff11Vec.back().empty());
    InstrInterval InstrIntervalEmpty;

    // A - Empty
    auto Diff1E = InstrInterval1 - InstrIntervalEmpty;
    EXPECT_EQ(Diff1E.size(), 1u);
    EXPECT_TRUE(Diff1E.back() == InstrInterval1);
  }

  // InstrInterval iterator
  {
    InstrInterval InstrInterval(I0, I1);
    SmallVector<SBInstruction *> Instrs;
    for (SBInstruction &IRef : InstrInterval)
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I0);
    EXPECT_EQ(Instrs[1], I1);

    Instrs.clear();
    const auto &ConstInstrInterval = InstrInterval;
    for (SBInstruction &IRef : ConstInstrInterval)
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I0);
    EXPECT_EQ(Instrs[1], I1);
  }
  // InstrInterval reverse iterator
  {
    InstrInterval InstrInterval(I0, I1);
    SmallVector<SBInstruction *> Instrs;
    for (SBInstruction &IRef : reverse(InstrInterval))
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I1);
    EXPECT_EQ(Instrs[1], I0);
  }

  // Check end()-- when InstrInterval.ToI is the BB terminator.
  {
    InstrInterval InstrInterval(I4);
    auto It = InstrInterval.end();
    --It;
    EXPECT_EQ(&*It, I4);
  }

  // Check extend(I)
  {
    InstrInterval InstrInterval(I2);
    InstrInterval.extend(I4);
    EXPECT_EQ(InstrInterval.from(), I2);
    EXPECT_EQ(InstrInterval.to(), I4);

    InstrInterval.extend(I1);
    EXPECT_EQ(InstrInterval.from(), I1);
    EXPECT_EQ(InstrInterval.to(), I4);

    InstrInterval.extend(I3);
    EXPECT_EQ(InstrInterval.from(), I1);
    EXPECT_EQ(InstrInterval.to(), I4);
  }

  // Erase an Instruction that is not in the region.
  {
    InstrInterval InstrInterval(I2);
#ifndef NDEBUG
    EXPECT_DEATH(InstrInterval.erase(I1), ".*not in interval.*");
#endif
    InstrInterval.erase(I1, /*CheckContained=*/false);
    EXPECT_EQ(InstrInterval.from(), I2);
    EXPECT_EQ(InstrInterval.to(), I2);
  }
  {
    InstrInterval InstrInterval(I2, I3);
#ifndef NDEBUG
    EXPECT_DEATH(InstrInterval.erase(I1), ".*not in interval.*");
#endif
    InstrInterval.erase(I1, /*CheckContained=*/false);
    EXPECT_EQ(InstrInterval.from(), I2);
    EXPECT_EQ(InstrInterval.to(), I3);
  }
}

TEST(InstrInterval, ContainsIt) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  store i8 2, ptr %ptr
  store i8 3, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  SBContext Ctxt(C);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  (void)I0;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;

  {
    InstrInterval R(I3, I3);
    EXPECT_TRUE(R.contains(I3->getIterator()));
    EXPECT_TRUE(R.contains(I4->getIterator()));
    EXPECT_FALSE(R.contains(I2->getIterator()));
    EXPECT_FALSE(R.contains(BB->end()));
  }
  {
    InstrInterval R(I2, I4);
    EXPECT_TRUE(R.contains(I4->getIterator()));
    EXPECT_TRUE(R.contains(BB->end()));
    EXPECT_FALSE(R.contains(I1->getIterator()));
  }
}

TEST(InstrInterval, ExtendEmpty) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  SBContext Ctxt(C);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;

  InstrInterval R;
  R.extend(I0);
  EXPECT_TRUE(R.contains(I0));
}

#ifndef NDEBUG
TEST(InstrInterval, InstrIntervalNotify) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  store i8 2, ptr %ptr
  store i8 3, ptr %ptr
  ret void
}
)IR");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBVecContext Ctxt(C, AA);
  Function &F = *M->getFunction("foo");

  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  {
    Ctxt.disableCallbacks();
    InstrInterval InstrInterval(I0, I4);
    auto WhereIt = I0->getNextNode()->getIterator();
    // Move FromI to itself.
    InstrInterval.notifyMoveInstr(I0, WhereIt, BB);
    I1->moveBefore(*BB, WhereIt);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move ToI to itself.
    WhereIt = BB->end();
    InstrInterval.notifyMoveInstr(I4, WhereIt, BB);
    I4->moveBefore(*BB, WhereIt);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move instruction to FromI.
    InstrInterval.notifyMoveInstr(I1, I0->getIterator(), BB);
    I1->moveBefore(I0);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move FromI.
    InstrInterval.notifyMoveInstr(I1, I2->getIterator(), BB);
    I1->moveBefore(I2);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move instruction to ToI.
    InstrInterval.notifyMoveInstr(I2, BB->end(), BB);
    I2->moveBefore(*BB, BB->end());
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move ToI to FromI.
    InstrInterval.notifyMoveInstr(I2, I1->getIterator(), BB);
    I2->moveBefore(I1);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());

    // Move internal instructions.
    InstrInterval.notifyMoveInstr(I3, I4->getIterator(), BB);
    I3->moveBefore(I4);
    EXPECT_EQ(InstrInterval.from(), &BB->front());
    EXPECT_EQ(InstrInterval.to(), &BB->back());
  }
}
#endif

#ifndef NDEBUG
TEST(InstrInterval, InstrIntervalNotify2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  store i8 2, ptr %ptr
  store i8 3, ptr %ptr
  ret void
}
)IR");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBVecContext Ctxt(C, AA);

  Function &F = *M->getFunction("foo");
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  (void)I0;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  (void)I4;

  Ctxt.disableCallbacks();
  {
    // Single-instruction region.
    InstrInterval InstrInterval(I1, I1);
    // Trying to move I1 out of the region should crash!
    EXPECT_DEATH(InstrInterval.notifyMoveInstr(I1, I3->getIterator(), BB),
                 ".*");
    // Move to itself should be a nop.
    InstrInterval.notifyMoveInstr(I1, std::next(I1->getIterator()), BB);
    I1->moveBefore(*BB, std::next(I1->getIterator()));
    EXPECT_EQ(InstrInterval.from(), I1);
    EXPECT_EQ(InstrInterval.to(), I1);
  }
  {
    InstrInterval InstrInterval(I2, I4);
    // To help debug the scheduler, trying to move I1, an external instruction,
    // into the region, should crash. If the scheduler wants to move new
    // instructions into the scheduled region it should first extend the DAG's
    // region to include them.
    EXPECT_DEATH(InstrInterval.notifyMoveInstr(I1, I4->getIterator(), BB),
                 ".*");
  }
  {
    // Moving I2 before I2 should not change the region.
    InstrInterval InstrInterval(I1, I2);
    InstrInterval.notifyMoveInstr(I2, I2->getIterator(), BB);
    I2->moveBefore(*BB, I2->getIterator());
    EXPECT_EQ(InstrInterval.from(), I1);
    EXPECT_EQ(InstrInterval.to(), I2);
  }
  {
    // Moving I2 before I1 should change the region to {I2, I1}.
    InstrInterval InstrInterval(I1, I2);
    InstrInterval.notifyMoveInstr(I2, I1->getIterator(), BB);
    I2->moveBefore(*BB, I1->getIterator());
    EXPECT_EQ(InstrInterval.from(), I2);
    EXPECT_EQ(InstrInterval.to(), I1);
    // Revert IR
    I2->moveAfter(I1);
  }
  {
    // Moving I1 after I2 should change the region to {I2, I1}.
    InstrInterval InstrInterval(I1, I2);
    InstrInterval.notifyMoveInstr(I1, std::next(I2->getIterator()), BB);
    I1->moveBefore(*BB, std::next(I2->getIterator()));
    EXPECT_EQ(InstrInterval.from(), I2);
    EXPECT_EQ(InstrInterval.to(), I1);
  }
}
#endif
