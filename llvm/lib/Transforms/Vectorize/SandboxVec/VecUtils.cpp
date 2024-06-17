//===- VecUtils.cpp - Sandbox Vectorizer Utilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Transforms/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"

using namespace llvm;

template <typename LoadOrStoreT>
std::optional<int>
SBVecUtilsPrivileged::getPointerDiffInBytes(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                            ScalarEvolution &SE,
                                            const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  Value *PtrOp1 = ValueAttorney::getValue(I1->getPointerOperand());
  Value *PtrOp2 = ValueAttorney::getValue(I2->getPointerOperand());
  Value *Ptr1 = getUnderlyingObject(PtrOp1);
  Value *Ptr2 = getUnderlyingObject(PtrOp2);
  if (Ptr1 != Ptr2)
    return false;
  Type *ElemTy = Type::getInt8Ty(SE.getContext());
  // getPointersDiff(arg1, arg2) computes the difference arg2-arg1
  return getPointersDiff(ElemTy, PtrOp1, ElemTy, PtrOp2, DL, SE,
                         /*StrictCheck=*/false, /*CheckType=*/false);
}

template std::optional<int>
SBVecUtilsPrivileged::getPointerDiffInBytes<SBLoadInstruction>(
    SBLoadInstruction *, SBLoadInstruction *, ScalarEvolution &,
    const DataLayout &);
template std::optional<int>
SBVecUtilsPrivileged::getPointerDiffInBytes<SBStoreInstruction>(
    SBStoreInstruction *, SBStoreInstruction *, ScalarEvolution &,
    const DataLayout &);

template <typename LoadOrStoreT>
bool SBVecUtils::comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                  ScalarEvolution &SE, const DataLayout &DL) {
  auto Diff = SBVecUtilsPrivileged::getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  return *Diff > 0;
}

template bool SBVecUtils::comesBeforeInMem<SBLoadInstruction>(
    SBLoadInstruction *, SBLoadInstruction *, ScalarEvolution &,
    const DataLayout &);
template bool SBVecUtils::comesBeforeInMem<SBStoreInstruction>(
    SBStoreInstruction *, SBStoreInstruction *, ScalarEvolution &,
    const DataLayout &);

template <typename LoadOrStoreT>
bool SBVecUtils::areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                ScalarEvolution &SE, const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  auto Diff = SBVecUtilsPrivileged::getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  Type *ElmTy = getLoadStoreType(I1);
  int ElmBytes = DL.getTypeSizeInBits(ElmTy) / 8;
  return *Diff == ElmBytes;
}

template bool SBVecUtils::areConsecutive<SBLoadInstruction>(SBLoadInstruction *,
                                                            SBLoadInstruction *,
                                                            ScalarEvolution &,
                                                            const DataLayout &);
template bool SBVecUtils::areConsecutive<SBStoreInstruction>(
    SBStoreInstruction *, SBStoreInstruction *, ScalarEvolution &,
    const DataLayout &);

template <typename LoadOrStoreT>
bool SBVecUtils::areConsecutive(const DmpVector<SBValue *> &SBBndl,
                                ScalarEvolution &SE, const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  assert(isa<LoadOrStoreT>(SBBndl[0]) && "Expected SBStoreInst or SBLoadInst!");
  auto *LastS = cast<LoadOrStoreT>(SBBndl[0]);
  for (SBValue *V : drop_begin(SBBndl)) {
    assert(isa<LoadOrStoreT>(V) && "Unimplemented: we only support StoreInst!");
    auto *S = cast<LoadOrStoreT>(V);
    if (!SBVecUtils::areConsecutive(LastS, S, SE, DL))
      return false;
    LastS = S;
  }
  return true;
}

template bool SBVecUtils::areConsecutive<SBLoadInstruction>(
    const DmpVector<SBValue *> &, ScalarEvolution &, const DataLayout &);
template bool SBVecUtils::areConsecutive<SBStoreInstruction>(
    const DmpVector<SBValue *> &, ScalarEvolution &, const DataLayout &);

template <typename LoadOrStoreT>
Type *SBVecUtils::getLoadStoreType(LoadOrStoreT *SBI) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  if constexpr (std::is_same<LoadOrStoreT, SBLoadInstruction>::value) {
    return cast<SBLoadInstruction>(SBI)->getType();
  } else if constexpr (std::is_same<LoadOrStoreT, SBStoreInstruction>::value) {
    return cast<SBStoreInstruction>(SBI)->getValueOperand()->getType();
  } else {
    llvm_unreachable("Expected SBLoadInstruction or SBStoreInstruction");
  }
}

template Type *
SBVecUtils::getLoadStoreType<SBLoadInstruction>(SBLoadInstruction *);
template Type *
SBVecUtils::getLoadStoreType<SBStoreInstruction>(SBStoreInstruction *);

Type *SBVecUtils::getCommonScalarType(const DmpVector<SBValue *> &Bndl) {
  SBValue *V0 = Bndl[0];
  Type *Ty0 = SBVecUtils::getExpectedType(V0);
  Type *ScalarTy = SBVecUtils::getElementType(Ty0);
  for (auto *V : drop_begin(Bndl)) {
    Type *NTy = SBVecUtils::getExpectedType(V);
    Type *NScalarTy = SBVecUtils::getElementType(NTy);
    if (NScalarTy != ScalarTy)
      return nullptr;
  }
  return ScalarTy;
}

Type *SBVecUtils::getCommonScalarTypeFast(const DmpVector<SBValue *> &Bndl) {
  SBValue *V0 = Bndl[0];
  Type *Ty0 = SBVecUtils::getExpectedType(V0);
  Type *ScalarTy = SBVecUtils::getElementType(Ty0);
  assert(getCommonScalarType(Bndl) && "Expected common scalar type!");
  return ScalarTy;
}

Value *SBVecUtils::getExpectedValue(Instruction *I) {
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getValueOperand();
  if (auto *RI = dyn_cast<ReturnInst>(I))
    return RI->getReturnValue();
  return I;
}

SBValue *SBVecUtils::getExpectedValue(const SBInstruction *I) {
  if (auto *SI = dyn_cast<SBStoreInstruction>(I))
    return SI->getValueOperand();
  if (auto *RI = dyn_cast<SBReturnInstruction>(I))
    return RI->getReturnValue();
  return const_cast<SBInstruction *>(I);
}

Type *SBVecUtils::getExpectedType(Value *V) {
  if (isa<Instruction>(V)) {
    // A Return's value operand can be null if it returns void.
    if (auto *RI = dyn_cast<ReturnInst>(V)) {
      if (RI->getReturnValue() == nullptr)
        return RI->getType();
    }
    return getExpectedValue(cast<Instruction>(V))->getType();
  }
  return V->getType();
}

Type *SBVecUtils::getExpectedType(const SBValue *V) {
  if (isa<SBInstruction>(V)) {
    // A Return's value operand can be null if it returns void.
    if (auto *RI = dyn_cast<SBReturnInstruction>(V)) {
      if (RI->getReturnValue() == nullptr)
        return RI->getType();
    }
    return getExpectedValue(cast<SBInstruction>(V))->getType();
  }
  return V->getType();
}

unsigned SBVecUtils::getNumLanes(const SBValue *SBV) {
  Type *Ty = SBVecUtils::getExpectedType(SBV);
  return isa<FixedVectorType>(Ty) ? cast<FixedVectorType>(Ty)->getNumElements()
                                  : 1;
}

unsigned SBVecUtils::getNumLanes(const DmpVector<SBValue *> &Bndl) {
  unsigned Lanes = 0;
  for (SBValue *SBV : Bndl)
    Lanes += getNumLanes(SBV);
  return Lanes;
}

unsigned SBVecUtils::getNumBits(SBValue *SBV, const DataLayout &DL) {
  Type *Ty = SBVecUtils::getExpectedType(SBV);
  return DL.getTypeSizeInBits(Ty);
}
template <typename BndlT>
static unsigned getNumBitsCommon(const BndlT &Bndl, const DataLayout &DL) {
  unsigned Bits = 0;
  for (SBValue *SBV : Bndl)
    Bits += SBVecUtils::getNumBits(SBV, DL);
  return Bits;
}
unsigned SBVecUtils::getNumBits(const DmpVector<SBValue *> &Bndl,
                                const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

unsigned SBVecUtils::getNumBits(const DmpVector<SBInstruction *> &Bndl,
                                const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

Type *SBVecUtils::getWideType(Type *ElemTy, uint32_t NumElts) {
  if (ElemTy->isVectorTy()) {
    auto *VecTy = cast<FixedVectorType>(ElemTy);
    ElemTy = VecTy->getElementType();
    NumElts = VecTy->getNumElements() * NumElts;
  }
  return FixedVectorType::get(ElemTy, NumElts);
}

bool SBVecUtils::areInSameBB(const DmpVector<Value *> &Instrs) {
  if (Instrs.empty())
    return true;
  auto *I0 = cast<Instruction>(Instrs[0]);
  return all_of(drop_begin(Instrs.instrRange()), [I0](Instruction *I) {
    return I->getParent() == I0->getParent();
  });
}
bool SBVecUtils::areInSameBB(const DmpVector<SBValue *> &SBInstrs) {
  if (SBInstrs.empty())
    return true;
  auto *I0 = cast<SBInstruction>(SBInstrs[0]);
  return all_of(drop_begin(SBInstrs), [I0](SBValue *SBV) {
    return cast<SBInstruction>(SBV)->getParent() == I0->getParent();
  });
}

/// \Returns the next iterator after \p I, but will also skip PHIs if \p I is
/// a PHINode.
template <typename BBT, typename InstrT, typename PHIT>
static typename BBT::iterator getNextIteratorSkippingPHIs(InstrT *I) {
  auto NextIt = std::next(I->getIterator());
  typename BBT::iterator ItE = I->getParent()->end();
  while (NextIt != ItE && isa<PHIT>(&*NextIt))
    ++NextIt;
  return NextIt;
}

BasicBlock::iterator
SBVecUtils::getInsertPointAfter(const DmpVector<Value *> &Bndl, BasicBlock *BB,
                                bool SkipPHIs, bool SkipPads) {
  Instruction *LowestI = nullptr;
  for (Value *V : Bndl) {
    if (V == nullptr)
      continue;
    if (!isa<Instruction>(V))
      continue;
    Instruction *I = cast<Instruction>(V);
    // A nullptr instruction means that we are at the top of BB.
    Instruction *WhereI = I->getParent() == BB ? I : nullptr;
    if (LowestI == nullptr ||
        // If WhereI == null then a non-null LowestI will always come after it.
        (WhereI != nullptr && LowestI->comesBefore(WhereI)))
      LowestI = WhereI;
  }

  BasicBlock::iterator It;
  if (LowestI == nullptr)
    It = SkipPHIs ? BB->getFirstNonPHIIt() : BB->begin();
  else
    It = SkipPHIs
             ? getNextIteratorSkippingPHIs<BasicBlock, Instruction, PHINode>(
                   LowestI)
             : std::next(LowestI->getIterator());
  if (SkipPads) {
    if (It != BB->end()) {
      Instruction *I = &*It;
      if (LLVM_UNLIKELY(isa<LandingPadInst>(I) || isa<CatchPadInst>(I) ||
                        isa<CleanupPadInst>(I)))
        ++It;
    }
  }
  return It;
}

SBBasicBlock::iterator
SBVecUtils::getInsertPointAfter(const DmpVector<SBValue *> &Bndl,
                                SBBasicBlock *BB, bool SkipPHIs,
                                bool SkipPads) {
  SBInstruction *LowestI = nullptr;
  for (SBValue *V : Bndl) {
    if (V == nullptr)
      continue;
    if (!isa<SBInstruction>(V))
      continue;
    SBInstruction *I = cast<SBInstruction>(V);
    // A nullptr instruction means that we are at the top of BB.
    SBInstruction *WhereI = I->getParent() == BB ? I : nullptr;
    if (LowestI == nullptr ||
        // If WhereI == null then a non-null LowestI will always come after it.
        (WhereI != nullptr && LowestI->comesBefore(WhereI)))
      LowestI = WhereI;
  }

  SBBasicBlock::iterator It;
  if (LowestI == nullptr)
    It = SkipPHIs ? BB->getFirstNonPHIIt() : BB->begin();
  else
    It = SkipPHIs ? getNextIteratorSkippingPHIs<SBBasicBlock, SBInstruction,
                                                SBPHINode>(LowestI)
                  : std::next(LowestI->getIterator());
  if (SkipPads) {
    if (It != BB->end()) {
      SBInstruction *I = &*It;
      if (I->isPad())
        ++It;
    }
  }
  return It;
}

std::pair<SBBasicBlock *, SBBasicBlock::iterator>
SBVecUtils::getInsertPointAfterInstrs(const DmpVector<SBValue *> &InstrRange) {
  // Find the instr that is lowest in the BB.
  SBInstruction *LastI = nullptr;
  for (auto *SBV : InstrRange) {
    auto *I = cast<SBInstruction>(SBV);
    if (LastI == nullptr || LastI->comesBefore(I))
      LastI = I;
  }
  // If Bndl contains Arguments or Constants, use the beginning of the BB.
  SBBasicBlock::iterator WhereIt = std::next(LastI->getIterator());
  SBBasicBlock *WhereBB = LastI->getParent();
  return {WhereBB, WhereIt};
}

std::optional<int> SBVecUtils::getInsertLane(InsertElementInst *InsertI) {
  auto *IdxOp = InsertI->getOperand(2);
  if (!isa<ConstantInt>(IdxOp))
    return std::nullopt;
  return cast<ConstantInt>(IdxOp)->getZExtValue();
}

std::optional<int> SBVecUtils::getExtractLane(ExtractElementInst *ExtractI) {
  auto *IdxOp = ExtractI->getIndexOperand();
  if (!isa<ConstantInt>(IdxOp))
    return std::nullopt;
  return cast<ConstantInt>(IdxOp)->getZExtValue();
}
std::optional<int> SBVecUtils::getConstantIndex(Instruction *InsertOrExtractI) {
  if (auto *InsertI = dyn_cast<InsertElementInst>(InsertOrExtractI))
    return SBVecUtils::getInsertLane(InsertI);
  if (auto *ExtractI = dyn_cast<ExtractElementInst>(InsertOrExtractI))
    return SBVecUtils::getExtractLane(ExtractI);
  llvm_unreachable("Expect Insert or Extract only!");
}

bool SBVecUtils::differentMathFlags(const DmpVector<SBValue *> &SBBndl) {
  FastMathFlags FMF0 = cast<SBInstruction>(SBBndl[0])->getFastMathFlags();
  return any_of(drop_begin(SBBndl), [FMF0](auto *SBV) {
    return cast<SBInstruction>(SBV)->getFastMathFlags() != FMF0;
  });
}

bool SBVecUtils::differentWrapFlags(const DmpVector<SBValue *> &SBBndl) {
  bool NUW0 = cast<SBInstruction>(SBBndl[0])->hasNoUnsignedWrap();
  bool NSW0 = cast<SBInstruction>(SBBndl[0])->hasNoSignedWrap();
  return any_of(drop_begin(SBBndl), [NUW0, NSW0](auto *SBV) {
    return cast<SBInstruction>(SBV)->hasNoUnsignedWrap() != NUW0 ||
           cast<SBInstruction>(SBV)->hasNoSignedWrap() != NSW0;
  });
}

SBInstruction *SBVecUtils::getLowest(const DmpVector<SBValue *> &Instrs) {
  SBInstruction *LowestI = cast<SBInstruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<SBInstruction>(SBV);
    if (LowestI->comesBefore(SBI))
      LowestI = SBI;
  }
  return LowestI;
}

SBInstruction *SBVecUtils::getHighest(const DmpVector<SBValue *> &Instrs) {
  SBInstruction *HighestI = cast<SBInstruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<SBInstruction>(SBV);
    if (HighestI->comesAfter(SBI))
      HighestI = SBI;
  }
  return HighestI;
}

unsigned SBVecUtils::getAddrOperandIdx(SBInstruction *LoadOrStore) {
  if (isa<SBLoadInstruction>(LoadOrStore))
    return 0u;
  assert(isa<SBStoreInstruction>(LoadOrStore) &&
         "Expected only load or store!");
  return 1u;
}

void SBVecUtilsPrivileged::propagateMetadata(
    SBInstruction *SBI, const DmpVector<SBValue *> &SBVals) {
  auto *I = cast<Instruction>(ValueAttorney::getValue(SBI));
  // llvm::propagateMetadata() will propagate SBRegion metadata too, but we
  // don't want this to happen. So save the metadata here and set them later.
  auto *SavedSBRegionMD = I->getMetadata(SBRegion::MDKind);
  SmallVector<Value *> Vals;
  Vals.reserve(SBVals.size());
  for (auto *SBV : SBVals)
    Vals.push_back(ValueAttorney::getValue(SBV));
  llvm::propagateMetadata(I, Vals);
  // Override SBRegion meteadata with the value before propagateMetadata().
  I->setMetadata(SBRegion::MDKind, SavedSBRegionMD);

  // Now go over !dbg metadata. We copy the first !dbg metadata in `SBVals`.
  MDNode *DbgMD = nullptr;
  for (auto *SBV : SBVals)
    if (auto *I = dyn_cast<Instruction>(ValueAttorney::getValue(SBV)))
      if ((DbgMD = I->getMetadata(LLVMContext::MD_dbg)))
        break;
  if (DbgMD != nullptr)
    I->setMetadata(LLVMContext::MD_dbg, DbgMD);
}
