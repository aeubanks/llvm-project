//===- SandboxIRVec.cpp - Vectorization-specific SandboxIR ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"

using namespace llvm;

bool ShuffleMask::isIdentity() const {
  if (Indices.empty())
    return true;
  for (auto [Lane, Idx] : enumerate(Indices))
    if (Idx != (int)Lane)
      return false;
  return true;
}

bool ShuffleMask::isInOrder() const {
  if (Indices.empty())
    return true;
  int LastIdx = Indices.front();
  for (int Idx : drop_begin(Indices)) {
    if (Idx != LastIdx + 1)
      return false;
    LastIdx = Idx;
  }
  return true;
}

bool ShuffleMask::isIncreasingOrder() const {
  if (Indices.empty())
    return true;
  int LastIdx = Indices.front();
  for (int Idx : drop_begin(Indices)) {
    if (Idx <= LastIdx)
      return false;
    LastIdx = Idx;
  }
  return true;
}

bool ShuffleMask::operator==(const ShuffleMask &Other) const {
  return equal(Indices, Other.Indices);
}

#ifndef NDEBUG
void ShuffleMask::dump(raw_ostream &OS) const {
  for (auto [Lane, ShuffleIdx] : enumerate(Indices)) {
    if (Lane != 0)
      OS << ", ";
    OS << ShuffleIdx;
  }
}

void ShuffleMask::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void ShuffleMask::verify() const {
  auto NumLanes = (int)Indices.size();
  for (auto Idx : Indices)
    assert(Idx < NumLanes && "Bad index!");
}
#endif // NDEBUG

void SBVecContext::SchedulerDeleter::operator()(Scheduler *Ptr) const {
  delete Ptr;
}

void SBVecContext::quickFlush() {
  InQuickFlush = true;
  SchedForSBBB.clear();

  RemoveInstrCallbacks.clear();
  InsertInstrCallbacks.clear();
  MoveInstrCallbacks.clear();

  RemoveInstrCallbacksBB.clear();
  InsertInstrCallbacksBB.clear();
  MoveInstrCallbacksBB.clear();

  Sched = nullptr;
  LLVMValueToSBValueMap.clear();
  MultiInstrMap.clear();
  InQuickFlush = false;
}

Scheduler *SBVecContext::getScheduler(SBBasicBlock *SBBB) const {
  auto It = SchedForSBBB.find(SBBB);
  return It != SchedForSBBB.end() ? It->second.get() : nullptr;
}

const DependencyGraph &SBVecContext::getDAG(SBBasicBlock *SBBB) const {
  return getScheduler(SBBB)->getDAG();
}

void SBVecContext::createdSBBasicBlock(SBBasicBlock &BB) {
  // Create a scheduler object for this particular BB.
  // Note: This should be done *after* we populate the BB.
  auto Pair = SchedForSBBB.try_emplace(
      &BB, std::unique_ptr<Scheduler, SchedulerDeleter>(
               new Scheduler(BB, AA, *this)));
  (void)Pair;
  assert(Pair.second && "Creating a scheduler for SBBB for the second time!");
}

#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
void SBVecContext::afterMoveInstrHook(SBBasicBlock &BB) {
  if (!getTracker().inRevert() && getScheduler(&BB) != nullptr)
    getScheduler(&BB)->getDAG().verify();
}
#endif

// Pack
SBPackInstruction *
SBVecContext::createSBPackInstruction(const DmpVector<Value *> &PackInstrs) {
  assert(all_of(PackInstrs,
                [](Value *V) {
                  return isa<InsertElementInst>(V) ||
                         isa<ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
  auto NewPtr = std::unique_ptr<SBPackInstruction>(
      new SBPackInstruction(PackInstrs, *this));
  return cast<SBPackInstruction>(registerSBValue(std::move(NewPtr)));
}

// Unpack
SBUnpackInstruction *
SBVecContext::getSBUnpackInstruction(ExtractElementInst *ExtractI) const {
  auto *SBV = getSBValue(ExtractI);
  return SBV ? cast<SBUnpackInstruction>(SBV) : nullptr;
}

SBUnpackInstruction *
SBVecContext::createSBUnpackInstruction(ExtractElementInst *ExtractI) {
  assert(getSBUnpackInstruction(ExtractI) == nullptr && "Already exists!");
  auto *Op = getSBValue(ExtractI->getVectorOperand());
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  Value *Idx = ExtractI->getIndexOperand();
  assert(isa<ConstantInt>(Idx) && "Can only handle constant int index!");
  auto Lane = cast<ConstantInt>(Idx)->getSExtValue();
  auto NewPtr = std::unique_ptr<SBUnpackInstruction>(
      new SBUnpackInstruction(ExtractI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<SBUnpackInstruction>(registerSBValue(std::move(NewPtr)));
}

SBUnpackInstruction *
SBVecContext::getOrCreateSBUnpackInstruction(ExtractElementInst *ExtractI) {
  if (auto *Unpack = getSBUnpackInstruction(ExtractI))
    return Unpack;
  return createSBUnpackInstruction(ExtractI);
}

SBUnpackInstruction *
SBVecContext::getSBUnpackInstruction(ShuffleVectorInst *ShuffleI) const {
  auto *SBV = getSBValue(ShuffleI);
  return SBV ? cast<SBUnpackInstruction>(SBV) : nullptr;
}

SBUnpackInstruction *
SBVecContext::createSBUnpackInstruction(ShuffleVectorInst *ShuffleI) {
  assert(getSBUnpackInstruction(ShuffleI) == nullptr && "Already exists!");
  auto *Op = getSBValue(ShuffleI->getOperand(1));
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  auto Lane = SBUnpackInstruction::getShuffleLane(ShuffleI);
  auto NewPtr = std::unique_ptr<SBUnpackInstruction>(
      new SBUnpackInstruction(ShuffleI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<SBUnpackInstruction>(registerSBValue(std::move(NewPtr)));
}

SBUnpackInstruction *
SBVecContext::getOrCreateSBUnpackInstruction(ShuffleVectorInst *ShuffleI) {
  if (auto *Unpack = getSBUnpackInstruction(ShuffleI))
    return Unpack;
  return createSBUnpackInstruction(ShuffleI);
}

// Shuffle
SBShuffleInstruction *
SBVecContext::getSBShuffleInstruction(ShuffleVectorInst *ShuffleI) const {
  return cast_or_null<SBShuffleInstruction>(getSBValue(ShuffleI));
}

SBShuffleInstruction *
SBVecContext::createSBShuffleInstruction(ShuffleVectorInst *ShuffleI) {
  assert(getSBShuffleInstruction(ShuffleI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBShuffleInstruction>(
      new SBShuffleInstruction(ShuffleI, *this));
  return cast<SBShuffleInstruction>(registerSBValue(std::move(NewPtr)));
}

SBShuffleInstruction *
SBVecContext::getOrCreateSBShuffleInstruction(ShuffleVectorInst *ShuffleI) {
  if (auto *Shuffle = getSBShuffleInstruction(ShuffleI))
    return Shuffle;
  return createSBShuffleInstruction(ShuffleI);
}

SBValue *
SBVecContext::createSBValueFromExtractElement(ExtractElementInst *ExtractI,
                                              int Depth) {
  // Check that all indices are ConstantInts.
  if (!isa<ConstantInt>(ExtractI->getIndexOperand()))
    return getOrCreateSBExtractElementInstruction(ExtractI);
  getOrCreateSBValueInternal(ExtractI->getVectorOperand(), Depth + 1);
  // ExtractI could be a member of either Unpack or Pack from vectors.
  if (SBValue *Extract = getSBValue(ExtractI))
    return Extract;
  return createSBUnpackInstruction(ExtractI);
}

#ifndef NDEBUG
static std::optional<int> getPoisonVectorLanes(Value *V) {
  if (!isa<PoisonValue>(V))
    return std::nullopt;
  auto *Ty = V->getType();
  if (!isa<FixedVectorType>(Ty))
    return std::nullopt;
  return cast<FixedVectorType>(Ty)->getNumElements();
}
#endif

/// Checks if \p PackBottomInsert contains the last insert of an insert/extract
/// packing pattern, and if so returns the packed values in order, or an empty
/// vector, along with their operands.
// The simplest pattern is:
//   %i0 = insert poison, %v0, 0
//   %i1 = insert %i0,    %v1, 1
//   %i2 = insert %i1,    %v2, 2
//   ...
//   %iN = insert %iN-1,  %vN, N  ; <-- This is `PackBottomInsert`
static std::pair<DmpVector<Value *>, DmpVector<Value *>>
matchPackAndGetPackInstrs(InsertElementInst *PackBottomInsert) {
  // All instructions in the pattern must be in canonical form and back-to-back.
  // The canonical form rules:
  // 1. The bottom instruction must be an insert to the last lane.
  // 2. The rest of the inserts must have constant indexes in increasing order.
  // 3. If the pack pattern includes vector operands, the extracts from the
  //    vector are also part of the pattern. They must:
  //    a. have constant indexes in order, starting from 0 at the bottom.
  //    b. be positioned right before the insert instruction that uses it.
  //    c. have a single user: the pattern's insert.
  //    d. All extracts in the group extract from the same vector.
  // 4. No gaps (i.e. other instrs) between the instructions that form the pack.
  // 5. The topmost insert's vector value operand must be Poison.
  // 6. All pattern instrs (except the bottom most one) must have a single-use
  //    and it must be the next instruction after it and a member of the pattern
  // 7. Inserts should cover all lanes of the vector.
  // If the pattern is not in a canonical form, matching will fail.
  //
  // Example 1:
  //  %Pack0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  //  %Pack1 = insertelement <2 x i32> %Pack0, i32 %v1, i64 1
  //
  // Example 2:
  //  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  //  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  //  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  //  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  //  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  //
  // Example 3:
  //  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  //  %PackIns0 = insertelement <2 x i32> poison, i32 %PackExtr0, i32 0
  //  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  //  %PackIns1 = insertelement <2 x i32> %PackIns0, i32 %PackExtr1, i32 1
  //
  // Example 4:
  //  %Pack = insertelement <2 x i32> poison, i32 %v, i64 1
  int ExpectedExtractLane = 0;
  DmpVector<Value *> PackInstrs;
  int TotalLanes = SBVecUtils::getNumLanes(PackBottomInsert);
  int ExpectedInsertLane = TotalLanes - 1;
  InsertElementInst *LastInsert = nullptr;
  ExtractElementInst *LastExtractInGroup = nullptr;
  // Walk the chain bottom-up collecting the matched instrs into `PackInstrs`
  for (Instruction *CurrI = PackBottomInsert;
       CurrI != nullptr && (ExpectedInsertLane >= 0 || ExpectedExtractLane > 0);
       ExpectedInsertLane -= (isa<InsertElementInst>(CurrI) ? 1 : 0),
                   CurrI = CurrI->getPrevNode()) {
    // Checks for both Insert and Extract:
    bool IsAtBottom = PackInstrs.empty();
    if (IsAtBottom) {
      // The bottom instr must be an Insert (Rule 1).
      if (!isa<InsertElementInst>(CurrI))
        return {};
    } else {
      // If not the last instruction and it does not have a single user then
      // discard it (Rule 6).
      if (!CurrI->hasOneUse())
        return {};
      // Discard user is not the previous instr in the pattern (Rule 6).
      User *SingleUser = *CurrI->users().begin();
      if (SingleUser != LastInsert)
        return {};
      assert(isa<InsertElementInst>(SingleUser) && "The user must be an Inset");
    }

    // We expect a constant lane that matches ExpectedInsertLane (Rules 1,2,7).
    if (auto InsertI = dyn_cast<InsertElementInst>(CurrI)) {
      auto LaneOpt = SBVecUtils::getConstantIndex(CurrI);
      if (!LaneOpt || *LaneOpt != ExpectedInsertLane)
        return {};
      assert((IsAtBottom ||
              cast<InsertElementInst>(*CurrI->users().begin())->getOperand(0) ==
                  CurrI) &&
             "CurrI must be the user's vector operand!");
      LastInsert = InsertI;
    } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(CurrI)) {
      // The extract's lane must be constant (Rule 3a).
      auto ExtractLaneOpt = SBVecUtils::getExtractLane(ExtractI);
      if (!ExtractLaneOpt)
        return {};
      int ExtractLanes =
          cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
              ->getNumElements();
      bool IsFirstExtractInGroup = ExpectedExtractLane == 0;
      ExpectedExtractLane =
          IsFirstExtractInGroup ? ExtractLanes - 1 : ExpectedExtractLane - 1;
      assert(ExpectedExtractLane >= 0 && "Bad calculation of expected lane");
      // The extract's lane must be in order (Rule 3a).
      if (*ExtractLaneOpt != ExpectedExtractLane)
        return {};
      // Make sure all extracts groups extract from the same vector (Rule 3d).
      if (LastExtractInGroup != nullptr &&
          ExtractI->getVectorOperand() !=
              LastExtractInGroup->getVectorOperand())
        return {};
      assert(!IsAtBottom && "An extract should not be at the pattern bottom!");
      assert(cast<InsertElementInst>(*CurrI->users().begin())->getOperand(1) ==
                 CurrI &&
             "CurrI must be the user's scalar operand!");
      LastExtractInGroup = ExpectedExtractLane > 0 ? ExtractI : nullptr;
    } else {
      // No non-pattern instructions allowed (Rule 4).
      return {};
    }

    // Collect CurrI, it looks good.
    PackInstrs.push_back(CurrI);
  }
  // Missing insert.
  if (ExpectedInsertLane != -1)
    return {};
  // Missing extract.
  if (ExpectedExtractLane != 0)
    return {};
#ifndef NDEBUG
  Instruction *TopI = cast<Instruction>(PackInstrs.back());
  assert((getPoisonVectorLanes(TopI->getOperand(0)) ||
          *SBVecUtils::getConstantIndex(TopI) == 0) &&
         "TopI is pointing to the wrong instruction!");
#endif // NDEBUG
  // If this is the top-most insert, its operand must be poison (Rule 5).
  if (!isa<PoisonValue>(LastInsert->getOperand(0)))
    return {};

  // Collect operands.
  DmpVector<Value *> Operands;
  for (unsigned Idx = 0, E = PackInstrs.size(); Idx < E; ++Idx) {
    Value *V = PackInstrs[Idx];
    if (isa<InsertElementInst>(V)) {
      Operands.push_back(cast<InsertElementInst>(V)->getOperand(1));
      continue;
    }
    assert(isa<ExtractElementInst>(V) && "Expected Extract!");
    auto *Extract = cast<ExtractElementInst>(V);
    Value *Op = Extract->getVectorOperand();
    Operands.push_back(Op);
    // Now we need to skip all Inserts and Extracts reading `Extract`.
    unsigned Skip = SBVecUtils::getNumLanes(Op) * 2 - 1;
    Idx += Skip;
  }
  return {PackInstrs, Operands};
}

SBValue *
SBVecContext::createSBValueFromInsertElement(InsertElementInst *InsertI,
                                             int Depth) {
  if (auto *Insert = getSBValue(InsertI))
    return Insert;
  // Check if this is the bottom of an InsertElementInst packing pattern.
  auto [PackInstrs, PackOperands] = matchPackAndGetPackInstrs(InsertI);
  if (PackInstrs.empty())
    return getOrCreateSBInsertElementInstruction(InsertI);
  // Else create a new SBPackInstruction.
  return createSBPackInstruction(PackInstrs);
}

SBValue *
SBVecContext::createSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI,
                                             int Depth) {
  // Check that we are only using the first operand.
  // TODO: Is a poison/undef operand always 2nd operand when canonicalized?
  if (SBUnpackInstruction::isUnpack(ShuffleI)) {
    getOrCreateSBValueInternal(ShuffleI->getOperand(0), Depth + 1);
    getOrCreateSBValueInternal(ShuffleI->getOperand(1), Depth + 1);
    return getOrCreateSBUnpackInstruction(ShuffleI);
  }
  if (SBShuffleInstruction::isShuffle(ShuffleI))
    return getOrCreateSBShuffleInstruction(ShuffleI);
  return getOrCreateSBShuffleVectorInstruction(ShuffleI);
}

PackInstrBundle::PackInstrBundle(const DmpVector<Value *> &PackInstrsBndl) {
  PackInstrs.reserve(PackInstrsBndl.size());
  copy(PackInstrsBndl.instrRange(), std::back_inserter(PackInstrs));
  // Sort in reverse program order.
  sort(PackInstrs, [](auto *I1, auto *I2) { return I2->comesBefore(I1); });
}

SBValue *SBPackInstruction::create(const DmpVector<SBValue *> &PackOps,
                                   SBBasicBlock::iterator WhereIt,
                                   SBBasicBlock *WhereBB,
                                   SBVecContext &SBCtxt) {
  std::variant<DmpVector<Value *>, Constant *> BorC =
      SBPackInstruction::createIR(PackOps, WhereIt, WhereBB);
  // CreateIR packed constants which resulted in a single folded Constant.
  if (Constant **CPtr = std::get_if<Constant *>(&BorC))
    return SBCtxt.getOrCreateSBValue(*CPtr);

  for (Value *V : std::get<DmpVector<Value *>>(BorC))
    SBCtxt.createMissingConstantOperands(V);
  // If we created Instructions then create and return a Pack.
  SBValue *NewSBV =
      SBCtxt.createSBPackInstruction(std::get<DmpVector<Value *>>(BorC));
  return NewSBV;
}

SBValue *SBPackInstruction::create(const DmpVector<SBValue *> &PackOps,
                                   SBInstruction *InsertBefore,
                                   SBVecContext &SBCtxt) {
  return create(PackOps, InsertBefore->getIterator(), InsertBefore->getParent(),
                SBCtxt);
}

SBValue *SBPackInstruction::create(const DmpVector<SBValue *> &PackOps,
                                   SBBasicBlock *InsertAtEnd,
                                   SBVecContext &SBCtxt) {
  return create(PackOps, InsertAtEnd->end(), InsertAtEnd, SBCtxt);
}

SBUse SBPackInstruction::getOperandUseInternal(unsigned OperandIdx,
                                               bool Verify) const {
  assert((!Verify || OperandIdx < getNumOperands()) && "Out of bounds!");
  Use &LLVMUse = PackInstrBundle::getBndlOperandUse(OperandIdx);
  return SBUse(&LLVMUse, const_cast<SBPackInstruction *>(this), Ctxt);
}

bool SBPackInstruction::isRealOperandUse(Use &OpUse) const {
  bool IsReal = false;
  bool Found = true;
  doOnOperands([&OpUse, &IsReal, &Found](Use &Use, bool IsRealOp) {
    if (&Use != &OpUse)
      return false; // Don't break
    IsReal = IsRealOp;
    Found = true;
    return true; // Break
  });
  assert(Found && "OpUse not found!");
  return IsReal;
}

void SBPackInstruction::setOperand(unsigned OperandIdx, SBValue *Operand) {
  assert(OperandIdx < getNumOperands() && "Out of bounds!");
  assert(Operand->getType() == SBUser::getOperand(OperandIdx)->getType() &&
         "Operand of wrong type!");
  Value *NewOp = ValueAttorney::getValue(Operand);
  unsigned RealOpIdx = 0;
  doOnOperands([NewOp, OperandIdx, &RealOpIdx, this](Use &Use, bool IsRealOp) {
    if (RealOpIdx == OperandIdx) {
      auto &Tracker = getTracker();
      if (Tracker.tracking())
        Tracker.track(std::make_unique<SetOperand>(this, OperandIdx, Tracker));
      Use.set(NewOp);
    }
    if (IsRealOp)
      ++RealOpIdx;
    // Break once we are done updating the operands.
    if (RealOpIdx > OperandIdx)
      return true; // Break
    return false;  // Don't break
  });
}

InsertElementInst *
SBPackInstruction::getBottomInsert(const DmpVector<Value *> &Instrs) const {
  // Get the bottom insert by removing the vector operands from the set until we
  // have only the bottom left.
  DenseSet<Value *> AllPackInstrs(Instrs.begin(), Instrs.end());
  for (auto *PackI : Instrs) {
    assert((isa<InsertElementInst>(PackI) || isa<ExtractElementInst>(PackI)) &&
           "Expected Insert or Extract");
    AllPackInstrs.erase(cast<Instruction>(PackI)->getOperand(0));
    AllPackInstrs.erase(cast<Instruction>(PackI)->getOperand(1));
  }
  assert(AllPackInstrs.size() == 1 && "Unexpected pack instruction structure");
  return cast<InsertElementInst>(*AllPackInstrs.begin());
}

bool SBPackInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Pack;
}

std::variant<DmpVector<Value *>, Constant *>
SBPackInstruction::createIR(const DmpVector<SBValue *> &ToPack,
                            SBBasicBlock::iterator WhereIt,
                            SBBasicBlock *WhereBB) {
  // A Pack should be placed after the latest packed value.
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  auto *LLVMBB = SBBasicBlockAttorney::getBB(WhereBB);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostIRInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(LLVMBB);

  Type *ScalarTy = SBVecUtils::getCommonScalarTypeFast(ToPack);
  unsigned Lanes = SBVecUtils::getNumLanes(ToPack);
  auto *VecTy = SBVecUtils::getWideType(ScalarTy, Lanes);

  // Create a series of pack instructions.
  DmpVector<Value *> AllPackInstrs;
  Value *LastInsert = PoisonValue::get(VecTy);

  auto Collect = [&AllPackInstrs](Value *NewV) {
    assert(isa<Instruction>(NewV) && "Expected instruction!");
    auto *I = cast<Instruction>(NewV);
    AllPackInstrs.push_back(I);
  };

  unsigned InsertIdx = 0;
  for (SBValue *SBV : ToPack) {
    Value *Elm = ValueAttorney::getValue(SBV);
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            LLVMIRBuilder.CreateExtractElement(Elm, ExtrLane, "XPack");
        if (auto *ExtrC = dyn_cast<Constant>(ExtrI))
          WhereBB->getContext().getOrCreateSBConstant(ExtrC);
        else
          Collect(ExtrI);
        // This may also return a Constant if ExtrI is a Constant.
        LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, ExtrI,
                                                       InsertIdx++, "Pack");
        if (auto *C = dyn_cast<Constant>(LastInsert)) {
          if (InsertIdx == Lanes)
            return C;
          WhereBB->getContext().getOrCreateSBValue(C);
        } else
          Collect(LastInsert);
      }
    } else {
      // This may be folded into a Constant if LastInsert is a Constant. In that
      // case we only collect the last constant.
      LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, Elm,
                                                     InsertIdx++, "Pack");
      if (auto *C = dyn_cast<Constant>(LastInsert)) {
        if (InsertIdx == Lanes)
          return C;
        WhereBB->getContext().getOrCreateSBValue(C);
      } else
        Collect(LastInsert);
    }
  }
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  WhereBB->verifyLLVMIR();
#endif
  return AllPackInstrs;
}

Use &PackInstrBundle::getExternalFacingOperandUse(
    InsertElementInst *InsertI) const {
  // Get the Insert's edge and check if its source is a Pack Extract. If it is,
  // then don't use the Insert's edge, but rather the Extract's edge.
  Use &OpUse = InsertI->getOperandUse(1);
  Value *Op = OpUse.get();
  if (!isa<ExtractElementInst>(Op) ||
      find(PackInstrs, cast<ExtractElementInst>(Op)) == PackInstrs.end())
    return OpUse;
  // This is an extract used in the pack-from-vector pattern.
  return cast<ExtractElementInst>(Op)->getOperandUse(0);
}

InsertElementInst *PackInstrBundle::getInsertAtLane(int Lane) const {
  auto It = find_if(PackInstrs, [Lane](Value *V) {
    return isa<InsertElementInst>(V) &&
           *SBVecUtils::getInsertLane(cast<InsertElementInst>(V)) == Lane;
  });
  return It != PackInstrs.end() ? cast<InsertElementInst>(*It) : nullptr;
}

InsertElementInst *PackInstrBundle::getTopInsert() const {
  auto Range = reverse(PackInstrs);
  auto It = find_if(Range, [](auto *I) { return isa<InsertElementInst>(I); });
  assert(It != Range.end() && "Not found!");
  return cast<InsertElementInst>(*It);
}

InsertElementInst *PackInstrBundle::getBotInsert() const {
  auto It =
      find_if(PackInstrs, [](auto *I) { return isa<InsertElementInst>(I); });
  assert(It != PackInstrs.end() && "Not found!");
  return cast<InsertElementInst>(*It);
}

static bool isSingleUseEdge(Use &ExtFacingUse) {
  return !isa<ExtractElementInst>(ExtFacingUse.getUser());
}

static bool isLastOfMultiUseEdge(Use &ExtFacingUse) {
  User *U = ExtFacingUse.getUser();
  assert(isa<ExtractElementInst>(U) &&
         "A multi-Use edge must have an Extract operand!");
  // If the user is not an extract, then this is a single-Use edge.
  auto *ExtractI = cast<ExtractElementInst>(U);
  auto ExtrIdx = *SBVecUtils::getExtractLane(ExtractI);
  int Lanes = cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
                  ->getNumElements();
  return ExtrIdx == Lanes - 1;
}

void PackInstrBundle::doOnOperands(
    function_ref<bool(Use &, bool)> DoOnOpFn) const {
  // Constant operands may be folded into the poison vector, or poison operands
  // can also be folded into a single vector poison value.
  auto *TopInsertI = getTopInsert();
  auto *PoisonVal = cast<Constant>(TopInsertI->getOperand(0));
  auto *PoisonConstantVec = dyn_cast<ConstantVector>(PoisonVal);
  auto Lanes = cast<FixedVectorType>(PoisonVal->getType())->getNumElements();
  assert((PoisonConstantVec == nullptr ||
          PoisonConstantVec->getNumOperands() == Lanes) &&
         "Bad Lanes or PoisonConstantVec!");
  for (auto Lane : seq<unsigned>(0, Lanes)) {
    InsertElementInst *InsertI = getInsertAtLane(Lane);
    // A missing insert means that the operand was folded into the poison vector
    Use *OpUsePtr = nullptr;
    if (InsertI != nullptr) {
      OpUsePtr = &getExternalFacingOperandUse(InsertI);
    } else if (PoisonConstantVec != nullptr) {
      OpUsePtr = &PoisonConstantVec->getOperandUse(Lane);
    } else {
      auto *TopInsertI = cast<InsertElementInst>(PackInstrs.front());
      OpUsePtr = &TopInsertI->getOperandUse(0);
    }
    Use &OpUse = *OpUsePtr;
    // insert %val0  0  <- Single edge
    // insert %extr0 1
    // insert %extr0 2  <- Last of 2-wide Multi-edge
    // insert %val1  3  <- Single edge
    bool IsOnlyOrLastUse =
        isSingleUseEdge(OpUse) || isLastOfMultiUseEdge(OpUse);
    bool Break = DoOnOpFn(OpUse, IsOnlyOrLastUse);
    if (Break)
      break;
  }
}

unsigned SBPackInstruction::getOperandUseIdx(const Use &UseToMatch) const {
#ifndef NDEBUG
  verifyUserOfLLVMUse(UseToMatch);
#endif
  unsigned RealOpIdx = 0;
  bool Found = false;
  doOnOperands([&UseToMatch, &RealOpIdx, &Found](Use &Use, bool IsRealOp) {
    if (&Use == &UseToMatch) {
      Found = true;
      return true; // Ask to break
    }
    if (IsRealOp)
      ++RealOpIdx;
    return false; // Don't break
  });
  assert(Found && "Use not found in external facing operands!");
  return RealOpIdx;
}

Use &PackInstrBundle::getBndlOperandUse(unsigned OperandIdx) const {
  unsigned RealOpIdx = 0;
  // Special case for op_end().
  if (OperandIdx == getNumOperands())
    return *getBotInsert()->op_end();

  Use *ReturnUse = nullptr;
  doOnOperands([&RealOpIdx, &ReturnUse, OperandIdx](Use &Use, bool IsRealOp) {
    if (IsRealOp) {
      if (RealOpIdx == OperandIdx) {
        ReturnUse = &Use;
        return true; // Ask to break
      }
      ++RealOpIdx;
    }
    return false; // Don't break
  });
  assert(ReturnUse != nullptr && "Expected non-null operand!");
  return *ReturnUse;
}

unsigned PackInstrBundle::getNumOperands() const {
  // Not breaking for any operand will give us the total number of operands.
  unsigned RealOpCnt = 0;
  doOnOperands([&RealOpCnt](Use &Use, bool IsRealOp) {
    if (IsRealOp)
      ++RealOpCnt;
    return false; // Don't break
  });
  return RealOpCnt;
}

#ifndef NDEBUG
void PackInstrBundle::verifyInstrBundle() const {
  // Make sure that the consecutive Extracts that make up the
  // pack-from-vector pattern have the same operand. This could break during
  // a SBPackInstruction::setOperand() operation.
  ExtractElementInst *LastExtractI = nullptr;
  doOnOperands([&LastExtractI](Use &Use, bool IsRealOp) -> bool {
    if (IsRealOp && LastExtractI != nullptr) {
      // We expect an extract that extracts from the same vector as
      // LastExtractI, but the next lane.
      assert(isa<ExtractElementInst>(Use.getUser()) && "Expect Extract");
      auto *ExtractI = cast<ExtractElementInst>(Use.getUser());
      assert(Use.get() == ExtractI->getVectorOperand() && "Sanity check");
      // Skip <1 x type>
      if (cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
              ->getNumElements() != 1u) {
        assert(ExtractI->getVectorOperand() ==
                   LastExtractI->getVectorOperand() &&
               "Most likely setOperand() did not update all Extracts!");
        assert(ExtractI->getIndexOperand() != LastExtractI->getIndexOperand() &&
               "Expected different indices");
      }
      LastExtractI = nullptr;
    } else {
      LastExtractI = dyn_cast<ExtractElementInst>(Use.getUser());
    }
    return false;
  });
}
#endif

SBPackInstruction::SBPackInstruction(const DmpVector<Value *> &Instrs,
                                     SBContext &SBCtxt)
    : PackInstrBundle(Instrs), SBInstruction(ClassID::Pack, Opcode::Pack,
                                             getBottomInsert(Instrs), SBCtxt) {
  assert(all_of(PackInstrs,
                [](Value *V) {
                  return isa<InsertElementInst>(V) ||
                         isa<ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, PackInstrs.size()))
    assert(this->PackInstrs[Idx]->comesBefore(this->PackInstrs[Idx - 1]) &&
           "Expecte reverse program order!");
  assert(all_of(drop_begin(this->PackInstrs),
                [this](auto *I) {
                  return I->comesBefore(cast<Instruction>(Val));
                }) &&
         "Val should be the bottom instruction!");
#endif
}

DmpVector<Instruction *> SBPackInstruction::getLLVMInstrs() const {
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, PackInstrs.size())) {
    auto *I1 = PackInstrs[Idx - 1];
    auto *I2 = PackInstrs[Idx];
    assert(((!I1->getParent() || I2->getParent()) || I2->comesBefore(I1)) &&
           "Expected reverse program order!");
  }
#endif
  return PackInstrs;
}

DmpVector<Instruction *>
SBPackInstruction::getLLVMInstrsWithExternalOperands() const {
  SmallVector<Instruction *> IRInstrs;
  for (Instruction *I : PackInstrs) {
    if (auto *InsertI = dyn_cast<InsertElementInst>(I)) {
      // If this is an internal insert, it must have an Extract operand, which
      // is the external facing IR instruction.
      if (auto *ExtractOp =
              dyn_cast<ExtractElementInst>(InsertI->getOperand(1))) {
        if (find(PackInstrs, ExtractOp) != PackInstrs.end())
          // ExtractOp is the out-facing instruction, not the insert.
          IRInstrs.push_back(ExtractOp);
      } else {
        // This is an external-facing Insert.
        IRInstrs.push_back(InsertI);
      }
    }
  }
  return IRInstrs;
}

unsigned SBPackInstruction::getUseOperandNo(const SBUse &SBUse) const {
  unsigned OpNo = 0;
  llvm::Use *UseToMatch = SBUse.LLVMUse;
  doOnOperands([&OpNo, UseToMatch](Use &LLVMUse, bool IsRealOp) -> bool {
    if (&LLVMUse == UseToMatch)
      return true; // break
    if (IsRealOp)
      ++OpNo;
    return false; // don't break
  });
  return OpNo;
}

SBUser::op_iterator SBPackInstruction::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

SBUser::op_iterator SBPackInstruction::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

SBUser::const_op_iterator SBPackInstruction::op_begin() const {
  return const_cast<SBPackInstruction *>(this)->op_begin();
}

SBUser::const_op_iterator SBPackInstruction::op_end() const {
  return const_cast<SBPackInstruction *>(this)->op_end();
}

void SBPackInstruction::detachExtras() {
  auto *PackV = ValueAttorney::getValue(this);
  for (auto *PI : getPackInstrs())
    if (PI != PackV) // Skip the bottom value, gets detached later
      Ctxt.detachValue(PI);
}

#ifndef NDEBUG
void SBPackInstruction::verify() const {
  if (any_of(operands(), [](SBValue *Op) { return SBVecUtils::isVector(Op); }))
    assert((isa<FixedVectorType>(getOperand(0)->getType()) ||
            getNumOperands() <
                cast<FixedVectorType>(SBVecUtils::getExpectedType(this))
                    ->getNumElements()) &&
           "This has vector operands. We expect fewer operands than lanes");
  verifyInstrBundle();
}

void SBPackInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  OS.indent(2) << "PackInstrs:\n";
  for (auto *I : PackInstrs)
    OS.indent(2) << *I << "\n";
  dumpCommonFooter(OS);
}
void SBPackInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}

void SBPackInstruction::dump(raw_ostream &OS) const {
  // Sort pack instructions in program order.
  auto SortedInstrs = getLLVMInstrs();
  if (all_of(SortedInstrs, [](Instruction *I) { return I->getParent(); }))
    sort(SortedInstrs, [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  else
    OS << "** Error: Not all IR Instrs have a parent! **";

  auto ExtFacing = getLLVMInstrsWithExternalOperands();
  unsigned NumOperands = getNumOperands();

  for (auto [Idx, I] : enumerate(SortedInstrs)) {
    OS << *I;
    dumpCommonSuffix(OS);
    // Print the lane.
    bool IsExt = find(ExtFacing, I) != ExtFacing.end();
    if (IsExt) {
      Use *OpUse = nullptr;
      if (auto *InsertI = dyn_cast<InsertElementInst>(I)) {
        OpUse = &InsertI->getOperandUse(1);
      } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(I)) {
        OpUse = &ExtractI->getOperandUse(0);
      }
      OS << " OpIdx=";
      if (OpUse == nullptr)
        OS << "** ERROR: Can't get OpIdx! **";
      else
        OS << getOperandUseIdx(*OpUse) << "/" << NumOperands - 1 << " ";
    }
    OS << (Idx + 1 != SortedInstrs.size() ? "\n" : "");
  }
}
void SBPackInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

SBShuffleInstruction::SBShuffleInstruction(SBValue *Op, const ShuffleMask &Mask,
                                           SBBasicBlock::iterator WhereIt,
                                           SBBasicBlock *WhereBB)
    : SBShuffleInstruction(createIR(Op, Mask, WhereIt, WhereBB),
                           WhereBB->getContext()) {
  Ctxt.createMissingConstantOperands(Val);
  assert(Val != nullptr && "Shuffle was folded!");
}

SBShuffleInstruction *
SBShuffleInstruction::create(SBValue *Op, ShuffleMask &Mask,
                             SBBasicBlock::iterator WhereIt,
                             SBBasicBlock *WhereBB, SBVecContext &SBCtxt) {
  auto NewPtr = std::unique_ptr<SBShuffleInstruction>(
      new SBShuffleInstruction(Op, Mask, WhereIt, WhereBB));
  return cast<SBShuffleInstruction>(SBCtxt.registerSBValue(std::move(NewPtr)));
}

SBShuffleInstruction *SBShuffleInstruction::create(SBValue *Op,
                                                   ShuffleMask &Mask,
                                                   SBInstruction *InsertBefore,
                                                   SBVecContext &SBCtxt) {
  return SBShuffleInstruction::create(Op, Mask, InsertBefore->getIterator(),
                                      InsertBefore->getParent(), SBCtxt);
}

SBShuffleInstruction *SBShuffleInstruction::create(SBValue *Op,
                                                   ShuffleMask &Mask,

                                                   SBBasicBlock *InsertAtEnd,
                                                   SBVecContext &SBCtxt) {
  return SBShuffleInstruction::create(Op, Mask, InsertAtEnd->end(), InsertAtEnd,
                                      SBCtxt);
}

SBUser::op_iterator SBShuffleInstruction::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}
SBUser::op_iterator SBShuffleInstruction::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}
SBUser::const_op_iterator SBShuffleInstruction::op_begin() const {
  return const_cast<SBShuffleInstruction *>(this)->op_begin();
}
SBUser::const_op_iterator SBShuffleInstruction::op_end() const {
  return const_cast<SBShuffleInstruction *>(this)->op_end();
}

void SBShuffleInstruction::setOperand(unsigned OperandIdx, SBValue *Operand) {
  assert(OperandIdx == 0 && "A SBShuffleInstruction has exactly 1 operand!");
  SBUser::setOperand(OperandIdx, Operand);
}

bool SBShuffleInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Shuffle;
}

ShuffleVectorInst *
SBShuffleInstruction::createIR(SBValue *Op, const ShuffleMask &Mask,
                               SBBasicBlock::iterator WhereIt,
                               SBBasicBlock *WhereBB) {
  Value *Vec = ValueAttorney::getValue(Op);
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  auto *LLVMBB = SBBasicBlockAttorney::getBB(WhereBB);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostIRInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(LLVMBB);
  auto *Shuffle = cast_or_null<ShuffleVectorInst>(
      LLVMIRBuilder.CreateShuffleVector(Vec, Mask, "Shuf"));
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  WhereBB->verifyLLVMIR();
#endif
  return Shuffle;
}

#ifndef NDEBUG
void SBShuffleInstruction::verify() const {
  assert(getMask().size() == SBVecUtils::getNumLanes(this) &&
         "Expected same number of indices as lanes.");
  assert((int)SBVecUtils::getNumLanes(this) ==
             SBVecUtils::getNumLanes(getOperand(0)->getType()) &&
         "A SBShuffle should not unpack, it should only reorder lanes!");
  getMask().verify();
  assert(getNumOperands() == 1 && "Expected a single operand");
}
void SBShuffleInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "  " << getMask() << "\n";
  dumpCommonFooter(OS);
}
void SBShuffleInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBShuffleInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " ; " << getMask();
}
void SBShuffleInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBUnpackInstruction::SBUnpackInstruction(ExtractElementInst *ExtractI,
                                         SBValue *UnpackOp, unsigned UnpackLane,
                                         SBContext &SBCtxt)
    : SBInstruction(ClassID::Unpack, Opcode::Unpack, ExtractI, SBCtxt) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

SBUnpackInstruction::SBUnpackInstruction(ShuffleVectorInst *ShuffleI,
                                         SBValue *UnpackOp, unsigned UnpackLane,
                                         SBContext &SBCtxt)
    : SBInstruction(ClassID::Unpack, Opcode::Unpack, ShuffleI, SBCtxt) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

SBValue *SBUnpackInstruction::create(SBValue *Op, unsigned UnpackLane,
                                     unsigned NumLanesToUnpack,
                                     SBBasicBlock::iterator WhereIt,
                                     SBBasicBlock *WhereBB,
                                     SBVecContext &SBCtxt) {
  Value *V = SBUnpackInstruction::createIR(Op, UnpackLane, NumLanesToUnpack,
                                           WhereIt, WhereBB);
  SBCtxt.createMissingConstantOperands(V);
  auto *NewSBV = SBCtxt.getOrCreateSBValue(V);
  return NewSBV;
}

Value *SBUnpackInstruction::createIR(SBValue *UnpackOp, unsigned Lane,
                                     unsigned Lanes,
                                     SBBasicBlock::iterator WhereIt,
                                     SBBasicBlock *WhereBB) {
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  Value *OpVec = ValueAttorney::getValue(UnpackOp);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostIRInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(SBBasicBlockAttorney::getBB(WhereBB));
  Value *Unpack;
  if (Lanes == 1) {
    // If we are unpacking a scalar, we can use an ExtractElementInst.
    Unpack = LLVMIRBuilder.CreateExtractElement(OpVec, Lane, "Unpack");
  } else {
    // If we are unpacking a vector element, we need to use a Shuffle.
    ShuffleMask::IndicesVecT ShuffleIndices;
    ShuffleIndices.reserve(Lanes);
    for (auto Ln : seq<unsigned>(Lane, Lane + Lanes))
      ShuffleIndices.push_back(Ln);
    ShuffleMask Mask(std::move(ShuffleIndices));
    Unpack = LLVMIRBuilder.CreateShuffleVector(OpVec, Mask, "Unpack");
  }
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  WhereBB->verifyLLVMIR();
#endif
  return Unpack;
}

SBValue *SBUnpackInstruction::create(SBValue *Op, unsigned UnpackLane,
                                     unsigned NumLanesToUnpack,
                                     SBInstruction *InsertBefore,
                                     SBVecContext &SBCtxt) {
  return SBUnpackInstruction::create(Op, UnpackLane, NumLanesToUnpack,
                                     InsertBefore->getIterator(),
                                     InsertBefore->getParent(), SBCtxt);
}

SBValue *SBUnpackInstruction::create(SBValue *Op, unsigned UnpackLane,
                                     unsigned NumLanesToUnpack,
                                     SBBasicBlock *InsertAtEnd,
                                     SBVecContext &SBCtxt) {
  return SBUnpackInstruction::create(Op, UnpackLane, NumLanesToUnpack,
                                     InsertAtEnd->end(), InsertAtEnd, SBCtxt);
}

bool SBUnpackInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Unpack;
}

#ifndef NDEBUG
void SBUnpackInstruction::verify() const {
  // TODO:
}
void SBUnpackInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << " " << "lane:" << getUnpackLane() << "\n";
  dumpCommonFooter(OS);
}
void SBUnpackInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBUnpackInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " lane:" << getUnpackLane();
}
void SBUnpackInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
