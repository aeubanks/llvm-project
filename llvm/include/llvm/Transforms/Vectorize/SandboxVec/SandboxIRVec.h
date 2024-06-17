//===- SandboxIRVec.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines vectorization-specific SandboxIR instructions
//
// SBInstruction -+- SBPackInstruction
//                |
//                +- SBUnpackInstruction
//                |
//                +- SBShuffleInstruction
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIRVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIRVEC_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Transforms/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include <variant>

namespace llvm {

class Scheduler;

class ShuffleMask {
public:
  using IndicesVecT = SmallVector<int, 8>;

private:
  IndicesVecT Indices;

public:
  ShuffleMask(IndicesVecT &&Indices) : Indices(std::move(Indices)) {}
  ShuffleMask(std::initializer_list<int> Indices) : Indices(Indices) {}
  explicit ShuffleMask(ArrayRef<int> Indices) : Indices(Indices) {}
  static ShuffleMask getIdentity(unsigned Lanes) {
    IndicesVecT Vec;
    Vec.reserve(Lanes);
    for (unsigned Idx = 0; Idx != Lanes; ++Idx)
      Vec.push_back(Idx);
    return ShuffleMask(std::move(Vec));
  }
  /// \Returns true if the mask is a perfect identity mask with consecutive
  /// indices, i.e., performs no lane shuffling, like 0,1,2,3...
  bool isIdentity() const;
  bool isInOrder() const;
  bool isIncreasingOrder() const;
  bool operator==(const ShuffleMask &Other) const;
  bool operator!=(const ShuffleMask &Other) const { return !(*this == Other); }
  size_t size() const { return Indices.size(); }
  int operator[](int Idx) const { return Indices[Idx]; }
  operator ArrayRef<int>() const { return Indices; }
  /// \Returns the inverse mask which when applied to the original gives us an
  /// identity mask. This can also be thought of as the horizontally flipped
  /// mask.
  /// For example:
  /// Mask0 (3,0,1,2) represents shuffling from BCDA to ABCD. This returns Mask1
  /// (1,2,3,0) which corresponds to the reverse shuffle from ABCD to BCDA.
  /// If you combine Mask0 and Mask1 you get the identity mask: (0,1,2,3).
  ShuffleMask getInverse() const {
    IndicesVecT NewIndices;
    NewIndices.resize(Indices.size());
    for (auto [Cnt, OrigIdx] : enumerate(Indices))
      NewIndices[OrigIdx] = Cnt;
    return ShuffleMask(std::move(NewIndices));
  }
  /// \Returns a mask that is equivalent of applying \p Other on top of this,
  /// back to back.
  /// For example, given MaskA (0,2,1,3) and MaskB (1,2,3,0), then
  /// MaskA.combine(MaskB) returns (1,3,2,0) and
  /// MaskB.combine(MaskA) returns (2,1,3,0).
  ShuffleMask combine(const ShuffleMask &Other) const {
    IndicesVecT NewIndices;
    NewIndices.resize(Indices.size());
    for (auto [Cnt, OrigIdx] : enumerate(Indices))
      NewIndices[Cnt] = Other[OrigIdx];
    return ShuffleMask(std::move(NewIndices));
  }
  hash_code hash() const {
    return hash_combine_range(Indices.begin(), Indices.end());
  }
  friend hash_code hash_value(const ShuffleMask &M) { return M.hash(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const ShuffleMask &M) {
    M.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  void verify() const;
#endif
};

class SBPackInstruction;
class SBUnpackInstruction;
class SBShuffleInstruction;

class SBVecContext : public SBContext {

  SBValue *createSBValueFromExtractElement(ExtractElementInst *ExtractI,
                                           int Depth) final;

  SBValue *createSBValueFromInsertElement(InsertElementInst *InsertI,
                                          int Depth) final;

  SBValue *createSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI,
                                          int Depth) final;

  /// Creates a scheduler for \p BB.
  void createdSBBasicBlock(SBBasicBlock &BB) final;

#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  /// Verify the DAG and Scheduler.
  void afterMoveInstrHook(SBBasicBlock &BB) final;
#endif

  AliasAnalysis &AA;

  /// This holds a pointer to the scheduler that is currently active.
  /// It helps avoid passing the scheduler as argument to all SandboxIR
  /// modifying functions. It gets set by the Scheduler's constructor.
  Scheduler *Sched = nullptr;
  void setScheduler(Scheduler &NewSched) { Sched = &NewSched; }
  void clearScheduler() { Sched = nullptr; }

  /// Helper deleter that allows us to use std::unique_ptr<Schduler> here,
  /// where it is not defined (due to cyclic header file dependencies).
  struct SchedulerDeleter {
    void operator()(Scheduler *) const;
  };
  DenseMap<SBBasicBlock *, std::unique_ptr<Scheduler, SchedulerDeleter>>
      SchedForSBBB;

  // Pack
  SBPackInstruction *
  createSBPackInstruction(const DmpVector<Value *> &PackInstrs);
  friend class SBPackInstruction; // For createSBPackInstruction()
  // Unpack
  SBUnpackInstruction *
  getSBUnpackInstruction(ExtractElementInst *ExtractI) const;
  SBUnpackInstruction *createSBUnpackInstruction(ExtractElementInst *ExtractI);
  SBUnpackInstruction *
  getOrCreateSBUnpackInstruction(ExtractElementInst *ExtractI);
  SBUnpackInstruction *
  getSBUnpackInstruction(ShuffleVectorInst *ShuffleI) const;
  SBUnpackInstruction *createSBUnpackInstruction(ShuffleVectorInst *ShuffleI);
  SBUnpackInstruction *
  getOrCreateSBUnpackInstruction(ShuffleVectorInst *ShuffleI);
  // Shuffle
  SBShuffleInstruction *
  getSBShuffleInstruction(ShuffleVectorInst *ShuffleI) const;
  SBShuffleInstruction *createSBShuffleInstruction(ShuffleVectorInst *ShuffleI);
  SBShuffleInstruction *
  getOrCreateSBShuffleInstruction(ShuffleVectorInst *ShuffleI);

public:
  SBVecContext(LLVMContext &LLVMCtxt, AliasAnalysis &AA)
      : SBContext(LLVMCtxt), AA(AA) {}

  void quickFlush() final;

  /// Erase the scheduler that corresponds to \p BB upon BB's destruction.
  void destroyingBB(SBBasicBlock &BB) final { SchedForSBBB.erase(&BB); }

  Scheduler *getScheduler(SBBasicBlock *SBBB) const;
  const DependencyGraph &getDAG(SBBasicBlock *SBBB) const;
};

/// The InsertElementInsts that make up a pack.
/// NOTE: We are using a seprate class for it because we need to get it
/// initialized before the call to createIR().
class PackInstrBundle {
protected:
  /// Contains all instructions in the packing pattern, including Inserts into
  /// the final vector and also Extracts from vector operands.
  DmpVector<Instruction *> PackInstrs;

  /// Given an \p InsertI return either its use accessing its immediate
  /// operand, or the use of the extract providing the insert's operand, if
  /// part of a pack-from-vector patter. In any way this returns the use
  /// pointing to the external operand.
  Use &getExternalFacingOperandUse(InsertElementInst *InsertI) const;
  /// \Returns the Pack Insert at \p Lane or nullptr.
  InsertElementInst *getInsertAtLane(int Lane) const;
  /// Iterate over operands and call:
  ///   DoOnOpFn(Use, IsRealOp)
  /// The callback should return true to break out of the operand loop.
  void doOnOperands(function_ref<bool(Use &, bool)> DoOnOpFn) const;
  /// \Returns the operand at \p OperandIdx. This works for both insert-only and
  /// extract/insert pack instructions.
  Use &getBndlOperandUse(unsigned OperandIdx) const;
  /// \Returns the number of operands. This works for both insert-only and
  /// extract/insert pack instructions.
  /// NOTE: This is linear to the number of entries in PackInstrs.
  unsigned getNumOperands() const;
  /// \Returns the top-most InsertElement instruction. This is the one that
  /// inserts into poison.
  InsertElementInst *getTopInsert() const;
  /// \Returns the bottom InsertElement instr.
  InsertElementInst *getBotInsert() const;

public:
  PackInstrBundle() = default;
  PackInstrBundle(const DmpVector<Value *> &PackInstrs);
#ifndef NDEBUG
  void verifyInstrBundle() const;
#endif
};

/// Packs multiple scalar values into a vector.
class SBPackInstruction : public PackInstrBundle, public SBInstruction {
  friend SBContext; // for eraseBundleInstrs().
  /// Use SBBasicBlock::createSBPackInstruction(). Don't call the
  /// constructor directly.
  /// Create a Pack that packs \p ToPack.
  SBPackInstruction(const DmpVector<SBValue *> &ToPack, SBBasicBlock *Parent);
  /// Create a Pack from its LLVM IR values.
  SBPackInstruction(const DmpVector<Value *> &Instrs, SBContext &SBCtxt);
  friend class SBBasicBlock;
  InsertElementInst *getBottomInsert(const DmpVector<Value *> &Instrs) const;
  /// \Returns pack instrs and constants.
  static std::variant<DmpVector<Value *>, Constant *>
  createIR(const DmpVector<SBValue *> &ToPack, SBBasicBlock::iterator WhereIt,
           SBBasicBlock *WhereBB);
  /// \Returns all the IR instructions that make up this Pack in reverse program
  /// order.
  DmpVector<Instruction *> getLLVMInstrs() const final;
  DmpVector<Instruction *> getLLVMInstrsWithExternalOperands() const final;
  friend void SBInstruction::eraseFromParent();

  void detachExtras() final;
  friend class SBVecContext; // For createIR()

protected:
#ifndef NDEBUG // TODO: Use a helper client-attorney class instead.
  // Public for testing
public:
#endif
  unsigned getOperandUseIdx(const Use &UseToMatch) const override;
  SBUse getOperandUseInternal(unsigned OperandIdx, bool Verify) const final;
  bool isRealOperandUse(Use &OpUse) const final;

public:
  static SBValue *create(const DmpVector<SBValue *> &PackOps,
                         SBBasicBlock::iterator WhereIt, SBBasicBlock *WhereBB,
                         SBVecContext &SBCtxt);
  static SBValue *create(const DmpVector<SBValue *> &PackOps,
                         SBInstruction *InsertBefore, SBVecContext &SBCtxt);
  static SBValue *create(const DmpVector<SBValue *> &PackOps,
                         SBBasicBlock *InsertAtEnd, SBVecContext &SBCtxt);
  unsigned getUseOperandNo(const SBUse &Use) const final;
  unsigned getNumOfIRInstrs() const final { return PackInstrs.size(); }
  // Since a Pack corresponds to a sequence of insertelement instructions,
  // the internals of which we don't care too much from the vectorizer's
  // persctive, we need to make sure the operands work as expected.
  SBUser::op_iterator op_begin() final;
  SBUser::op_iterator op_end() final;
  SBUser::const_op_iterator op_begin() const final;
  SBUser::const_op_iterator op_end() const final;

  void setOperand(unsigned OperandIdx, SBValue *Operand) final;
  unsigned getNumOperands() const final {
    return PackInstrBundle::getNumOperands();
  }
  /// \Returns the Inserts that do the packing.
  const auto &getPackInstrs() const { return PackInstrs; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final {
    auto Hash = SBValue::hashCommon();
    for (SBValue *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
#ifndef NDEBUG
  void verify() const final;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
  friend class SBPackInstructionAttorney;
};

/// Client-attorney class for accessing SBPackInstruction's protected members.
class SBPackInstructionAttorney {
public:
  // For tests
  static auto getLLVMInstrsWithExternalOperands(SBPackInstruction *Pack) {
    return Pack->getLLVMInstrsWithExternalOperands();
  }
  // For tests
  static auto getLLVMInstrs(SBPackInstruction *Pack) {
    return Pack->getLLVMInstrs();
  }
};

/// Reorders the lanes of its operand.
class SBShuffleInstruction : public SBInstruction {
private:
  /// Use SBBasicBlock::createSBShuffleInstruction(). Don't call the
  /// constructor directly.
  SBShuffleInstruction(ShuffleVectorInst *ShuffleI, SBContext &Ctxt)
      : SBInstruction(ClassID::Shuffle, Opcode::Shuffle, ShuffleI, Ctxt) {}
  SBShuffleInstruction(SBValue *Op, const ShuffleMask &Mask,
                       SBBasicBlock::iterator WhereIt, SBBasicBlock *WhereBB);
  friend class SBVecContext;
  ShuffleVectorInst *createIR(SBValue *Op, const ShuffleMask &Mask,
                              SBBasicBlock::iterator WhereIt,
                              SBBasicBlock *WhereBB);
  SBUse getOperandUseInternal(unsigned OperandIdx, bool Verify) const final {
    return getOperandUseDefault(OperandIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<Instruction *> getLLVMInstrs() const final {
    return {cast<Instruction>(Val)};
  }
  DmpVector<Instruction *> getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }
  void detachExtras() final {}

public:
  static SBShuffleInstruction *create(SBValue *Op, ShuffleMask &Mask,
                                      SBBasicBlock::iterator WhereIt,
                                      SBBasicBlock *WhereBB,
                                      SBVecContext &SBCtxt);
  static SBShuffleInstruction *create(SBValue *Op, ShuffleMask &Mask,
                                      SBInstruction *InsertBefore,
                                      SBVecContext &SBCtxt);
  static SBShuffleInstruction *create(SBValue *Op, ShuffleMask &Mask,
                                      SBBasicBlock *InsertAtEnd,
                                      SBVecContext &SBCtxt);
  static bool isShuffle(ShuffleVectorInst *ShuffleI) {
    if (!ShuffleI->isSingleSource())
      return false;
    // For now we expect a canonicalized shuffle where the poison value is op 2.
    if (!isa<PoisonValue>(ShuffleI->getOperand(1)))
      return false;
    return true;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
#ifndef NDEBUG
    assert(Use.getUser() == this && "Use does not point to this!");
    assert(Use.LLVMUse == &cast<User>(Val)->getOperandUse(0) &&
           "Use does not match!");
#endif
    return 0u;
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  // Since a Shuffle is a specific single-input SBInstruction,
  // we need to make sure the operands work as expected.
  SBUser::op_iterator op_begin() final;
  SBUser::op_iterator op_end() final;
  SBUser::const_op_iterator op_begin() const final;
  SBUser::const_op_iterator op_end() const final;

  void setOperand(unsigned OperandIdx, SBValue *Operand) final;
  unsigned getNumOperands() const final { return 1; }

  ShuffleMask getMask() const {
    return ShuffleMask(cast<ShuffleVectorInst>(Val)->getShuffleMask());
  }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final { return hash_combine(getMask(), hashCommon()); }
#ifndef NDEBUG
  void verify() const final;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

/// Extracts a scalar or a vector from a vector. Scalars are extracted with an
/// `extreactelement`, while vectors with a `shufflevector`.
class SBUnpackInstruction : public SBInstruction {
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    llvm::Use *LLVMUse;
    if (OpIdx != getNumOperands()) {
      unsigned LLVMOpIdx = isa<ExtractElementInst>(Val) ? OpIdx : OpIdx + 1;
      LLVMUse = &cast<User>(Val)->getOperandUse(LLVMOpIdx);
    } else
      LLVMUse = cast<User>(Val)->op_end();
    return SBUse(LLVMUse, const_cast<SBUnpackInstruction *>(this), Ctxt);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<Instruction *> getLLVMInstrs() const final {
    return {cast<Instruction>(Val)};
  }
  DmpVector<Instruction *> getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

  void detachExtras() final {}

public:
  /// Please note that this may return a constant if folded.
  static SBValue *create(SBValue *Op, unsigned UnpackLane,
                         unsigned NumLanesToUnpack,
                         SBBasicBlock::iterator WhereIt, SBBasicBlock *WhereBB,
                         SBVecContext &SBCtxt);
  static SBValue *create(SBValue *Op, unsigned UnpackLane,
                         unsigned NumLanesToUnpack, SBInstruction *InsertBefore,
                         SBVecContext &SBCtxt);
  static SBValue *create(SBValue *Op, unsigned UnpackLane,
                         unsigned NumLanesToUnpack, SBBasicBlock *InsertAtEnd,
                         SBVecContext &SBCtxt);
  unsigned getUseOperandNo(const SBUse &Use) const final {
    unsigned LLVMOperandNo = Use.LLVMUse->getOperandNo();
    return isa<ExtractElementInst>(Val) ? LLVMOperandNo : LLVMOperandNo - 1;
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// Use SBBasicBlock::createSBUnpackInstruction(). Don't call the
  /// constructor directly.
  SBUnpackInstruction(ExtractElementInst *ExtractI, SBValue *UnpackOp,
                      unsigned UnpackLane, SBContext &SBCtxt);
  SBUnpackInstruction(ShuffleVectorInst *ShuffleI, SBValue *UnpackOp,
                      unsigned UnpackLane, SBContext &SBCtxt);
  friend class SBBasicBlock;

  static Value *createIR(SBValue *UnpackOp, unsigned Lane, unsigned Lanes,
                         SBBasicBlock::iterator WhereIt, SBBasicBlock *WhereBB);
  /// \Returns true if \p ShuffleI is an unpack.
  static bool isUnpack(ShuffleVectorInst *ShuffleI) {
    auto Mask = ShuffleI->getShuffleMask();
    auto NumInputElms =
        SBVecUtils::getNumElements(ShuffleI->getOperand(0)->getType());
    if (!ShuffleVectorInst::isSingleSourceMask(Mask, NumInputElms))
      return false;
    if (!ShuffleI->changesLength())
      return false;
    if (Mask.size() == 0 || (int)Mask.size() == NumInputElms)
      return false;
    if (!isa<PoisonValue>(ShuffleI->getOperand(0)))
      return false;
    // We expect element indexes in order.
    int Mask0 = Mask[0];
    if (Mask0 < NumInputElms)
      return false;
    int LastElm = Mask0;
    for (auto Elm : drop_begin(Mask)) {
      if (Elm != LastElm + 1)
        return false;
      LastElm = Elm;
    }
    return true;
  }
  /// Helper that returns the lane unpacked by \p ShuffleI.
  static int getShuffleLane(ShuffleVectorInst *ShuffleI) {
    assert(isUnpack(ShuffleI) && "Expected an unpack!");
    int TotalElms = SBVecUtils::getNumLanes(ShuffleI->getOperand(0)->getType());
    int Elm = ShuffleI->getMaskValue(0);
    int Lane = Elm - TotalElms;
    assert(Lane >= 0 && "Expected non-negative!");
    return Lane;
  }
  unsigned getUnpackLane() const {
    ConstantInt *IdxC = nullptr;
    if (auto *Extract = dyn_cast<ExtractElementInst>(Val)) {
      IdxC = cast<ConstantInt>(Extract->getIndexOperand());
      return IdxC->getSExtValue();
    }
    return getShuffleLane(cast<ShuffleVectorInst>(Val));
  }
  unsigned getNumOperands() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final {
    return hash_combine(getUnpackLane(), hashCommon());
  }
#ifndef NDEBUG
  void verify() const final;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIRVEC_H
