//===- VecUtils.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helpers for the Sandbox Vectorizer
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/SandboxIR/DmpVector.h"

namespace llvm {

class SBInstruction;
class SBBBIterator;
class SBValue;
class ScalarEvolution;

class SBVecUtils {
public:
  /// \Returns true if \p I1 is accessing a prior memory location than \p I2.
  template <typename LoadOrStoreT>
  static bool comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                               ScalarEvolution &SE, const DataLayout &DL);
  /// \Returns true if \p I1 and \p I2 are load/stores accessing consecutive
  /// memory addresses.
  template <typename LoadOrStoreT>
  static bool areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                             ScalarEvolution &SE, const DataLayout &DL);

  template <typename LoadOrStoreT>
  static bool areConsecutive(const DmpVector<SBValue *> &SBBndl,
                             ScalarEvolution &SE, const DataLayout &DL);

  /// If \p SBI is a load it returns its type. If a store it returns its value
  /// operand type.
  template <typename LoadOrStoreT>
  static Type *getLoadStoreType(LoadOrStoreT *SBI);
  /// \Returns the number of elements in \p Ty, that is the number of lanes if
  /// vector or 1 if scalar.
  static int getNumElements(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() : 1;
  }
  /// Returns \p Ty if scalar or its element type if vector.
  static Type *getElementType(Type *Ty) {
    assert((!isa<VectorType>(Ty) || isa<FixedVectorType>(Ty)) &&
           "Expected only Fixed vector types!");
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getElementType() : Ty;
  }
  /// \Returns the scalar type shared among Nodes in Bndl. \Returns nullptr if
  /// they don't share a common scalar type.
  static Type *getCommonScalarType(const DmpVector<SBValue *> &Bndl);
  /// Same as getCommonScalarType() but expects that there is a common scalar
  /// type. If not it will crash in a DEBUG build.
  static Type *getCommonScalarTypeFast(const DmpVector<SBValue *> &Bndl);

  /// Unlike other instructions, getting the value of a store requires a
  /// function call.
  static Value *getExpectedValue(Instruction *I);
  static SBValue *getExpectedValue(const SBInstruction *I);

  /// A store's type is void, which is rather useless. This function does the
  /// right thing and returns the type of the stored value.
  static Type *getExpectedType(Value *V);
  static Type *getExpectedType(const SBValue *V);

  template <typename ValT> static bool isVector(ValT *V) {
    return isa<FixedVectorType>(SBVecUtils::getExpectedType(V));
  }

  /// \Returns the number of vector lanes of \p Ty or 1 if not a vector.
  /// NOTE: It crashes if \p V is a scalable vector.
  static int getNumLanes(Type *Ty) {
    assert(!isa<ScalableVectorType>(Ty) && "Expect fixed vector");
    if (!isa<FixedVectorType>(Ty))
      return 1;
    return cast<FixedVectorType>(Ty)->getNumElements();
  }

  /// \Returns the expected vector lanes of \p V or 1 if not a vector.
  /// NOTE: It crashes if \p V is a scalable vector.
  static int getNumLanes(Value *V) {
    return SBVecUtils::getNumLanes(getExpectedType(V));
  }
  static unsigned getNumLanes(const SBValue *SBV);
  /// This works even if Bndl contains Nodes with vector type.
  static unsigned getNumLanes(const DmpVector<SBValue *> &Bndl);

  static unsigned getNumBits(SBValue *SBV, const DataLayout &DL);
  static unsigned getNumBits(const DmpVector<SBInstruction *> &Bndl,
                             const DataLayout &DL);
  static unsigned getNumBits(const DmpVector<SBValue *> &Bndl,
                             const DataLayout &DL);

  // Returns the next power of 2.
  static unsigned getCeilPowerOf2(unsigned Num) {
    if (Num == 0)
      return Num;
    Num--;
    for (int ShiftBy = 1; ShiftBy < 32; ShiftBy <<= 1)
      Num |= Num >> ShiftBy;
    return Num + 1;
  }
  static unsigned getFloorPowerOf2(unsigned Num) {
    if (Num == 0)
      return Num;
    unsigned Mask = Num;
    Mask >>= 1;
    for (int ShiftBy = 1; ShiftBy < 32; ShiftBy <<= 1)
      Mask |= Mask >> ShiftBy;
    return Num & ~Mask;
  }
  static bool isPowerOf2(unsigned Num) { return getFloorPowerOf2(Num) == Num; }

  /// \Returns the a type that is \p NumElts times wider than \p ElemTy.
  /// It works for both scalar and vector \p ElemTy.
  static Type *getWideType(Type *ElemTy, uint32_t NumElts);

  static bool areInSameBB(const DmpVector<Value *> &Instrs);
  static bool areInSameBB(const DmpVector<SBValue *> &SBInstrs);

  /// \Returns the iterator right after the lowest instruction in \p Bndl. If \p
  /// Bndl contains only non-instructions, or if the instructions in BB are at
  /// different blocks, other than \p BB, it returns the beginning of \p BB.
  static BasicBlock::iterator
  getInsertPointAfter(const DmpVector<Value *> &Bndl, BasicBlock *BB,
                      bool SkipPHIs = true, bool SkipPads = true);

  static SBBBIterator getInsertPointAfter(const DmpVector<SBValue *> &Bndl,
                                          SBBasicBlock *BB,
                                          bool SkipPHIs = true,
                                          bool SkipPads = true);

  static std::pair<SBBasicBlock *, SBBBIterator>
  getInsertPointAfterInstrs(const DmpVector<SBValue *> &InstrRange);

  template <typename BuilderT, typename InstrRangeT>
  static void setInsertPointAfter(const InstrRangeT &Instrs, BasicBlock *BB,
                                  BuilderT &Builder, bool SkipPHIs = true) {
    auto WhereIt = getInsertPointAfter(Instrs, BB, SkipPHIs);
    Builder.SetInsertPoint(BB, WhereIt);
  }

  /// \Returns the number that corresponds to the index operand of \p InsertI.
  static std::optional<int> getInsertLane(InsertElementInst *InsertI);
  /// \Returns the number that corresponds to the index operand of \p ExtractI.
  static std::optional<int> getExtractLane(ExtractElementInst *ExtractI);
  /// \Returns the constant index lane of an insert or extract, or nullopt if
  /// not a constant.
  static std::optional<int> getConstantIndex(Instruction *InsertOrExtractI);

  static bool differentMathFlags(const DmpVector<SBValue *> &SBBndl);
  static bool differentWrapFlags(const DmpVector<SBValue *> &SBBndl);

  /// \Returns the lowest in BB among \p Instrs.
  static SBInstruction *getLowest(const DmpVector<SBValue *> &Instrs);
  static SBInstruction *getHighest(const DmpVector<SBValue *> &Instrs);

  static unsigned getAddrOperandIdx(SBInstruction *LoadOrStore);
};

/// TODO: These are utility functions that can access LLVM IR through
/// ValueAttorney. They should be removed at some point.
class SBVecUtilsPrivileged {
public:
  /// Proxy for llvm::propagateMetadata().
  static void propagateMetadata(SBInstruction *SBI,
                                const DmpVector<SBValue *> &SBVals);
  /// \Returns the number gap between the memory locations accessed by \p I1 and
  /// \p I2 in bytes.
  template <typename LoadOrStoreT>
  static std::optional<int>
  getPointerDiffInBytes(LoadOrStoreT *I1, LoadOrStoreT *I2, ScalarEvolution &SE,
                        const DataLayout &DL);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H
