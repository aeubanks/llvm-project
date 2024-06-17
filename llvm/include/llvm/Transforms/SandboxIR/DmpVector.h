//===- DmpVector.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A DmpVector is a vector that you can dump().
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H
#define LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class SBValue;
class SBInstruction;
class SBBasicBlock;

/// Just like a small vector but with dump() and operator<<.
template <typename T> class DmpVectorBase : public SmallVector<T> {
public:
  DmpVectorBase() : SmallVector<T>() {}
  DmpVectorBase(std::initializer_list<T> Vals) : SmallVector<T>(Vals) {}
  DmpVectorBase(ArrayRef<T> Vals) : SmallVector<T>(Vals) {}
  DmpVectorBase(SmallVector<T> &&Vals) : SmallVector<T>(std::move(Vals)) {}
  DmpVectorBase(DmpVectorBase &&Other) : SmallVector<T>(std::move(Other)) {}
  DmpVectorBase(const DmpVectorBase &Other) : SmallVector<T>(Other) {}
  explicit DmpVectorBase(size_t Sz) { this->grow(Sz); }
  template <typename ItT>
  DmpVectorBase(ItT Begin, ItT End) : SmallVector<T>(Begin, End) {}
  DmpVectorBase &operator=(const DmpVectorBase &Other) {
    SmallVector<T>::operator=(Other);
    return *this;
  }
  void reserve(size_t Sz) { this->grow(Sz); }
  hash_code hash() const {
    return hash_combine_range(this->begin(), this->end());
  }
  friend hash_code hash_value(const DmpVectorBase &B) { return B.hash(); }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : *this) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const DmpVectorBase<T> &Vec) {
    return Vec.dump(OS);
  }
};

template <typename T> class DmpVector : public DmpVectorBase<T> {
public:
  DmpVector() : DmpVectorBase<T>() {}
  DmpVector(std::initializer_list<T> Vals) : DmpVectorBase<T>(Vals) {}
  DmpVector(ArrayRef<T> Vals) : DmpVectorBase<T>(Vals) {}
  DmpVector(SmallVector<T> &&Vals) : DmpVectorBase<T>(std::move(Vals)) {}
  DmpVector(DmpVector &&Other) : DmpVectorBase<T>(std::move(Other)) {}
  DmpVector(const DmpVector &Other) : DmpVectorBase<T>(std::move(Other)) {}
  explicit DmpVector(size_t Sz) : DmpVectorBase<T>(Sz) {}
  template <typename ItT>
  DmpVector(ItT Begin, ItT End) : DmpVectorBase<T>(Begin, End) {}
  DmpVector &operator=(const DmpVector &Other) {
    DmpVectorBase<T>::operator=(Other);
    return *this;
  }
};

/// An immutable view of a DmpVector with dump().
/// This inherits from ArrayRef, so it has a similar API.
template <typename T> class DmpVectorView : public ArrayRef<T> {
public:
  /// DmpVector constructor.
  DmpVectorView(const DmpVector<T> &Vec) : ArrayRef<T>(Vec) {}
  /// Range constructor.
  DmpVectorView(T *Begin, T *End) : ArrayRef<T>(Begin, End) {}
  /// ArrayRef constructor.
  DmpVectorView(const ArrayRef<T> &Array) : ArrayRef<T>(Array) {}
  /// Default constructor.
  DmpVectorView() : ArrayRef<T>() {}

  hash_code hash() const {
    return hash_combine_range(this->begin(), this->end());
  }
  friend hash_code hash_value(const DmpVectorView &View) { return View.hash(); }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : *this) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const DmpVectorView<T> &Vec) {
    return Vec.dump(OS);
  }
};

/// Spcialization for Value*
template <> class DmpVector<Value *> : public DmpVectorBase<Value *> {
public:
  DmpVector<Value *>() : DmpVectorBase() {}
  DmpVector<Value *>(std::initializer_list<Value *> Vals)
      : DmpVectorBase(Vals) {}
  DmpVector<Value *>(ArrayRef<Value *> Vals) : DmpVectorBase(Vals) {}
  DmpVector<Value *>(SmallVector<Value *> &&Vals)
      : DmpVectorBase(std::move(Vals)) {}
  DmpVector<Value *>(DmpVector<Value *> &&Other)
      : DmpVectorBase(std::move(Other)) {}
  DmpVector<Value *>(const DmpVector<Value *> &Other) : DmpVectorBase(Other) {}
  explicit DmpVector<Value *>(size_t Sz) : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<Value *>(ItT Begin, ItT End) : DmpVectorBase(Begin, End) {}
  DmpVector<Value *> &operator=(const DmpVector<Value *> &Other) {
    DmpVectorBase<Value *>::operator=(Other);
    return *this;
  }
  static DmpVector<Value *> create(const DmpVector<SBValue *> &SBVec);

  using instr_iterator = mapped_iterator<iterator, Instruction *(*)(Value *)>;
  using const_instr_iterator =
      mapped_iterator<const_iterator, Instruction *(*)(Value *)>;

  static Instruction *valueToInstr(Value *V) { return cast<Instruction>(V); }
  instr_iterator ibegin() { return map_iterator(begin(), valueToInstr); }
  instr_iterator iend() { return map_iterator(end(), valueToInstr); }
  const_instr_iterator ibegin() const {
    return map_iterator<const_iterator, Instruction *(*)(Value *)>(
        begin(), valueToInstr);
  }
  const_instr_iterator iend() const {
    return map_iterator<const_iterator, Instruction *(*)(Value *)>(
        end(), valueToInstr);
  }
  using instr_range_t = iterator_range<instr_iterator>;
  instr_range_t instrRange() { return make_range(ibegin(), iend()); }
  using const_instr_range_t = iterator_range<const_instr_iterator>;
  const_instr_range_t instrRange() const {
    return make_range(ibegin(), iend());
  }
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<Value *>> {
  static inline DmpVector<Value *> getEmptyKey() {
    return DmpVector<Value *>((Value *)-1);
  }
  static inline DmpVector<Value *> getTombstoneKey() {
    return DmpVector<Value *>((Value *)-2);
  }
  static unsigned getHashValue(const DmpVector<Value *> &B) { return B.hash(); }
  static bool isEqual(const DmpVector<Value *> &B1,
                      const DmpVector<Value *> &B2) {
    return B1 == B2;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const SBValue &SBV);

/// Spcialization for SBValue*
template <> class DmpVector<SBValue *> : public DmpVectorBase<SBValue *> {
  void init(const DmpVector<Value *> &Vec, const SBBasicBlock &SBBB);

public:
  DmpVector<SBValue *>() : DmpVectorBase<SBValue *>() {}
  DmpVector<SBValue *>(std::initializer_list<SBValue *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<SBValue *>(ArrayRef<SBValue *> Instrs) : DmpVectorBase(Instrs) {}
  DmpVector<SBValue *>(SmallVector<SBValue *> &&Instrs)
      : DmpVectorBase(std::move(Instrs)) {}
  DmpVector<SBValue *>(DmpVector<SBValue *> &&Other)
      : DmpVectorBase<SBValue *>(std::move(Other)) {}
  DmpVector<SBValue *>(const DmpVector<SBValue *> &Other)
      : DmpVectorBase<SBValue *>(Other) {}
  explicit DmpVector<SBValue *>(size_t Sz) : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<SBValue *>(ItT Begin, ItT End) : DmpVectorBase(Begin, End) {}
  DmpVector<SBValue *>(const DmpVector<Value *> &Vec,
                       const SBBasicBlock &SBBB) {
    init(Vec, SBBB);
  }
  DmpVector<SBValue *> &operator=(const DmpVector<SBValue *> &Other) {
    DmpVectorBase<SBValue *>::operator=(Other);
    return *this;
  }
  DmpVector<Value *> getLLVMValueVector() const;

  using instr_iterator =
      mapped_iterator<iterator, SBInstruction *(*)(SBValue *)>;
  using const_instr_iterator =
      mapped_iterator<const_iterator, SBInstruction *(*)(SBValue *)>;

  static SBInstruction *valueToInstr(SBValue *V) {
    return cast<SBInstruction>(V);
  }
  instr_iterator ibegin() { return map_iterator(begin(), valueToInstr); }
  instr_iterator iend() { return map_iterator(end(), valueToInstr); }
  const_instr_iterator ibegin() const {
    return map_iterator<const_iterator, SBInstruction *(*)(SBValue *)>(
        begin(), valueToInstr);
  }
  const_instr_iterator iend() const {
    return map_iterator<const_iterator, SBInstruction *(*)(SBValue *)>(
        end(), valueToInstr);
  }
  using instr_range_t = iterator_range<instr_iterator>;
  instr_range_t instrRange() { return make_range(ibegin(), iend()); }
  using const_instr_range_t = iterator_range<const_instr_iterator>;
  const_instr_range_t instrRange() const {
    return make_range(ibegin(), iend());
  }
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<SBValue *>> {
  static inline DmpVector<SBValue *> getEmptyKey() {
    return DmpVector<SBValue *>((SBValue *)-1);
  }
  static inline DmpVector<SBValue *> getTombstoneKey() {
    return DmpVector<SBValue *>((SBValue *)-2);
  }
  static unsigned getHashValue(const DmpVector<SBValue *> &B) {
    return B.hash();
  }
  static bool isEqual(const DmpVector<SBValue *> &B1,
                      const DmpVector<SBValue *> &B2) {
    return B1 == B2;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const SBInstruction &SBI);

/// Spcialization for SBInstruction*
template <>
class DmpVector<SBInstruction *> : public DmpVectorBase<SBInstruction *> {
  void init(const DmpVector<Value *> &Vec, const SBBasicBlock &SBBB);

public:
  DmpVector<SBInstruction *>() : DmpVectorBase<SBInstruction *>() {}
  DmpVector<SBInstruction *>(std::initializer_list<SBInstruction *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<SBInstruction *>(ArrayRef<SBInstruction *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<SBInstruction *>(SmallVector<SBInstruction *> &&Instrs)
      : DmpVectorBase(std::move(Instrs)) {}
  DmpVector<SBInstruction *>(DmpVector<SBInstruction *> &&Other)
      : DmpVectorBase<SBInstruction *>(std::move(Other)) {}
  DmpVector<SBInstruction *>(const DmpVector<SBInstruction *> &Other)
      : DmpVectorBase<SBInstruction *>(Other) {}
  explicit DmpVector<SBInstruction *>(size_t Sz) : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<SBInstruction *>(ItT Begin, ItT End) : DmpVectorBase(Begin, End) {}
  DmpVector<SBInstruction *>(const DmpVector<Value *> &Vec,
                             const SBBasicBlock &SBBB) {
    init(Vec, SBBB);
  }
  DmpVector<SBInstruction *> &
  operator=(const DmpVector<SBInstruction *> &Other) {
    DmpVectorBase<SBInstruction *>::operator=(Other);
    return *this;
  }
  DmpVector<SBValue *> getOperandBundle(unsigned OpIdx) const;
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<SBInstruction *>> {
  static inline DmpVector<SBInstruction *> getEmptyKey() {
    return DmpVector<SBInstruction *>((SBInstruction *)-1);
  }
  static inline DmpVector<SBInstruction *> getTombstoneKey() {
    return DmpVector<SBInstruction *>((SBInstruction *)-2);
  }
  static unsigned getHashValue(const DmpVector<SBInstruction *> &Vec) {
    return Vec.hash();
  }
  static bool isEqual(const DmpVector<SBInstruction *> &Vec1,
                      const DmpVector<SBInstruction *> &Vec2) {
    return Vec1 == Vec2;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H
