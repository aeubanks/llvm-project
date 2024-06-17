//===- InstrInterval.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/Transforms/SandboxIR/SandboxIR.h"

using namespace llvm;

template <typename DerefType, typename InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType> &
InstrIntervalIterator<DerefType, InstrIntervalType>::operator++() {
  assert(I != nullptr && "already at end()!");
  I = I->getNextNode();
  return *this;
}
template <typename DerefType, typename InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType>::operator++(int) {
  auto ItCopy = *this;
  ++*this;
  return ItCopy;
}
template <typename DerefType, typename InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType> &
InstrIntervalIterator<DerefType, InstrIntervalType>::operator--() {
  // `I` is nullptr for end() when ToI is the BB terminator.
  I = I != nullptr ? I->getPrevNode() : R.ToI;
  return *this;
}

template <typename DerefType, typename InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType>
InstrIntervalIterator<DerefType, InstrIntervalType>::operator--(int) {
  auto ItCopy = *this;
  --*this;
  return ItCopy;
}

InstrInterval::InstrInterval(SBInstruction *I1, SBInstruction *I2) {
  assert(!I1->isDbgInfo() && !I2->isDbgInfo() &&
         "No debug instructions allowed!");
  if (I1 != I2 && I2->comesBefore(I1))
    std::swap(I1, I2);
  FromI = I1;
  ToI = I2;
  assert((FromI == ToI || FromI->comesBefore(ToI)) &&
         "Expected FromI before ToI or equal");
}

// Explicit instantiation.
namespace llvm {
template class InstrIntervalIterator<SBInstruction &, InstrInterval>;
template class InstrIntervalIterator<SBInstruction const &,
                                     InstrInterval const>;
} // namespace llvm

template <typename IntervalT> void InstrInterval::init(IntervalT Instrs) {
  // Find the first and last instr among `Instrs`.
  SBInstruction *TopI = cast<SBInstruction>(*Instrs.begin());
  SBInstruction *BotI = TopI;
  for (SBValue *SBV : drop_begin(Instrs)) {
    auto *I = cast<SBInstruction>(SBV);
    if (I->comesBefore(TopI))
      TopI = I;
    if (BotI->comesBefore(I))
      BotI = I;
  }
  FromI = TopI;
  ToI = BotI;
  assert((FromI == ToI || FromI->comesBefore(ToI)) &&
         "Expected FromI before ToI!");
}

// Explicit instantiations.
template InstrInterval::InstrInterval(DmpVector<SBValue *>);
template InstrInterval::InstrInterval(ArrayRef<SBValue *>);
template InstrInterval::InstrInterval(ArrayRef<SBInstruction *>);
template void InstrInterval::init(ArrayRef<SBValue *>);
template void InstrInterval::init(ArrayRef<SBInstruction *>);

InstrInterval::InstrInterval(const DmpVector<SBValue *> &SBVals) {
  init(SBVals);
}

InstrInterval
InstrInterval::getUnionSingleSpan(const InstrInterval &Other) const {
  if (empty())
    return Other;
  if (Other.empty())
    return *this;
  auto *NewFromI = FromI->comesBefore(Other.FromI) ? FromI : Other.FromI;
  auto *NewToI = ToI->comesBefore(Other.ToI) ? Other.ToI : ToI;
  return {NewFromI, NewToI};
}

InstrInterval InstrInterval::getIntersection(const InstrInterval &Other) const {
  if (empty())
    return *this; // empty
  if (Other.empty())
    return InstrInterval();
  // 1. No overlap
  // A---B      this
  //       C--D Other
  if (ToI->comesBefore(Other.FromI) || Other.ToI->comesBefore(FromI))
    return InstrInterval();
  // 2. Overlap.
  // A---B   this
  //   C--D  Other
  auto NewFromI = FromI->comesBefore(Other.FromI) ? Other.FromI : FromI;
  auto NewToI = ToI->comesBefore(Other.ToI) ? ToI : Other.ToI;
  return InstrInterval(NewFromI, NewToI);
}

SmallVector<InstrInterval, 2>
InstrInterval::operator-(const InstrInterval &Other) const {
  if (disjoint(Other))
    return {*this};
  if (Other.empty())
    return {*this};
  if (*this == Other)
    return {InstrInterval()};
  InstrInterval Intersection = getIntersection(Other);
  SmallVector<InstrInterval, 2> Result;
  // Part 1, skip if empty.
  if (FromI != Intersection.FromI)
    Result.emplace_back(FromI, Intersection.FromI->getPrevNode());
  // Part 2, skip if empty.
  if (Intersection.ToI != ToI)
    Result.emplace_back(Intersection.ToI->getNextNode(), ToI);
  return Result;
}

InstrInterval
InstrInterval::getSingleDifference(const InstrInterval &Other) const {
  auto Diffs = *this - Other;
  if (Diffs.empty())
    return {};
  assert(Diffs.size() == 1 &&
         "Expected up to one interval in the difference operation!");
  return Diffs[0];
}

bool InstrInterval::contains(const SBBBIterator &It) const {
  assert(!empty() && "Expected a non-empty interval!");
  SBBasicBlock *BB = from()->getParent();
  if (It == BB->end())
    return to() == &BB->back();
  SBInstruction *I = &*It;
  return contains(I) || I == to()->getNextNode();
}

bool InstrInterval::contains(SBInstruction *I) const {
  if (empty())
    return false;
  return (FromI == I || FromI->comesBefore(I)) &&
         (I == ToI || I->comesBefore(ToI));
}

void InstrInterval::extend(SBInstruction *I) {
  if (empty()) {
    FromI = I;
    ToI = I;
    return;
  }
  if (contains(I))
    return;
  if (I->comesBefore(FromI))
    FromI = I;
  if (ToI->comesBefore(I))
    ToI = I;
}

bool InstrInterval::empty() const {
  assert(((FromI == nullptr && ToI == nullptr) ||
          (FromI != nullptr && ToI != nullptr)) &&
         "Either none or both should be null");
  return FromI == nullptr;
}

bool InstrInterval::contains(const InstrInterval &Other) const {
  if (Other.empty())
    return true;
  return (FromI == Other.FromI || FromI->comesBefore(Other.FromI)) &&
         (ToI == Other.ToI || Other.ToI->comesBefore(ToI));
}

bool InstrInterval::disjoint(const InstrInterval &Other) const {
  if (Other.empty())
    return true;
  if (empty())
    return true;
  return Other.ToI->comesBefore(FromI) || ToI->comesBefore(Other.FromI);
}

InstrInterval::iterator InstrInterval::end() {
  return iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

InstrInterval::const_iterator InstrInterval::end() const {
  return const_iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

void InstrInterval::erase(SBInstruction *I, bool CheckContained) {
  assert((!CheckContained || contains(I)) && "Instruction not in interval!");
  if (empty())
    return;
  if (FromI == ToI) {
    // Corner case: if the interval contains only one node
    if (I == FromI) {
      FromI = nullptr;
      ToI = nullptr;
    }
    return;
  }
  if (I == FromI)
    FromI = FromI->getNextNode();
  if (I == ToI)
    ToI = ToI->getPrevNode();
  assert((FromI == ToI || FromI->comesBefore(ToI)) && "Malformed interval!");
}

void InstrInterval::notifyMoveInstr(SBInstruction *I,
                                    const SBBBIterator &BeforeIt,
                                    SBBasicBlock *BB) {
  assert(contains(I) && contains(BeforeIt) &&
         "This function can only handle intra-interval instruction movement, "
         "which is what we expect from the scheduler.");
  // `I` doesn't move, so early return.
  if (std::next(I->getIterator()) == BeforeIt)
    return;

  // If `I` is at the interval's boundaries we need to move the boundaries to
  // the next/prev bundle accordingly.
  if (I == FromI) {
    assert(I != ToI && "This is equivalent to moving to itself, should have "
                       "early returned earlier!");
    FromI = I->getNextNode();
  } else if (I == ToI) {
    assert(I != FromI && "This is equivalent to moving to itself, should have "
                         "early returned earlier!");
    ToI = I->getPrevNode();
  }
  // If the destination is before/after the boundaries,
  if (BeforeIt == FromI->getIterator()) {
    // Destination is just above FromI, so update FromI.
    assert(BeforeIt != std::next(ToI->getIterator()) &&
           "Should have been handled earlier!");
    FromI = I;
    return;
  }
  if (BeforeIt == std::next(ToI->getIterator())) {
    // Destination is just below ToI, so update ToI.
    ToI = I;
    return;
  }
}

void InstrInterval::clear() {
  FromI = nullptr;
  ToI = nullptr;
}

#ifndef NDEBUG
void InstrInterval::dump(raw_ostream &OS) const {
  if (empty()) {
    OS << "Empty\n";
    return;
  }
  OS << "FromI:";
  if (FromI != nullptr)
    OS << *FromI;
  else
    OS << "NULL";
  OS << "\n";

  OS << "ToI:  ";
  if (ToI != nullptr)
    OS << *ToI;
  else
    OS << "NULL";
  OS << "\n";

  if (FromI != nullptr && ToI != nullptr) {
    if (FromI != ToI && !FromI->comesBefore(ToI)) {
      OS << "ERROR: FromI does not come before ToI !\n";
      return;
    }
    for (SBInstruction *I = FromI, *IE = ToI->getNextNode(); I != IE;
         I = I->getNextNode())
      OS << *I << "\n";
  }
}
LLVM_DUMP_METHOD void InstrInterval::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

#endif
