//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarcy that resembles that of LLVM IR:
//
//          +- SBArgument   +- SBConstant     +- SBOpaqueInstruction
//          |               |                 |
// SBValue -+- SBUser ------+- SBInstruction -+- SBInsertElementInstruction
//          |                                 |
//          +- SBBasicBlock                   +- SBExtractElementInstruction
//          |                                 |
//          +- SBFunction                     +- SBShuffleVectorInstruction
//                                            |
//                                            +- SBStoreInstruction
//                                            |
//                                            +- SBLoadInstruction
//                                            |
//                                            +- SBCmpInstruction
//                                            |
//                                            +- SBCastInstruction
//                                            |
//                                            +- SBPHINode
//                                            |
//                                            +- SBSelectInstruction
//                                            |
//                                            +- SBBinaryOperator
//                                            |
//                                            +- SBUnaryOperator
//
// SBUse
//

#ifndef LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
#define LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/SandboxIR/DmpVector.h"
#include "llvm/Transforms/SandboxIR/SandboxIRTracker.h"
#include <iterator>

using namespace llvm::PatternMatch;

namespace llvm {

class SBBasicBlock;
class SBValue;
class SBUser;
class SBPackInstruction;
class SBContext;
class SBFunction;
class DependencyGraph;

/// Represents a Def-use/Use-def edge in SandboxIR.
/// NOTE: Unlike llvm::Use, this is not an integral part of the use-def chains.
/// It is also not uniqued and is currently passed by value, so you can have to
/// SBUse objects for the same use-def edge.
class SBUse {
  llvm::Use *LLVMUse;
  friend class SBUseAttorney; // For LLVMUse
  SBUser *User;
  SBContext *Ctxt;

  /// Don't allow the user to create a SBUse directly.
  SBUse(llvm::Use *LLVMUse, SBUser *User, SBContext &Ctxt)
      : LLVMUse(LLVMUse), User(User), Ctxt(&Ctxt) {}
  SBUse() : LLVMUse(nullptr), Ctxt(nullptr) {}

  friend class SBUser;               // For constructor
  friend class SBValue;              // For constructor
  friend class SBOperandUseIterator; // For constructor
  friend class SBUserUseIterator;    // For constructor
  // Several instructions need access to the SBUse() constructor for their
  // implementation of getOperandUseInternal().
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"

public:
  operator SBValue *() const { return get(); }
  SBValue *get() const;
  SBUser *getUser() const { return User; }
  unsigned getOperandNo() const;
  SBContext *getContext() const { return Ctxt; }
  bool operator==(const SBUse &Other) const {
    assert(Ctxt == Other.Ctxt && "Contexts differ!");
    return LLVMUse == Other.LLVMUse && User == Other.User;
  }
  bool operator!=(const SBUse &Other) const { return !(*this == Other); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

/// A client-attorney class for SBUse.
class SBUseAttorney {
  static Use *getLLVMUse(SBUse &Use) { return Use.LLVMUse; }
  friend class SBBasicBlock; // For getLLVMUse()
};

/// Returns the operand edge when dereferenced.
class SBOperandUseIterator {
  SBUse Use;
  /// Don't let the user create a non-empty SBOperandUseIterator.
  SBOperandUseIterator(const SBUse &Use) : Use(Use) {}
  friend class SBUser; // For constructor
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBUse;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  SBOperandUseIterator() {}
  value_type operator*() const;
  SBOperandUseIterator &operator++();
  bool operator==(const SBOperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const SBOperandUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Returns user edge when dereferenced.
class SBUserUseIterator {
  SBUse Use;
  /// Don't let the user create a non-empty SBUserUseIterator.
  SBUserUseIterator(const SBUse &Use) : Use(Use) {}
  friend class SBValue; // For constructor

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBUse;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  SBUserUseIterator() {}
  value_type operator*() const;
  SBUserUseIterator &operator++();
  bool operator==(const SBUserUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const SBUserUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Simple adaptor class for SBUserUseIterator and SBOperandUseIterator that
/// returns \p RetTy* when dereferenced, that is SBUser* or SBValue*.
template <typename RetTy, typename ItTy> class RetTyAdaptor {
  ItTy It;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = RetTy;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;
  RetTyAdaptor(ItTy It) : It(It) {}
  RetTyAdaptor() = default;
  RetTyAdaptor &operator++() {
    ++It;
    return *this;
  }
  pointer operator*() const {
    static_assert(std::is_same<ItTy, SBUserUseIterator>::value ||
                      std::is_same<ItTy, SBOperandUseIterator>::value,
                  "Unsupported ItTy!");
    if constexpr (std::is_same<ItTy, SBUserUseIterator>::value) {
      return (*It).getUser();
    } else if constexpr (std::is_same<ItTy, SBOperandUseIterator>::value) {
      return (*It).get();
    }
  }
  bool operator==(const RetTyAdaptor &Other) const { return It == Other.It; }
  bool operator!=(const RetTyAdaptor &Other) const { return !(*this == Other); }
};

/// A SBValue has users. This is the base class.
class SBValue {
public:
  enum class ClassID : unsigned {
#define DEF_VALUE(ID, CLASS) ID,
#define DEF_USER(ID, CLASS) ID,
#define DEF_INSTR(ID, OPC, CLASS) ID,
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
      // clang-format off
#define DEF_VALUE(ID, CLASS) case ClassID::ID: return #ID;
#define DEF_USER(ID,  CLASS) case ClassID::ID: return #ID;
#define DEF_INSTR(ID, OPC, CLASS) case ClassID::ID: return #ID;
      // clang-format on
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
    }
    llvm_unreachable("Unimplemented ID");
  }

  /// For isa/dyn_cast.
  ClassID SubclassID;
#ifndef NDEBUG
  /// A unique ID used for forming the name (used for debugging).
  unsigned UID;
#endif
  /// The LLVM Value that corresponds to this SBValue.
  /// NOTE: Some SBInstructions, like Packs, may include more than one value.
  Value *Val = nullptr;
  friend class ValueAttorney; // For Val

  /// All values point to the context.
  SBContext &Ctxt;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt);
  virtual ~SBValue() = default;
  ClassID getSubclassID() const { return SubclassID; }

  using use_iterator = SBUserUseIterator;
  using const_use_iterator = SBUserUseIterator;

  use_iterator use_begin();
  const_use_iterator use_begin() const {
    return const_cast<SBValue *>(this)->use_begin();
  }
  use_iterator use_end() { return use_iterator(SBUse(nullptr, nullptr, Ctxt)); }
  const_use_iterator use_end() const {
    return const_cast<SBValue *>(this)->use_end();
  }

  iterator_range<use_iterator> uses() {
    return make_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return make_range<const_use_iterator>(use_begin(), use_end());
  }

  using user_iterator = RetTyAdaptor<SBUser, SBUserUseIterator>;
  using const_user_iterator = user_iterator;

  user_iterator user_begin();
  user_iterator user_end() {
    return user_iterator(SBUse(nullptr, nullptr, Ctxt));
  }
  const_user_iterator user_begin() const {
    return const_cast<SBValue *>(this)->user_begin();
  }
  const_user_iterator user_end() const {
    return const_cast<SBValue *>(this)->user_end();
  }

  iterator_range<user_iterator> users() {
    return make_range<user_iterator>(user_begin(), user_end());
  }
  iterator_range<const_user_iterator> users() const {
    return make_range<const_user_iterator>(user_begin(), user_end());
  }
  /// \Returns the number of unique users.
  /// WARNING: This is a linear-time operation.
  unsigned getNumUsers() const;
  /// \Returns the number of user edges (not necessarily to unique users).
  /// WARNING: This is a linear-time operation.
  unsigned getNumUses() const;
  /// WARNING: This can be expensive, as it is linear to the number of users.
  bool hasNUsersOrMore(unsigned Num) const;
  bool hasNUsesOrMore(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt >= Num)
        return true;
    }
    return false;
  }
  bool hasNUses(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt > Num)
        return true;
    }
    return Cnt == Num;
  }

  SBValue *getSingleUser() const;

  Type *getType() const { return Val->getType(); }

  SBContext &getContext() const;
  SandboxIRTracker &getTracker();
  virtual hash_code hashCommon() const {
    return hash_combine(SubclassID, &Ctxt, Val);
  }
  /// WARNING: DstU can be nullptr if it is in a BB that is not in SandboxIR!
  void replaceUsesWithIf(
      SBValue *OtherV,
      llvm::function_ref<bool(SBUser *DstU, unsigned OpIdx)> ShouldReplace);
  void replaceAllUsesWith(SBValue *Other);
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const SBValue &SBV) { return SBV.hash(); }
#ifndef NDEBUG
  /// Should crash if there is something wrong with the instruction.
  virtual void verify() const = 0;
  /// Returns the name in the form 'T<number>.' like 'T1.'
  std::string getName() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SBValue &SBV) {
    SBV.dump(OS);
    return OS;
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  virtual void dumpVerbose(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dumpVerbose() const = 0;
#endif
};

/// Helper Attorney-Client class that gives access to the underlying IR.
class ValueAttorney {
private:
  static Value *getValue(const SBValue *SBV) { return SBV->Val; }

#define DEF_VALUE(ID, CLASS) friend class CLASS;
#define DEF_USER(ID, CLASS) friend class CLASS;
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"

  friend class SBInstruction;
  friend class DependencyGraph;
  template <typename T> friend class DmpVector;
  friend class SBAnalysis;
  friend class SBPassManager;
  friend class SBContext;
  friend class SBUser;
  friend class MemSeedContainer;
  friend class SandboxIRTracker;
  friend class SBRegionBuilderFromMD;
  friend class SBRegion;
  friend class SBVecUtilsPrivileged;

  friend void
  SBValue::replaceUsesWithIf(SBValue *,
                             llvm::function_ref<bool(SBUser *, unsigned)>);
  friend class Scheduler;
  friend class SBOperandUseIterator;
  friend class SBBBIterator;
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(SBValue *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(SBUser *, SBValue *, SBValue *,
                                              SandboxIRTracker &);
  friend class DeleteOnAccept;
  friend class CreateAndInsertInstr;
  friend class EraseFromParent;
};

/// A function argument.
class SBArgument : public SBValue {
  SBArgument(Argument *Arg, SBContext &SBCtxt);
  friend class SBContext; // for createSBArgument()

public:
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBArgument &TArg) { return TArg.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<Argument>(Val) && "Expected Argument!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SBArgument &TArg) {
    TArg.dump(OS);
    return OS;
  }
  void printAsOperand(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A SBValue with operands.
class SBUser : public SBValue {
protected:
  SBUser(ClassID ID, Value *V, SBContext &SBCtxt);
  friend class SBInstruction; // For constructors.

  /// \Returns the SBUse edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  SBUse getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  virtual SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const = 0;
  friend class SBOperandUseIterator; // for getOperandUseInternal()

  /// \Returns true if \p Use should be considered as an edge to its SandboxIR
  /// operand. Most instructions should return true.
  /// Currently it is only Uses from Vectors into Packs that return false.
  virtual bool isRealOperandUse(Use &Use) const = 0;
  friend class SBUserUseIterator; // for isRealOperandUse()

  /// The default implementation works only for single-LLVMIR-instruction
  /// SBUsers and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const SBUse &Use) const {
    return Use.LLVMUse->getOperandNo();
  }
#ifndef NDEBUG
  void verifyUserOfLLVMUse(const Use &Use) const;
#endif

public:
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  using op_iterator = SBOperandUseIterator;
  using const_op_iterator = SBOperandUseIterator;
  using op_range = iterator_range<op_iterator>;
  using const_op_range = iterator_range<const_op_iterator>;

  virtual op_iterator op_begin();
  virtual op_iterator op_end();
  virtual const_op_iterator op_begin() const;
  virtual const_op_iterator op_end() const;

  op_range operands() { return make_range<op_iterator>(op_begin(), op_end()); }
  const_op_range operands() const {
    return make_range<const_op_iterator>(op_begin(), op_end());
  }
  hash_code hashCommon() const override {
    auto Hash = SBValue::hashCommon();
    for (SBValue *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
  SBValue *getOperand(unsigned OpIdx) const {
    return getOperandUse(OpIdx).get();
  }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  SBUse getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const SBUse &Use) const = 0;
  SBValue *getSingleOperand() const;
  virtual void setOperand(unsigned OperandIdx, SBValue *Operand);
  virtual unsigned getNumOperands() const {
    return isa<User>(Val) ? cast<User>(Val)->getNumOperands() : 0;
  }
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  /// WARNING: This will replace even uses that are not in SandboxIR!
  bool replaceUsesOfWith(SBValue *FromV, SBValue *ToV);

#ifndef NDEBUG
  void verify() const override { assert(isa<User>(Val) && "Expected User!"); }
  void dumpCommonHeader(raw_ostream &OS) const final;
#endif

protected:
  /// \Returns the operand index that corresponds to \p UseToMatch.
  virtual unsigned getOperandUseIdx(const Use &UseToMatch) const = 0;
  friend class SBUserAttorney; // For testing
  friend void
  SBValue::replaceUsesWithIf(SBValue *,
                             llvm::function_ref<bool(SBUser *, unsigned)>);
};

/// A simple client-attorney class that exposes some protected members of
/// SBUser for use in tests.
class SBUserAttorney {
public:
  // For testing.
  static unsigned getOperandUseIdx(const SBUser *SBU, const Use &UseToMatch) {
    return SBU->getOperandUseIdx(UseToMatch);
  }
};

class SBConstant : public SBUser {
  /// Use SBContext::createSBConstant() instead.
  SBConstant(Constant *C, SBContext &SBCtxt);
  friend class SBContext; // For constructor.
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  SBContext &getParent() const { return getContext(); }
  hash_code hashCommon() const final { return SBUser::hashCommon(); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBConstant &SBC) { return SBC.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<Constant>(Val) && "Expected Constant!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SBConstant &SBC) {
    SBC.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBInstruction;

/// The SBBasicBlock::iterator.
class SBBBIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBInstruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  BasicBlock *BB;
  /// This should always point to the bottom IR instruction of a multi-IR
  /// SBInstruction.
  BasicBlock::iterator It;
  SBContext *SBCtxt;
  pointer getSBI(BasicBlock::iterator It) const;

public:
  SBBBIterator() : BB(nullptr), SBCtxt(nullptr) {}
  SBBBIterator(BasicBlock *BB, BasicBlock::iterator It, SBContext *SBCtxt)
      : BB(BB), It(It), SBCtxt(SBCtxt) {}
  reference operator*() const { return *getSBI(It); }
  SBBBIterator &operator++();
  SBBBIterator operator++(int) {
    auto Copy = *this;
    ++*this;
    return Copy;
  }
  SBBBIterator &operator--();
  SBBBIterator operator--(int) {
    auto Copy = *this;
    --*this;
    return Copy;
  }
  bool operator==(const SBBBIterator &Other) const {
    assert(SBCtxt == Other.SBCtxt && "SBBBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const SBBBIterator &Other) const { return !(*this == Other); }
  /// \Returns true if the internal iterator is at the beginning of the IR BB.
  /// NOTE: This is meant to be used internally, during the construction of a
  /// SBBB, during which SBBB->begin() fails due to the missing mapping of
  /// BB->begin() to SandboxIR.
  bool atBegin() const;
  /// \Returns the SBInstruction that corresponds to this iterator, or null if
  /// the instruction is not found in the IR-to-SandboxIR tables.
  pointer get() const { return getSBI(It); }
};

/// A SBUser with operands and opcode.
class SBInstruction : public SBUser {
public:
  enum class Opcode {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC) OPC,
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  };

protected:
  /// Don't create objects of this class. Use a sub-class instead.
  SBInstruction(ClassID ID, Opcode Opc, Instruction *I, SBContext &SBCtxt);

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"

  /// Any extra actions that need to be performed upon detach.
  virtual void detachExtras() = 0;
  friend class SBContext; // For detachExtras()

  /// A SBInstruction may map to multiple IR Instruction. This returns its
  /// topmost IR instruction.
  Instruction *getTopmostIRInstruction() const;

  /// \Returns all IR instructions that make up this SBInstruction in reverse
  /// program order.
  virtual DmpVector<Instruction *> getLLVMInstrs() const = 0;
  friend class SBCostModel; // For getLLVMInstrs().
  /// \Returns all IR instructions with external operands. Note: This is useful
  /// for multi-IR instructions like Packs, that are composed of both
  /// internal-only and external-facing IR Instructions.
  virtual DmpVector<Instruction *>
  getLLVMInstrsWithExternalOperands() const = 0;
  friend void DeleteOnAccept::apply();
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(SBValue *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(SBUser *, SBValue *, SBValue *,
                                              SandboxIRTracker &);
  friend bool SBUser::replaceUsesOfWith(SBValue *, SBValue *);
  friend class EraseFromParent;
  friend class DeleteOnAccept;

  Opcode Opc;
  /// Maps SBInstruction::Opcode to its corresponding IR opcode, if it exists.
  static Instruction::UnaryOps getIRUnaryOp(Opcode Opc);
  static Instruction::BinaryOps getIRBinaryOp(Opcode Opc);
  static Instruction::CastOps getIRCastOp(Opcode Opc);

  // Metadata is LLMV IR, so protect it. Access this via the
  // SBInstructionAttorney class.
  MDNode *getMetadata(unsigned KindID) const {
    return cast<Instruction>(Val)->getMetadata(KindID);
  }
  MDNode *getMetadata(StringRef Kind) const {
    return cast<Instruction>(Val)->getMetadata(Kind);
  }
  friend class SBInstructionAttorney;

public:
  static const char *getOpcodeName(Opcode Opc);
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, Opcode Opc) {
    OS << getOpcodeName(Opc);
    return OS;
  }
#endif
  /// This is used by SBBasicBlcok::iterator.
  virtual unsigned getNumOfIRInstrs() const = 0;
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  SBBBIterator getIterator() const;
  SBInstruction *getNextNode() const;
  SBInstruction *getPrevNode() const;
  /// \Returns the opcode of the Instruction contained.
  Opcode getOpcode() const { return Opc; }
  /// Detach this from its parent SBBasicBlock without deleting it.
  void removeFromParent();
  /// Detach this SBValue from its parent and delete it.
  void eraseFromParent();
  /// \Returns the parent graph or null if there is no parent graph, i.e., when
  /// it holds a Constant.
  SBBasicBlock *getParent() const;
  bool isFPMath() const { return isa<FPMathOperator>(Val); }
  FastMathFlags getFastMathFlags() const {
    return cast<Instruction>(Val)->getFastMathFlags();
  }
  bool canHaveWrapFlags() const {
    return isa<OverflowingBinaryOperator>(Val) || isa<TruncInst>(Val);
  }
  bool hasNoUnsignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<Instruction>(Val)->hasNoUnsignedWrap();
  }
  bool hasNoSignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<Instruction>(Val)->hasNoSignedWrap();
  }
  /// \Returns true if this is a landingpad, a catchpad or a cleanuppadd
  bool isPad() const {
    return isa<LandingPadInst>(Val) || isa<CatchPadInst>(Val) ||
           isa<CleanupPadInst>(Val);
  }
  bool isFenceLike() const { return cast<Instruction>(Val)->isFenceLike(); }
  int64_t getInstrNumber() const;
  bool comesBefore(SBInstruction *Other) const {
    return getInstrNumber() < Other->getInstrNumber();
  }
  bool comesAfter(SBInstruction *Other) { return Other->comesBefore(this); }
  /// \Returns a (very) approximate absolute distance between this instruction
  /// and \p ToI. This is a constant-time operation.
  uint64_t getApproximateDistanceTo(SBInstruction *ToI) const;
  void moveBefore(SBBasicBlock &SBBB, const SBBBIterator &WhereIt);
  void moveBefore(SBInstruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  void moveAfter(SBInstruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  hash_code hashCommon() const override {
    return hash_combine(SBUser::hashCommon(), getParent());
  }
  void insertBefore(SBInstruction *BeforeI);
  void insertAfter(SBInstruction *AfterI);
  void insertInto(SBBasicBlock *SBBB, const SBBBIterator &WhereIt);

  bool mayWriteToMemory() const {
    return cast<Instruction>(Val)->mayWriteToMemory();
  }
  bool mayReadFromMemory() const {
    return cast<Instruction>(Val)->mayReadFromMemory();
  }
  bool isTerminator() const { return cast<Instruction>(Val)->isTerminator(); }

  bool isStackRelated() const {
    auto IsInAlloca = [](Instruction *I) {
      return isa<AllocaInst>(I) && cast<AllocaInst>(I)->isUsedWithInAlloca();
    };
    auto *I = cast<Instruction>(Val);
    return match(I, m_Intrinsic<Intrinsic::stackrestore>()) ||
           match(I, m_Intrinsic<Intrinsic::stacksave>()) || IsInAlloca(I);
  }
  /// We consider \p I as a Mem instruction if it accesses memory or if it is
  /// stack-related. This is used to determine whether this instruction needs
  /// dependency edges.
  bool isMemInst() const {
    auto IsMem = [](Instruction *I) {
      return I->mayReadOrWriteMemory() &&
             (!isa<IntrinsicInst>(I) ||
              (cast<IntrinsicInst>(I)->getIntrinsicID() !=
                   Intrinsic::sideeffect &&
               cast<IntrinsicInst>(I)->getIntrinsicID() !=
                   Intrinsic::pseudoprobe));
    };
    return IsMem(cast<Instruction>(Val)) || isStackRelated();
  }
  bool isDbgInfo() const {
    auto *I = cast<Instruction>(Val);
    return isa<DbgInfoIntrinsic>(I);
  }
  /// \Returns the number of successors that this terminator instruction has.
  unsigned getNumSuccessors() const LLVM_READONLY {
    return cast<Instruction>(Val)->getNumSuccessors();
  }
  SBBasicBlock *getSuccessor(unsigned Idx) const LLVM_READONLY;

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const SBInstruction &SBI) {
    SBI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A client-attorney class for SBInstruction.
class SBInstructionAttorney {
public:
  friend class SBRegionBuilderFromMD;
  static MDNode *getMetadata(const SBInstruction *SBI, unsigned KindID) {
    return SBI->getMetadata(KindID);
  }
  static MDNode *getMetadata(const SBInstruction *SBI, StringRef Kind) {
    return SBI->getMetadata(Kind);
  }
};

class SBCmpInstruction : public SBInstruction {
  static Opcode getCmpOpcode(unsigned CmpOp) {
    switch (CmpOp) {
    case Instruction::FCmp:
      return Opcode::FCmp;
    case Instruction::ICmp:
      return Opcode::ICmp;
    }
    llvm_unreachable("Unhandled CmpOp!");
  }

  /// Use SBContext::createSBCmpInstruction(). Don't call the
  /// constructor directly.
  SBCmpInstruction(CmpInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Cmp, getCmpOpcode(CI->getOpcode()), CI, Ctxt) {
    assert((Opc == Opcode::FCmp || Opc == Opcode::ICmp) && "Bad Opcode!");
  }
  friend class SBContext; // for SBCmpInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(CmpInst::Predicate Pred, SBValue *LHS, SBValue *RHS,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "", MDNode *FPMathTag = nullptr);
  static SBValue *create(CmpInst::Predicate Pred, SBValue *LHS, SBValue *RHS,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "", MDNode *FPMathTag = nullptr);
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBCmpInstruction &SBSI) {
    return SBSI.hash();
  }
  auto getPredicate() const { return cast<CmpInst>(Val)->getPredicate(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<CmpInst>(Val) && "Expected CmpInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBCmpInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<SBInstruction::Opcode> {
  static inline SBInstruction::Opcode getEmptyKey() {
    return (SBInstruction::Opcode)-1;
  }
  static inline SBInstruction::Opcode getTombstoneKey() {
    return (SBInstruction::Opcode)-2;
  }
  static unsigned getHashValue(const SBInstruction::Opcode &B) {
    return (unsigned)B;
  }
  static bool isEqual(const SBInstruction::Opcode &B1,
                      const SBInstruction::Opcode &B2) {
    return B1 == B2;
  }
};

class SBStoreInstruction : public SBInstruction {
  /// Use SBContext::createSBStoreInstruction(). Don't call the
  /// constructor directly.
  SBStoreInstruction(StoreInst *SI, SBContext &Ctxt)
      : SBInstruction(ClassID::Store, Opcode::Store, SI, Ctxt) {}
  friend SBContext; // for SBStoreInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBStoreInstruction *create(SBValue *V, SBValue *Ptr, MaybeAlign Align,
                                    SBInstruction *InsertBefore,
                                    SBContext &SBCtxt);
  static SBStoreInstruction *create(SBValue *V, SBValue *Ptr, MaybeAlign Align,
                                    SBBasicBlock *InsertAtEnd,
                                    SBContext &SBCtxt);
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBStoreInstruction &SBSI) {
    return SBSI.hash();
  }
  SBValue *getValueOperand() const;
  SBValue *getPointerOperand() const;
  Align getAlign() const { return cast<StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<StoreInst>(Val)->isUnordered(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<StoreInst>(Val) && "Expected StoreInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBStoreInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBLoadInstruction : public SBInstruction {
  /// Use SBContext::createSBLoadInstruction(). Don't call the
  /// constructor directly.
  SBLoadInstruction(LoadInst *LI, SBContext &Ctxt)
      : SBInstruction(ClassID::Load, Opcode::Load, LI, Ctxt) {}
  friend SBContext; // for SBLoadInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }

  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBLoadInstruction *create(Type *Ty, SBValue *Ptr, MaybeAlign Align,
                                   SBInstruction *InsertBefore,
                                   SBContext &SBCtxt, const Twine &Name = "");
  static SBLoadInstruction *create(Type *Ty, SBValue *Ptr, MaybeAlign Align,
                                   SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                                   const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBLoadInstruction &SBLI) {
    return SBLI.hash();
  }
  SBValue *getPointerOperand() const;
  Align getAlign() const { return cast<LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<LoadInst>(Val)->isSimple(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<LoadInst>(Val) && "Expected LoadInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBLoadInstruction &SBLI) {
    SBLI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBCastInstruction : public SBInstruction {
  static Opcode getCastOpcode(Instruction::CastOps CastOp) {
    switch (CastOp) {
    case Instruction::ZExt:
      return Opcode::ZExt;
    case Instruction::SExt:
      return Opcode::SExt;
    case Instruction::FPToUI:
      return Opcode::FPToUI;
    case Instruction::FPToSI:
      return Opcode::FPToSI;
    case Instruction::FPExt:
      return Opcode::FPExt;
    case Instruction::PtrToInt:
      return Opcode::PtrToInt;
    case Instruction::IntToPtr:
      return Opcode::IntToPtr;
    case Instruction::SIToFP:
      return Opcode::SIToFP;
    case Instruction::UIToFP:
      return Opcode::UIToFP;
    case Instruction::Trunc:
      return Opcode::Trunc;
    case Instruction::FPTrunc:
      return Opcode::FPTrunc;
    case Instruction::BitCast:
      return Opcode::BitCast;
    case Instruction::AddrSpaceCast:
      return Opcode::AddrSpaceCast;
    case Instruction::CastOpsEnd:
      llvm_unreachable("Bad CastOp!");
    }
    llvm_unreachable("Unhandled CastOp!");
  }
  /// Use SBContext::createSBCastInstruction(). Don't call the
  /// constructor directly.
  SBCastInstruction(CastInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Cast, getCastOpcode(CI->getOpcode()), CI, Ctxt) {
  }
  friend SBContext; // for SBCastInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(Type *Ty, Opcode Op, SBValue *Operand,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(Type *Ty, Opcode Op, SBValue *Operand,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBCastInstruction &SBCI) {
    return SBCI.hash();
  }
  Instruction::CastOps getOpcode() const {
    return cast<CastInst>(Val)->getOpcode();
  }
  Type *getSrcTy() const { return cast<CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<CastInst>(Val) && "Expected CastInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBCastInstruction &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBPHINode : public SBInstruction {
  /// Use SBContext::createSBPHINode(). Don't call the
  /// constructor directly.
  SBPHINode(PHINode *PHI, SBContext &Ctxt)
      : SBInstruction(ClassID::PHI, Opcode::PHI, PHI, Ctxt) {}
  friend SBContext; // for SBPHINode()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(Type *Ty, unsigned NumReservedValues,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(Type *Ty, unsigned NumReservedValues,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBPHINode &SBCI) { return SBCI.hash(); }
  Type *getSrcTy() const { return cast<CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<PHINode>(Val) && "Expected PHINode!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SBPHINode &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBSelectInstruction : public SBInstruction {
  /// Use SBContext::createSBSelectInstruction(). Don't call the
  /// constructor directly.
  SBSelectInstruction(SelectInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Select, Opcode::Select, CI, Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(SBValue *Cond, SBValue *True, SBValue *False,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *Cond, SBValue *True, SBValue *False,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  SBValue *getCondition() { return getOperand(0); }
  SBValue *getTrueValue() { return getOperand(1); }
  SBValue *getFalseValue() { return getOperand(2); }

  void setCondition(SBValue *New) { setOperand(0, New); }
  void setTrueValue(SBValue *New) { setOperand(1, New); }
  void setFalseValue(SBValue *New) { setOperand(2, New); }
  void swapValues() { cast<SelectInst>(Val)->swapValues(); }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBSelectInstruction &SBSI) {
    return SBSI.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<SelectInst>(Val) && "Expected SelectInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBSelectInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBBinaryOperator : public SBInstruction {
  static Opcode getBinOpOpcode(Instruction::BinaryOps BinOp) {
    switch (BinOp) {
    case Instruction::Add:
      return Opcode::Add;
    case Instruction::FAdd:
      return Opcode::FAdd;
    case Instruction::Sub:
      return Opcode::Sub;
    case Instruction::FSub:
      return Opcode::FSub;
    case Instruction::Mul:
      return Opcode::Mul;
    case Instruction::FMul:
      return Opcode::FMul;
    case Instruction::UDiv:
      return Opcode::UDiv;
    case Instruction::SDiv:
      return Opcode::SDiv;
    case Instruction::FDiv:
      return Opcode::FDiv;
    case Instruction::URem:
      return Opcode::URem;
    case Instruction::SRem:
      return Opcode::SRem;
    case Instruction::FRem:
      return Opcode::FRem;
    case Instruction::Shl:
      return Opcode::Shl;
    case Instruction::LShr:
      return Opcode::LShr;
    case Instruction::AShr:
      return Opcode::AShr;
    case Instruction::And:
      return Opcode::And;
    case Instruction::Or:
      return Opcode::Or;
    case Instruction::Xor:
      return Opcode::Xor;
    case Instruction::BinaryOpsEnd:
      llvm_unreachable("Bad BinOp!");
    }
    llvm_unreachable("Unhandled BinOp!");
  }
  /// Use SBContext::createSBBinaryOperator(). Don't call the
  /// constructor directly.
  SBBinaryOperator(BinaryOperator *BO, SBContext &Ctxt)
      : SBInstruction(ClassID::BinOp, getBinOpOpcode(BO->getOpcode()), BO,
                      Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op, SBValue *LHS,
                                        SBValue *RHS, SBValue *CopyFrom,
                                        SBInstruction *InsertBefore,
                                        SBContext &SBCtxt,
                                        const Twine &Name = "");
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op, SBValue *LHS,
                                        SBValue *RHS, SBValue *CopyFrom,
                                        SBBasicBlock *InsertAtEnd,
                                        SBContext &SBCtxt,
                                        const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBBinaryOperator &SBBO) {
    return SBBO.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<BinaryOperator>(Val) && "Expected BinaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBBinaryOperator &SBBO) {
    SBBO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBUnaryOperator : public SBInstruction {
  static Opcode getUnaryOpcode(Instruction::UnaryOps UnOp) {
    switch (UnOp) {
    case Instruction::FNeg:
      return Opcode::FNeg;
    case Instruction::UnaryOpsEnd:
      llvm_unreachable("Bad UnOp!");
    }
    llvm_unreachable("Unhandled UnOp!");
  }
  /// Use SBContext::createSBUnaryOperator(). Don't call the
  /// constructor directly.
  SBUnaryOperator(UnaryOperator *UO, SBContext &Ctxt)
      : SBInstruction(ClassID::UnOp, getUnaryOpcode(UO->getOpcode()), UO,
                      Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op, SBValue *OpV,
                                        SBValue *CopyFrom,
                                        SBInstruction *InsertBefore,
                                        SBContext &SBCtxt,
                                        const Twine &Name = "");
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op, SBValue *OpV,
                                        SBValue *CopyFrom,
                                        SBBasicBlock *InsertAtEnd,
                                        SBContext &SBCtxt,
                                        const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBUnaryOperator &SBUO) {
    return SBUO.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<UnaryOperator>(Val) && "Expected UnaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SBUnaryOperator &SBUO) {
    SBUO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBInsertElementInstruction : public SBInstruction {
  /// Use SBContext::createSBInsertElementInstruction(). Don't call the
  /// constructor directly.
  SBInsertElementInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::Insert, Opcode::Insert, I, Ctxt) {}
  SBInsertElementInstruction(ClassID SubclassID, Instruction *I,
                             SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::Insert, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(SBValue *Vec, SBValue *NewElt, SBValue *Idx,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *Vec, SBValue *NewElt, SBValue *Idx,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Insert;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBInsertElementInstruction &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBInsertElementInstruction &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBExtractElementInstruction : public SBInstruction {
  /// Use SBContext::createSBExtractElementInstruction(). Don't call the
  /// constructor directly.
  SBExtractElementInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::Extract, Opcode::Extract, I, Ctxt) {}
  SBExtractElementInstruction(ClassID SubclassID, Instruction *I,
                              SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::Extract, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(SBValue *Vec, SBValue *Idx,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *Vec, SBValue *Idx, SBBasicBlock *InsertAtEnd,
                         SBContext &SBCtxt, const Twine &Name = "");
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Extract;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBExtractElementInstruction &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBExtractElementInstruction &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBShuffleVectorInstruction : public SBInstruction {
  /// Use SBContext::createSBShuffleVectorInstruction(). Don't call the
  /// constructor directly.
  SBShuffleVectorInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::ShuffleVec, Opcode::ShuffleVec, I, Ctxt) {}
  SBShuffleVectorInstruction(ClassID SubclassID, Instruction *I,
                             SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::ShuffleVec, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(SBValue *V1, SBValue *V2, SBValue *Mask,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *V1, SBValue *V2, SBValue *Mask,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *V1, SBValue *V2, ArrayRef<int> Mask,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &Name = "");
  static SBValue *create(SBValue *V1, SBValue *V2, ArrayRef<int> Mask,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &Name = "");
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::ShuffleVec;
  }
  SmallVector<int> getShuffleMask() const {
    SmallVector<int> Mask;
    cast<ShuffleVectorInst>(Val)->getShuffleMask(Mask);
    return Mask;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBShuffleVectorInstruction &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBShuffleVectorInstruction &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBReturnInstruction : public SBInstruction {
  /// Use SBContext::createSBReturnInstruction(). Don't call the
  /// constructor directly.
  SBReturnInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::Ret, Opcode::Ret, I, Ctxt) {}
  SBReturnInstruction(ClassID SubclassID, Instruction *I, SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::Ret, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(SBValue *RetVal, SBInstruction *InsertBefore,
                         SBContext &SBCtxt);
  static SBValue *create(SBValue *RetVal, SBBasicBlock *InsertAtEnd,
                         SBContext &SBCtxt);
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Ret;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// \Returns null if there is no return value.
  SBValue *getReturnValue() const;
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBReturnInstruction &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBReturnInstruction &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBCallInstruction : public SBInstruction {
  /// Use SBContext::createSBCallInstruction(). Don't call the
  /// constructor directly.
  SBCallInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::Call, Opcode::Call, I, Ctxt) {}
  SBCallInstruction(ClassID SubclassID, Instruction *I, SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::Call, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBCallInstruction *create(FunctionType *FTy, SBValue *Func,
                                   ArrayRef<SBValue *> Args,
                                   SBBBIterator WhereIt, SBBasicBlock *WhereBB,
                                   SBContext &SBCtxt,
                                   const Twine &NameStr = "");
  static SBCallInstruction *create(FunctionType *FTy, SBValue *Func,
                                   ArrayRef<SBValue *> Args,
                                   SBInstruction *InsertBefore,
                                   SBContext &SBCtxt,
                                   const Twine &NameStr = "");
  static SBCallInstruction *create(FunctionType *FTy, SBValue *Func,
                                   ArrayRef<SBValue *> Args,
                                   SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                                   const Twine &NameStr = "");

  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Call;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBCallInstruction &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const SBCallInstruction &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBGetElementPtrInstruction : public SBInstruction {
  /// Use SBContext::createSBGetElementPtrInstruction(). Don't call the
  /// constructor directly.
  SBGetElementPtrInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::GetElementPtr, Opcode::GetElementPtr, I, Ctxt) {}
  SBGetElementPtrInstruction(ClassID SubclassID, Instruction *I,
                             SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::GetElementPtr, I, Ctxt) {}
  friend class SBContext; // For accessing the constructor in create*()
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static SBValue *create(Type *Ty, SBValue *Ptr, ArrayRef<SBValue *> IdxList,
                         SBBBIterator WhereIt, SBBasicBlock *WhereBB,
                         SBContext &SBCtxt, const Twine &NameStr = "");
  static SBValue *create(Type *Ty, SBValue *Ptr, ArrayRef<SBValue *> IdxList,
                         SBInstruction *InsertBefore, SBContext &SBCtxt,
                         const Twine &NameStr = "");
  static SBValue *create(Type *Ty, SBValue *Ptr, ArrayRef<SBValue *> IdxList,
                         SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                         const Twine &NameStr = "");

  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::GetElementPtr;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBGetElementPtrInstruction &I) {
    return I.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBGetElementPtrInstruction &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBOpaqueInstruction : public SBInstruction {
  /// Use SBContext::createSBOpaqueInstruction(). Don't call the
  /// constructor directly.
  SBOpaqueInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::Opaque, Opcode::Opaque, I, Ctxt) {}
  SBOpaqueInstruction(ClassID SubclassID, Instruction *I, SBContext &Ctxt)
      : SBInstruction(SubclassID, Opcode::Opaque, I, Ctxt) {}
  friend class SBBasicBlock;
  friend class SBContext; // For creating SB constants.
  void detachExtras() final {}
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
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

public:
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBOpaqueInstruction &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBOpaqueInstruction &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBContext;

class SBBasicBlock : public SBValue {
  /// Assigns an ordering number to instructions in the block. This is used for
  /// quick comesBefore() lookups or for a rough estimate of distance.
  DenseMap<SBInstruction *, int64_t> InstrNumberMap;
  /// When we first assign numbers to instructions we use this step. This allows
  /// us to insert new instructions in between without renumbering the whole
  /// block.
public:
  static constexpr const int64_t InstrNumberingStep = 64;

private:
  void renumberInstructions();
  /// This is called after \p I has been inserted into its parent block.
  void assignInstrNumber(SBInstruction *I);
  void removeInstrNumber(SBInstruction *I);
  friend void SBInstruction::moveBefore(SBBasicBlock &, const SBBBIterator &);
  friend void SBInstruction::insertBefore(SBInstruction *);
  friend void SBInstruction::insertInto(SBBasicBlock *, const SBBBIterator &);
  friend void SBInstruction::eraseFromParent();
  friend void SBInstruction::removeFromParent();

public:
  int64_t getInstrNumber(const SBInstruction *I) const {
    auto It = InstrNumberMap.find(I);
    assert(It != InstrNumberMap.end() && "Missing InstrNumber!");
    return It->second;
  }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildSBBasicBlockFromIR(BasicBlock *BB);
  /// \Returns the iterator to the first non-PHI instruction.
  SBBBIterator getFirstNonPHIIt();

private:
  friend void SBValue::replaceUsesWithIf(
      SBValue *,
      llvm::function_ref<bool(SBUser *, unsigned)>);   // for ChangeTracker.
  friend void SBValue::replaceAllUsesWith(SBValue *);  // for ChangeTracker.
  friend void SBUser::setOperand(unsigned, SBValue *); // for ChangeTracker

  /// Detach SBBasicBlock from the underlying BB. This is called by the
  /// destructor.
  void detach();
  /// Use SBContext::createSBBasicBlock().
  SBBasicBlock(BasicBlock *BB, SBContext &SBCtxt);
  friend class SBContext; // For createSBBasicBlock().
  friend class SBBasicBlockAttorney;

public:
  ~SBBasicBlock();
  SBFunction *getParent() const;
  /// Detaches the block and its instructions from LLVM IR.
  void detachFromLLVMIR();
  using iterator = SBBBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctxt);
  }
  SBContext &getContext() const { return Ctxt; }
  SandboxIRTracker &getTracker();
  SBInstruction *getTerminator() const;
  auto LLVMSize() const { return cast<BasicBlock>(Val)->size(); }

  hash_code hash() const final {
    return hash_combine(SBValue::hashCommon(),
                        hash_combine_range(begin(), end()));
  }
  friend hash_code hash_value(const SBBasicBlock &SBBB) { return SBBB.hash(); }

  bool empty() const { return begin() == end(); }
  SBInstruction &front() const;
  SBInstruction &back() const;

#ifndef NDEBUG
  void verify() const final {
    assert(isa<BasicBlock>(Val) && "Expected BasicBlock!");
  }
  /// Verifies LLVM IR.
  void verifyFunctionIR() const {
    assert(!verifyFunction(*cast<BasicBlock>(Val)->getParent(), &errs()));
  }
  void verify();
  /// A simple LLVM IR verifier that checks that:
  /// (i)  definitions dominate uses, and
  /// (ii) PHIs are grouped at top.
  void verifyLLVMIR() const;
  // void verifyIR(const DmpVector<SBValue *> &Instrs) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SBBasicBlock &SBBB) {
    SBBB.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
  /// Dump a range of instructions near \p SBV.
  LLVM_DUMP_METHOD void dumpInstrs(SBValue *SBV, int Num) const;
#endif
};

/// A client-attorney class for SBBasicBlock that allows access to selected
/// private members.
class SBBasicBlockAttorney {
  static BasicBlock *getBB(SBBasicBlock *SBBB) {
    return cast<BasicBlock>(SBBB->Val);
  }
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
};

class SBContext {
public:
  using RemoveCBTy = std::function<void(SBInstruction *)>;
  using InsertCBTy = std::function<void(SBInstruction *)>;
  using MoveCBTy = std::function<void(SBInstruction *, SBBasicBlock &,
                                      const SBBBIterator &)>;

  friend class SBPackInstruction; // For detachValue()

protected:
  LLVMContext &LLVMCtxt;
  SandboxIRTracker ChangeTracker;
  IRBuilder<ConstantFolder> LLVMIRBuilder;

  /// Vector of callbacks called when an IR Instruction is about to get erased.
  SmallVector<std::unique_ptr<RemoveCBTy>> RemoveInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<RemoveCBTy>>>
      RemoveInstrCallbacksBB;
  SmallVector<std::unique_ptr<InsertCBTy>> InsertInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<InsertCBTy>>>
      InsertInstrCallbacksBB;
  SmallVector<std::unique_ptr<MoveCBTy>> MoveInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<MoveCBTy>>>
      MoveInstrCallbacksBB;

  /// Maps LLVM Value to the corresponding SBValue. Owns all SandboxIR objects.
  DenseMap<Value *, std::unique_ptr<SBValue>> LLVMValueToSBValueMap;
  /// In SandboxIR some instructions correspond to multiple IR Instructions,
  /// like Packs. For such cases we map the IR instructions to the key used in
  /// LLVMValueToSBValueMap.
  DenseMap<Value *, Value *> MultiInstrMap;

  friend SBBasicBlock::~SBBasicBlock(); // For removing the scheduler.
  /// This is true during quickFlush(). It helps with some assertions that would
  /// otherwise trigger.
  bool InQuickFlush = false;

  /// This is true during the initial creation of SandboxIR. This helps select
  /// different code paths during/after creation of SandboxIR.
  bool DontNumberInstrs = false;

  friend class SBContextAttorney; // for setScheduler(), clearScheduler()
  /// Removes \p V from the maps and returns the unique_ptr.
  std::unique_ptr<SBValue> detachValue(Value *V);

  friend void SBInstruction::eraseFromParent();
  friend void SBInstruction::removeFromParent();
  friend void SBInstruction::moveBefore(SBBasicBlock &, const SBBBIterator &);

  void runRemoveInstrCallbacks(SBInstruction *I);
  void runInsertInstrCallbacks(SBInstruction *I);
  void runMoveInstrCallbacks(SBInstruction *I, SBBasicBlock &SBBB,
                             const SBBBIterator &WhereIt);

  virtual SBValue *createSBValueFromExtractElement(ExtractElementInst *ExtractI,
                                                   int Depth) {
    return getOrCreateSBExtractElementInstruction(ExtractI);
  }
  SBValue *getSBValueFromExtractElement(ExtractElementInst *ExtractI) const;
  SBValue *getOrCreateSBValueFromExtractElement(ExtractElementInst *ExtractI,
                                                int Depth);

  virtual SBValue *createSBValueFromInsertElement(InsertElementInst *InsertI,
                                                  int Depth) {
    return getOrCreateSBInsertElementInstruction(InsertI);
  }
  SBValue *getSBValueFromInsertElement(InsertElementInst *InsertI) const;
  SBValue *getOrCreateSBValueFromInsertElement(InsertElementInst *InsertI,
                                               int Depth);

  virtual SBValue *createSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI,
                                                  int Depth) {
    return getOrCreateSBShuffleVectorInstruction(ShuffleI);
  }

  SBValue *getSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI) const;
  SBValue *getOrCreateSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI,
                                               int Depth);

  /// This runs right after \p SBB has been created.
  virtual void createdSBBasicBlock(SBBasicBlock &BB) {}

#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  /// Runs right after an instruction has moved in \p BB. This is used for
  /// testing the DAG and Scheduler by SBVecContext.
  virtual void afterMoveInstrHook(SBBasicBlock &BB) {}
#endif
  /// This is called by the SBBasicBlock's destructor.
  virtual void destroyingBB(SBBasicBlock &BB) {}

  /// Helper for avoiding recursion loop when creating SBConstants.
  SmallDenseSet<Constant *, 8> VisitedConstants;
  SBValue *getOrCreateSBValueInternal(Value *V, int Depth, User *U = nullptr);

public:
  SBContext(LLVMContext &LLVMCtxt);
  virtual ~SBContext() {}
  SandboxIRTracker &getTracker() { return ChangeTracker; }
  auto &getLLVMIRBuilder() { return LLVMIRBuilder; }
  size_t getNumValues() const {
    return LLVMValueToSBValueMap.size() + MultiInstrMap.size();
  }

  /// Remove \p SBV from all SandboxIR maps and stop owning it. This effectively
  /// detaches \p SBV from the underlying IR.
  std::unique_ptr<SBValue> detach(SBValue *SBV);
  SBValue *registerSBValue(std::unique_ptr<SBValue> &&SBVPtr);

  SBValue *getSBValue(Value *V) const;

  SBConstant *getSBConstant(Constant *C) const;
  SBConstant *getOrCreateSBConstant(Constant *C);

  SBValue *getOrCreateSBValue(Value *V);

  /// Helper function called when we create SBInstructions that create new
  /// constant operands. It goes through V's operands and creates SBConstants.
  void createMissingConstantOperands(Value *V);

  // Arguments
  SBArgument *getSBArgument(Argument *Arg) const;
  SBArgument *createSBArgument(Argument *Arg);
  SBArgument *getOrCreateSBArgument(Argument *Arg);

  // InsertElementInstruction
  SBInsertElementInstruction *
  getSBInsertElementInstruction(InsertElementInst *I) const;
  SBInsertElementInstruction *
  createSBInsertElementInstruction(InsertElementInst *I);
  SBInsertElementInstruction *
  getOrCreateSBInsertElementInstruction(InsertElementInst *I);

  // InsertElementInstruction
  SBExtractElementInstruction *
  getSBExtractElementInstruction(ExtractElementInst *I) const;
  SBExtractElementInstruction *
  createSBExtractElementInstruction(ExtractElementInst *I);
  SBExtractElementInstruction *
  getOrCreateSBExtractElementInstruction(ExtractElementInst *I);

  // ShuffleVectorInstruction
  SBShuffleVectorInstruction *
  getSBShuffleVectorInstruction(ShuffleVectorInst *I) const;
  SBShuffleVectorInstruction *
  createSBShuffleVectorInstruction(ShuffleVectorInst *I);
  SBShuffleVectorInstruction *
  getOrCreateSBShuffleVectorInstruction(ShuffleVectorInst *I);

  // Return
  SBReturnInstruction *getSBReturnInstruction(ReturnInst *I) const;
  SBReturnInstruction *createSBReturnInstruction(ReturnInst *I);
  SBReturnInstruction *getOrCreateSBReturnInstruction(ReturnInst *I);

  // Call
  SBCallInstruction *getSBCallInstruction(CallInst *I) const;
  SBCallInstruction *createSBCallInstruction(CallInst *I);
  SBCallInstruction *getOrCreateSBCallInstruction(CallInst *I);

  // GEP
  SBGetElementPtrInstruction *
  getSBGetElementPtrInstruction(GetElementPtrInst *I) const;
  SBGetElementPtrInstruction *
  createSBGetElementPtrInstruction(GetElementPtrInst *I);
  SBGetElementPtrInstruction *
  getOrCreateSBGetElementPtrInstruction(GetElementPtrInst *I);

  // OpaqueInstr
  SBOpaqueInstruction *getSBOpaqueInstruction(Instruction *I) const;
  SBOpaqueInstruction *createSBOpaqueInstruction(Instruction *I);
  SBOpaqueInstruction *getOrCreateSBOpaqueInstruction(Instruction *I);

  // Store
  SBStoreInstruction *getSBStoreInstruction(StoreInst *SI) const;
  SBStoreInstruction *createSBStoreInstruction(StoreInst *SI);
  SBStoreInstruction *getOrCreateSBStoreInstruction(StoreInst *SI);

  // Load
  SBLoadInstruction *getSBLoadInstruction(LoadInst *LI) const;
  SBLoadInstruction *createSBLoadInstruction(LoadInst *LI);
  SBLoadInstruction *getOrCreateSBLoadInstruction(LoadInst *LI);

  // Cast
  SBCastInstruction *getSBCastInstruction(CastInst *CI) const;
  SBCastInstruction *createSBCastInstruction(CastInst *CI);
  SBCastInstruction *getOrCreateSBCastInstruction(CastInst *CI);

  // PHI
  SBPHINode *getSBPHINode(PHINode *PHI) const;
  SBPHINode *createSBPHINode(PHINode *PHI);
  SBPHINode *getOrCreateSBPHINode(PHINode *PHI);

  // Select
  SBSelectInstruction *getSBSelectInstruction(SelectInst *SI) const;
  SBSelectInstruction *createSBSelectInstruction(SelectInst *SI);
  SBSelectInstruction *getOrCreateSBSelectInstruction(SelectInst *SI);

  // BinaryOperator
  SBBinaryOperator *getSBBinaryOperator(BinaryOperator *BO) const;
  SBBinaryOperator *createSBBinaryOperator(BinaryOperator *BO);
  SBBinaryOperator *getOrCreateSBBinaryOperator(BinaryOperator *BO);

  // UnaryOperator
  SBUnaryOperator *getSBUnaryOperator(UnaryOperator *UO) const;
  SBUnaryOperator *createSBUnaryOperator(UnaryOperator *UO);
  SBUnaryOperator *getOrCreateSBUnaryOperator(UnaryOperator *UO);

  // Cmp
  SBCmpInstruction *getSBCmpInstruction(CmpInst *CI) const;
  SBCmpInstruction *createSBCmpInstruction(CmpInst *CI);
  SBCmpInstruction *getOrCreateSBCmpInstruction(CmpInst *CI);

  // Block
  SBBasicBlock *getSBBasicBlock(BasicBlock *BB) const;
  SBBasicBlock *createSBBasicBlock(BasicBlock *BB);

  // Function
  SBFunction *getSBFunction(Function *F) const;
  SBFunction *createSBFunction(Function *F, bool CreateBBs = true);

  /// Register a callback that gets called when a SandboxIR instruction is about
  /// to be removed from its parent. Please not that this will also be called
  /// when reverting the creation of an instruction.
  /// \Returns the function pointer, which can be used later to remove it from
  /// the callback list.
  RemoveCBTy *registerRemoveInstrCallback(RemoveCBTy CB);
  void unregisterRemoveInstrCallback(RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallback(InsertCBTy CB);
  void unregisterInsertInstrCallback(InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallback(MoveCBTy CB);
  void unregisterMoveInstrCallback(MoveCBTy *CB);

  /// Register a callback that gets called if the instruction is removed from a
  /// specific BB.
  RemoveCBTy *registerRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy CB);
  void unregisterRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy CB);
  void unregisterInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy CB);
  void unregisterMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy *CB);

  /// Clears state for the whole context quickly. This is to speed up
  /// destruction of the whole SandboxIR.
  virtual void quickFlush();

#ifndef NDEBUG
  /// Used in tests
  void disableCallbacks() { CallbacksDisabled = true; }
#endif

protected:
#ifndef NDEBUG
  bool CallbacksDisabled = false;
#endif
  friend class SBContextAttorney;
};

/// A client-attorney class for SBContext.
class SBContextAttorney {
  friend class SBRegion;
  friend class SBRegionBuilderFromMD;

public:
  static LLVMContext &getLLVMContext(SBContext &Ctxt) { return Ctxt.LLVMCtxt; }
};

class SBFunction : public SBValue {
  Function *getFunction() const { return cast<Function>(Val); }

public:
  SBFunction(Function *F, SBContext &Ctxt)
      : SBValue(ClassID::Function, F, Ctxt) {}
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  /// Iterates over SBBasicBlocks
  class iterator {
    Function::iterator It;
#ifndef NDEBUG
    Function *F;
#endif
    SBContext *Ctxt;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = SBBasicBlock;
    using pointer = SBBasicBlock *;
    using reference = value_type &;
    using iterator_category = std::bidirectional_iterator_tag;

#ifndef NDEBUG
    iterator() : F(nullptr), Ctxt(nullptr) {}
    iterator(Function::iterator It, Function *F, SBContext &Ctxt)
        : It(It), F(F), Ctxt(&Ctxt) {}
#else
    iterator() : Ctxt(nullptr) {}
    iterator(Function::iterator It, SBContext &Ctxt) : It(It), Ctxt(&Ctxt) {}
#endif

    bool operator==(const iterator &Other) const {
      assert(F == Other.F && "Comparing iterators of different functions!");
      return It == Other.It;
    }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
    iterator &operator++() {
      assert(It != F->end() && "Already at end!");
      ++It;
      return *this;
    }
    iterator operator++(int) {
      auto Copy = *this;
      ++*this;
      return Copy;
    }
    iterator &operator--() {
      assert(It != F->begin() && "Already at begin!");
      --It;
      return *this;
    }
    reference operator*() {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<SBBasicBlock>(Ctxt->getSBValue(&*It));
    }
    const SBBasicBlock &operator*() const {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<SBBasicBlock>(Ctxt->getSBValue(&*It));
    }
  };

  SBArgument *getArg(unsigned Idx) const {
    Argument *Arg = getFunction()->getArg(Idx);
    return cast<SBArgument>(Ctxt.getSBValue(Arg));
  }

  size_t arg_size() const { return getFunction()->arg_size(); }
  bool arg_empty() const { return getFunction()->arg_empty(); }

  struct LLVMArgToSBArgConst {
    SBContext &Ctxt;
    LLVMArgToSBArgConst(SBContext &Ctxt) : Ctxt(Ctxt) {}
    const SBArgument &operator()(const Argument &Arg) const {
      return *cast<SBArgument>(Ctxt.getSBValue(const_cast<Argument *>(&Arg)));
    }
  };
  using const_arg_iterator =
      mapped_iterator<Function::const_arg_iterator, LLVMArgToSBArgConst>;

  const_arg_iterator arg_begin() const {
    LLVMArgToSBArgConst GetSBArg(Ctxt);
    const Function *F = cast<Function>(Val);
    return map_iterator(F->arg_begin(), GetSBArg);
  }
  const_arg_iterator arg_end() const {
    LLVMArgToSBArgConst GetSBArg(Ctxt);
    const Function *F = cast<Function>(Val);
    return map_iterator(F->arg_end(), GetSBArg);
  }
  iterator_range<const_arg_iterator> args() const {
    return make_range(arg_begin(), arg_end());
  }

  struct LLVMArgToSBArg {
    SBContext &Ctxt;
    LLVMArgToSBArg(SBContext &Ctxt) : Ctxt(Ctxt) {}
    SBArgument &operator()(Argument &Arg) const {
      return *cast<SBArgument>(Ctxt.getSBValue(&Arg));
    }
  };
  using arg_iterator = mapped_iterator<Function::arg_iterator, LLVMArgToSBArg>;

  arg_iterator arg_begin() {
    LLVMArgToSBArg GetSBArg(Ctxt);
    Function *F = cast<Function>(Val);
    return map_iterator(F->arg_begin(), GetSBArg);
  }
  arg_iterator arg_end() {
    LLVMArgToSBArg GetSBArg(Ctxt);
    Function *F = cast<Function>(Val);
    return map_iterator(F->arg_end(), GetSBArg);
  }
  iterator_range<arg_iterator> args() {
    return make_range(arg_begin(), arg_end());
  }

  SBBasicBlock &getEntryBlock() const {
    BasicBlock &EntryBB = getFunction()->getEntryBlock();
    return *cast<SBBasicBlock>(Ctxt.getSBValue(&EntryBB));
  }

  iterator begin() const {
    Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->begin(), F, Ctxt);
#else
    return iterator(F->begin(), Ctxt);
#endif
  }
  iterator end() const {
    Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->end(), F, Ctxt);
#else
    return iterator(F->end(), Ctxt);
#endif
  }

  /// Detaches the function, its blocks and its instructions from LLVM IR.
  void detachFromLLVMIR();

  hash_code hash() const final {
    auto Hash =
        hash_combine(SBValue::hashCommon(), hash_combine_range(begin(), end()));
    for (auto ArgIdx : seq<unsigned>(0, arg_size()))
      Hash = hash_combine(Hash, getArg(ArgIdx));
    return Hash;
  }
  friend hash_code hash_value(const SBFunction &SBF) { return SBF.hash(); }

#ifndef NDEBUG
  void verify() const final {
    assert(isa<Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

using sb_succ_iterator = SuccIterator<SBInstruction, SBBasicBlock>;
using const_sb_succ_iterator =
    SuccIterator<const SBInstruction, const SBBasicBlock>;
using sb_succ_range = iterator_range<sb_succ_iterator>;
using const_sb_succ_range = iterator_range<const_sb_succ_iterator>;

inline sb_succ_iterator succ_begin(SBInstruction *I) {
  return sb_succ_iterator(I);
}
inline const_sb_succ_iterator succ_begin(const SBInstruction *I) {
  return const_sb_succ_iterator(I);
}
inline sb_succ_iterator succ_end(SBInstruction *I) {
  return sb_succ_iterator(I, true);
}
inline const_sb_succ_iterator succ_end(const SBInstruction *I) {
  return const_sb_succ_iterator(I, true);
}
inline bool succ_empty(const SBInstruction *I) {
  return succ_begin(I) == succ_end(I);
}
inline unsigned succ_size(const SBInstruction *I) {
  return std::distance(succ_begin(I), succ_end(I));
}
inline sb_succ_range successors(SBInstruction *I) {
  return sb_succ_range(succ_begin(I), succ_end(I));
}
inline const_sb_succ_range successors(const SBInstruction *I) {
  return const_sb_succ_range(succ_begin(I), succ_end(I));
}

inline sb_succ_iterator succ_begin(SBBasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator());
}
inline const_sb_succ_iterator succ_begin(const SBBasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator());
}
inline sb_succ_iterator succ_end(SBBasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator(), true);
}
inline const_sb_succ_iterator succ_end(const SBBasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator(), true);
}
inline bool succ_empty(const SBBasicBlock *BB) {
  return succ_begin(BB) == succ_end(BB);
}
inline unsigned succ_size(const SBBasicBlock *BB) {
  return std::distance(succ_begin(BB), succ_end(BB));
}
inline sb_succ_range successors(SBBasicBlock *BB) {
  return sb_succ_range(succ_begin(BB), succ_end(BB));
}
inline const_sb_succ_range successors(const SBBasicBlock *BB) {
  return const_sb_succ_range(succ_begin(BB), succ_end(BB));
}

// GraphTraits for SBBasicBlock.
template <> struct GraphTraits<SBBasicBlock *> {
  using NodeRef = SBBasicBlock *;
  using ChildIteratorType = sb_succ_iterator;
  static NodeRef getEntryNode(SBBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};
template <> struct GraphTraits<const SBBasicBlock *> {
  using NodeRef = const SBBasicBlock *;
  using ChildIteratorType = const_sb_succ_iterator;
  static NodeRef getEntryNode(const SBBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};

template <typename RangeT>
DmpVector<SBValue *> getOperandBundle(const RangeT &Bndl, unsigned OpIdx) {
  DmpVector<SBValue *> OpVec;
  OpVec.reserve(Bndl.size());
  for (auto *SBV : Bndl) {
    auto *SBI = cast<SBInstruction>(SBV);
    assert(OpIdx < SBI->getNumOperands() && "Out of bounds!");
    OpVec.push_back(SBI->getOperand(OpIdx));
  }
  return OpVec;
}

template <typename RangeT>
SmallVector<DmpVector<SBValue *>, 2> getOperandBundles(const RangeT &Bndl) {
  SmallVector<DmpVector<SBValue *>, 2> OpVecs;
#ifndef NDEBUG
  unsigned NumOps = cast<SBInstruction>(Bndl[0])->getNumOperands();
  assert(all_of(drop_begin(Bndl),
                [NumOps](auto *V) {
                  return cast<SBInstruction>(V)->getNumOperands() == NumOps;
                }) &&
         "Expected same number of operands!");
#endif
  for (unsigned OpIdx :
       seq<unsigned>(cast<SBInstruction>(Bndl[0])->getNumOperands()))
    OpVecs.push_back(getOperandBundle(Bndl, OpIdx));
  return OpVecs;
}

} // namespace llvm
#endif // LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
