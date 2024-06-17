//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/SandboxIR/SandboxIR.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "SBVec"

SBValue *SBUse::get() const { return Ctxt->getSBValue(LLVMUse->get()); }
unsigned SBUse::getOperandNo() const { return User->getUseOperandNo(*this); }
#ifndef NDEBUG
void SBUse::dump(raw_ostream &OS) const {
  SBValue *Def = nullptr;
  if (LLVMUse == nullptr)
    OS << "<null> LLVM Use! ";
  else
    Def = Ctxt->getSBValue(LLVMUse->get());
  OS << "Def:  ";
  if (Def == nullptr)
    OS << "NULL";
  else
    OS << *Def;
  OS << "\n";

  OS << "User: ";
  if (User == nullptr)
    OS << "NULL";
  else
    OS << *User;
  OS << "\n";

  OS << "OperandNo: ";
  if (User == nullptr)
    OS << "N/A";
  else
    OS << getOperandNo();
  OS << "\n";
}

void SBUse::dump() const { dump(dbgs()); }
#endif // NDEBUG

SBUse SBOperandUseIterator::operator*() const { return Use; }

SBOperandUseIterator &SBOperandUseIterator::operator++() {
  assert(Use.LLVMUse != nullptr && "Already at end!");
  SBUser *User = Use.getUser();
  Use = User->getOperandUseInternal(Use.getOperandNo() + 1, /*Verify=*/false);
  return *this;
}

SBUse SBUserUseIterator::operator*() const { return Use; }

SBUserUseIterator &SBUserUseIterator::operator++() {
  llvm::Use *&LLVMUse = Use.LLVMUse;
  assert(LLVMUse != nullptr && "Already at end!");
  LLVMUse = LLVMUse->getNext();
  if (LLVMUse == nullptr) {
    Use.User = nullptr;
    return *this;
  }
  auto *Ctxt = Use.Ctxt;
  auto *LLVMUser = LLVMUse->getUser();
  SBUser *User = cast_or_null<SBUser>(Ctxt->getSBValue(LLVMUser));
  // This is for uses into Packs that should be skipped.
  // For example:
  //   %Op = add <2 x i8> %v, %v
  //   %Extr0 = extractelement <2 x i8> %Op, i64 0
  //   %Pack0 = insertelement <2 x i8> poison, i8 %Extr0, i64 0
  //   %Extr1 = extractelement <2 x i8> %Op, i64 1
  //   %Pack1 = insertelement <2 x i8> %Pack0, i8 %Extr1, i64 1
  // There should be only 1 Use edge from Op to Pack.
  if (User != nullptr && !User->isRealOperandUse(*LLVMUse))
    return ++(*this);
  Use.User = User;
  return *this;
}

SBValue::SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt)
    : SubclassID(SubclassID), Val(Val), Ctxt(Ctxt) {
#ifndef NDEBUG
  UID = Ctxt.getNumValues();
#endif
}

SBValue::use_iterator SBValue::use_begin() {
  llvm::Use *LLVMUse = nullptr;
  if (Val->use_begin() != Val->use_end())
    LLVMUse = &*Val->use_begin();
  SBUser *User =
      LLVMUse != nullptr
          ? cast_or_null<SBUser>(Ctxt.getSBValue(Val->use_begin()->getUser()))
          : nullptr;
  return use_iterator(SBUse(LLVMUse, User, Ctxt));
}

SBValue::user_iterator SBValue::user_begin() {
  auto UseBegin = Val->use_begin();
  auto UseEnd = Val->use_end();
  bool AtEnd = UseBegin == UseEnd;
  llvm::Use *LLVMUse = AtEnd ? nullptr : &*UseBegin;
  SBUser *User =
      AtEnd ? nullptr
            : cast_or_null<SBUser>(Ctxt.getSBValue(&*LLVMUse->getUser()));
  return user_iterator(SBUse(LLVMUse, User, Ctxt));
}

unsigned SBValue::getNumUsers() const {
  // Look for unique users.
  SmallPtrSet<SBValue *, 4> UserNs;
  for (User *U : Val->users())
    UserNs.insert(getContext().getSBValue(U));
  return UserNs.size();
}

unsigned SBValue::getNumUses() const {
  unsigned Cnt = 0;
  for (User *U : Val->users()) {
    (void)U;
    ++Cnt;
  }
  return Cnt;
}

bool SBValue::hasNUsersOrMore(unsigned Num) const {
  SmallPtrSet<SBValue *, 4> UserNs;
  for (User *U : Val->users()) {
    UserNs.insert(getContext().getSBValue(U));
    if (UserNs.size() >= Num)
      return true;
  }
  return false;
}

SBValue *SBValue::getSingleUser() const {
  assert(Val->hasOneUser() && "Expected single user");
  return *users().begin();
}

SBContext &SBValue::getContext() const { return Ctxt; }

SandboxIRTracker &SBValue::getTracker() { return getContext().getTracker(); }

void SBValue::replaceUsesWithIf(
    SBValue *OtherV,
    llvm::function_ref<bool(SBUser *DstU, unsigned OpIdx)> ShouldReplace) {
  assert(getType() == OtherV->getType() && "Can't replace with different type");
  Value *OtherVal = OtherV->Val;
  auto &Tracker = getTracker();
  Val->replaceUsesWithIf(
      OtherVal, [&ShouldReplace, &Tracker, this](Use &U) -> bool {
        SBUser *DstU = cast_or_null<SBUser>(Ctxt.getSBValue(U.getUser()));
        if (DstU == nullptr)
          return false;
        unsigned OpIdx = DstU->getOperandUseIdx(U);
        if (!ShouldReplace(DstU, OpIdx))
          return false;
        if (Tracker.tracking())
          // Tracking like so should be cheaper than replaceAllUsesWith()
          Tracker.track(std::make_unique<SetOperand>(DstU, OpIdx, Tracker));
        return true;
      });
}

void SBValue::replaceAllUsesWith(SBValue *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with SBValue of different type!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<ReplaceAllUsesWith>(this, Tracker));
  Val->replaceAllUsesWith(Other->Val);
}

#ifndef NDEBUG
std::string SBValue::getName() const {
  std::stringstream SS;
  SS << "T" << UID << ".";
  return SS.str();
}

void SBValue::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void SBValue::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";

  // TODO: For now also dump users, but should be removed.
  if (!isa<Constant>(Val)) {
    OS << "Users: ";
    for (auto *SBU : users()) {
      if (SBU != nullptr)
        OS << SBU->getName();
      else
        OS << "NULL";
      OS << ", ";
    }
  }
}

void SBValue::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void SBValue::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void SBValue::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void SBValue::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBArgument::SBArgument(Argument *Arg, SBContext &SBCtxt)
    : SBValue(ClassID::Argument, Arg, SBCtxt) {}

SBUser::SBUser(ClassID ID, Value *V, SBContext &SBCtxt)
    : SBValue(ID, V, SBCtxt) {}

SBUse SBUser::getOperandUseDefault(unsigned OpIdx, bool Verify) const {
  assert((!Verify || OpIdx < getNumOperands()) && "Out of bounds!");
  assert(isa<User>(Val) && "Non-users have no operands!");
  llvm::Use *LLVMUse;
  if (OpIdx != getNumOperands())
    LLVMUse = &cast<User>(Val)->getOperandUse(OpIdx);
  else
    LLVMUse = cast<User>(Val)->op_end();
  return SBUse(LLVMUse, const_cast<SBUser *>(this), Ctxt);
}

#ifndef NDEBUG
void SBUser::verifyUserOfLLVMUse(const Use &Use) const {
  assert(Ctxt.getSBValue(Use.getUser()) == this &&
         "Use not found in this SBUser's operands!");
}
#endif

bool SBUser::classof(const SBValue *From) {
  switch (From->getSubclassID()) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return true;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  default:
    return false;
  }
  return false;
}

SBUser::op_iterator SBUser::op_begin() {
  assert(isa<User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

SBUser::op_iterator SBUser::op_end() {
  assert(isa<User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

SBUser::const_op_iterator SBUser::op_begin() const {
  return const_cast<SBUser *>(this)->op_begin();
}

SBUser::const_op_iterator SBUser::op_end() const {
  return const_cast<SBUser *>(this)->op_end();
}

SBValue *SBUser::getSingleOperand() const {
  assert(getNumOperands() == 1 && "Expected exactly 1 operand");
  return getOperand(0);
}

void SBUser::setOperand(unsigned OperandIdx, SBValue *Operand) {
  if (!isa<User>(Val))
    llvm_unreachable("No operands!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<SetOperand>(this, OperandIdx, Tracker));
  cast<User>(Val)->setOperand(OperandIdx, ValueAttorney::getValue(Operand));
}

bool SBUser::replaceUsesOfWith(SBValue *FromV, SBValue *ToV) {
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(
        std::make_unique<ReplaceUsesOfWith>(this, FromV, ToV, Tracker));

  bool Change = false;
  Value *FromVIR = ValueAttorney::getValue(FromV);
  Value *ToVIR = ValueAttorney::getValue(ToV);
  if (auto *SBI = dyn_cast<SBInstruction>(Ctxt.getSBValue(Val))) {
    for (Instruction *I : SBI->getLLVMInstrs())
      Change |= I->replaceUsesOfWith(FromVIR, ToVIR);
    return Change;
  }
  return cast<User>(Val)->replaceUsesOfWith(FromVIR, ToVIR);
}

#ifndef NDEBUG
void SBUser::dumpCommonHeader(raw_ostream &OS) const {
  SBValue::dumpCommonHeader(OS);
  OS << "(";
  for (auto [OpIdx, Use] : enumerate(operands())) {
    SBValue *Op = Use;
    if (OpIdx != 0)
      OS << ", ";
    if (Op != nullptr)
      OS << Op->getName();
    else
      OS << "<NULL OpN>";
  }
  OS << ")";
}
#endif

SBBBIterator &SBBBIterator::operator++() {
  auto ItE = BB->end();
  assert(It != ItE && "Already at end!");
  ++It;
  if (It == ItE)
    return *this;
  SBInstruction &NextI = *cast<SBInstruction>(SBCtxt->getSBValue(&*It));
  unsigned Num = NextI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  It = std::next(It, Num - 1);
  return *this;
}

SBBBIterator &SBBBIterator::operator--() {
  assert(It != BB->begin() && "Already at begin!");
  if (It == BB->end()) {
    --It;
    return *this;
  }
  SBInstruction &CurrI = **this;
  unsigned Num = CurrI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  assert(std::prev(It, Num - 1) != BB->begin() && "Already at begin!");
  It = std::prev(It, Num);
  return *this;
}

SBInstruction::SBInstruction(ClassID ID, Opcode Opc, Instruction *I,
                             SBContext &SBCtxt)
    : SBUser(ID, I, SBCtxt), Opc(Opc) {
  assert((!isa<StoreInst>(I) || SubclassID == ClassID::Store) &&
         "Create a SBStoreInstruction!");
  assert((!isa<LoadInst>(I) || SubclassID == ClassID::Load) &&
         "Create a SBLoadInstruction!");
  assert((!isa<CastInst>(I) || I->getOpcode() == Instruction::AddrSpaceCast ||
          SubclassID == ClassID::Cast) &&
         "Create a SBCastInstruction!");
  assert((!isa<PHINode>(I) || SubclassID == ClassID::PHI) &&
         "Create a SBPHINode!");
  assert((!isa<CmpInst>(I) || SubclassID == ClassID::Cmp) &&
         "Create a SBCmpInstruction!");
  assert((!isa<SelectInst>(I) || SubclassID == ClassID::Select) &&
         "Create a SBSelectInstruction!");
  assert((!isa<BinaryOperator>(I) || SubclassID == ClassID::BinOp) &&
         "Create a SBBinaryOperator!");
  assert((!isa<UnaryOperator>(I) || SubclassID == ClassID::UnOp) &&
         "Create a SBUnaryOperator!");
}

Instruction *SBInstruction::getTopmostIRInstruction() const {
  SBInstruction *Prev = getPrevNode();
  if (Prev == nullptr) {
    // If at top of the BB, return the first BB instruction.
    return &*cast<BasicBlock>(ValueAttorney::getValue(getParent()))->begin();
  }
  // Else get the Previous SB IR instruction's bottom IR instruction and
  // return its successor.
  Instruction *PrevBotI = cast<Instruction>(ValueAttorney::getValue(Prev));
  return PrevBotI->getNextNode();
}

SBBBIterator SBInstruction::getIterator() const {
  auto *I = cast<Instruction>(Val);
  return SBBasicBlock::iterator(I->getParent(), I->getIterator(), &Ctxt);
}

bool SBBBIterator::atBegin() const {
  // Fast path: if the internal iterator is at begin().
  if (It == BB->begin())
    return true;
  // We may still be at begin if this is a multi-IR SBInstruction and It is
  // pointing to its bottom IR Instr.
  unsigned NumInstrs = getSBI(It)->getNumOfIRInstrs();
  if (NumInstrs == 1)
    // This is a single-IR SBI. Since It != BB->begin() we are not at begin.
    return false;
  return std::prev(It, NumInstrs - 1) == BB->begin();
}

SBInstruction *SBInstruction::getNextNode() const {
  assert(getParent() != nullptr && "Detached!");
  assert(getIterator() != getParent()->end() && "Already at end!");
  auto *CurrI = cast<Instruction>(Val);
  assert(CurrI->getParent() != nullptr && "LLVM IR instr is detached!");
  auto *NextI = CurrI->getNextNode();
  auto *NextSBI = cast_or_null<SBInstruction>(Ctxt.getSBValue(NextI));
  if (NextSBI == nullptr)
    return nullptr;
  return NextSBI;
}

SBInstruction *SBInstruction::getPrevNode() const {
  assert(getParent() != nullptr && "Detached!");
  auto It = getIterator();
  if (!It.atBegin())
    return std::prev(getIterator()).get();
  return nullptr;
}

Instruction::UnaryOps SBInstruction::getIRUnaryOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::FNeg:
    return static_cast<Instruction::UnaryOps>(Instruction::FNeg);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

Instruction::BinaryOps SBInstruction::getIRBinaryOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::Add:
    return static_cast<Instruction::BinaryOps>(Instruction::Add);
  case Opcode::FAdd:
    return static_cast<Instruction::BinaryOps>(Instruction::FAdd);
  case Opcode::Sub:
    return static_cast<Instruction::BinaryOps>(Instruction::Sub);
  case Opcode::FSub:
    return static_cast<Instruction::BinaryOps>(Instruction::FSub);
  case Opcode::Mul:
    return static_cast<Instruction::BinaryOps>(Instruction::Mul);
  case Opcode::FMul:
    return static_cast<Instruction::BinaryOps>(Instruction::FMul);
  case Opcode::UDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::UDiv);
  case Opcode::SDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::SDiv);
  case Opcode::FDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::FDiv);
  case Opcode::URem:
    return static_cast<Instruction::BinaryOps>(Instruction::URem);
  case Opcode::SRem:
    return static_cast<Instruction::BinaryOps>(Instruction::SRem);
  case Opcode::FRem:
    return static_cast<Instruction::BinaryOps>(Instruction::FRem);
  case Opcode::Shl:
    return static_cast<Instruction::BinaryOps>(Instruction::Shl);
  case Opcode::LShr:
    return static_cast<Instruction::BinaryOps>(Instruction::LShr);
  case Opcode::AShr:
    return static_cast<Instruction::BinaryOps>(Instruction::AShr);
  case Opcode::And:
    return static_cast<Instruction::BinaryOps>(Instruction::And);
  case Opcode::Or:
    return static_cast<Instruction::BinaryOps>(Instruction::Or);
  case Opcode::Xor:
    return static_cast<Instruction::BinaryOps>(Instruction::Xor);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

Instruction::CastOps SBInstruction::getIRCastOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::ZExt:
    return static_cast<Instruction::CastOps>(Instruction::ZExt);
  case Opcode::SExt:
    return static_cast<Instruction::CastOps>(Instruction::SExt);
  case Opcode::FPToUI:
    return static_cast<Instruction::CastOps>(Instruction::FPToUI);
  case Opcode::FPToSI:
    return static_cast<Instruction::CastOps>(Instruction::FPToSI);
  case Opcode::FPExt:
    return static_cast<Instruction::CastOps>(Instruction::FPExt);
  case Opcode::PtrToInt:
    return static_cast<Instruction::CastOps>(Instruction::PtrToInt);
  case Opcode::IntToPtr:
    return static_cast<Instruction::CastOps>(Instruction::IntToPtr);
  case Opcode::SIToFP:
    return static_cast<Instruction::CastOps>(Instruction::SIToFP);
  case Opcode::UIToFP:
    return static_cast<Instruction::CastOps>(Instruction::UIToFP);
  case Opcode::Trunc:
    return static_cast<Instruction::CastOps>(Instruction::Trunc);
  case Opcode::FPTrunc:
    return static_cast<Instruction::CastOps>(Instruction::FPTrunc);
  case Opcode::BitCast:
    return static_cast<Instruction::CastOps>(Instruction::BitCast);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

const char *SBInstruction::getOpcodeName(Opcode Opc) {
  switch (Opc) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC)                                                                \
  case Opcode::OPC:                                                            \
    return #OPC;
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  }
}

bool SBInstruction::classof(const SBValue *From) {
  switch (From->getSubclassID()) {
#define DEF_VALUE(ID, CLASS)                                                   \
  case ClassID::ID:                                                            \
    return false;
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return false;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  }
  return false;
}

int64_t SBInstruction::getInstrNumber() const {
  auto *Parent = getParent();
  assert(Parent != nullptr && "Can't get number of a detached instruction!");
  return Parent->getInstrNumber(this);
}

void SBInstruction::removeFromParent() {
  // Update InstrNumberMap
  getParent()->removeInstrNumber(this);
  // Run the callbacks before we unregister it.
  Ctxt.runRemoveInstrCallbacks(this);

  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InstrRemoveFromParent>(this, Tracker));

  // Detach all the IR instructions from their parent BB.
  for (Instruction *I : getLLVMInstrs()) {
    I->removeFromParent();
  }
}

void SBInstruction::eraseFromParent() {
  assert(users().empty() && "Still connected to users, can't erase!");
  auto IRInstrs = getLLVMInstrs();
  // Run the callbacks before we unregister it.
  Ctxt.runRemoveInstrCallbacks(this);
  auto &Tracker = getTracker();

  // Update InstrNumbers
  getParent()->removeInstrNumber(this);

  // Detach from instruction-specific maps.
  auto SBIPtr = getContext().detach(this);

  if (Tracker.tracking()) {
    // Track deletion from IR to SandboxIR maps.
    Tracker.track(
        std::make_unique<EraseFromParent>(std::move(SBIPtr), Ctxt, Tracker));
  } else if (!Tracker.inRevert()) {
    // 1. Regardless of whether we are tracking or not, we should not leak
    //    memory.
    //    So track instrs that got "deleted" such that we actually delete them.
    // 2. This also helps with avoid dangling uses of internal InsertElements of
    //    Packs because we only detach the external facing edges.
    // Note: reverting requires the tables be populated so tracking of the
    // erasing action should happen in this order.
    Tracker.track(std::make_unique<DeleteOnAccept>(this, Tracker));
  }

  if (Tracker.inRevert()) {
    // If this is called by CreateAndInsertInstr::revert() then we should just
    // erase all instructions.
    for (Instruction *I : getLLVMInstrs())
      I->eraseFromParent();
  } else {
    // We don't actually delete the IR instruction, because then it would be
    // impossible to bring it back from the dead at the same memory location.
    // Instead we remove it from its BB and track its current location.
    for (Instruction *I : getLLVMInstrs()) {
      I->removeFromParent();
    }
    for (Instruction *I : getLLVMInstrsWithExternalOperands()) {
      I->dropAllReferences();
    }
  }
}

SBBasicBlock *SBInstruction::getParent() const {
  auto *BB = cast<Instruction>(Val)->getParent();
  if (BB == nullptr)
    return nullptr;
  return Ctxt.getSBBasicBlock(BB);
}

uint64_t SBInstruction::getApproximateDistanceTo(SBInstruction *ToI) const {
  auto FromNum = getInstrNumber();
  auto ToNum = ToI->getInstrNumber();
  return std::abs(ToNum - FromNum) / SBBasicBlock::InstrNumberingStep;
}

void SBInstruction::moveBefore(SBBasicBlock &SBBB,
                               const SBBBIterator &WhereIt) {
  if (std::next(getIterator()) == WhereIt)
    // Destination is same as origin, nothing to do.
    return;
  auto &Tracker = getTracker();
  if (Tracker.tracking()) {
    Tracker.track(std::make_unique<MoveInstr>(this, Tracker));
  }
  Ctxt.runMoveInstrCallbacks(this, SBBB, WhereIt);

  auto *BB = cast<BasicBlock>(ValueAttorney::getValue(&SBBB));
  BasicBlock::iterator It;
  if (WhereIt == SBBB.end())
    It = BB->end();
  else {
    SBInstruction *WhereI = &*WhereIt;
    It = WhereI->getTopmostIRInstruction()->getIterator();
  }
  auto IRInstrsInProgramOrder(getLLVMInstrs());
  sort(IRInstrsInProgramOrder,
       [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  // Update instruction numbering (part 1)
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  // Do the actual move in LLVM IR.
  for (auto *I : IRInstrsInProgramOrder)
    I->moveBefore(*BB, It);
  // Update instruction numbering (part 2)
  SBBB.assignInstrNumber(this);
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  Ctxt.afterMoveInstrHook(SBBB);
#endif
}

void SBInstruction::insertBefore(SBInstruction *BeforeI) {
  // Maintain instruction numbering
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(BeforeI, getParent(), Tracker));
  Instruction *BeforeTopI = BeforeI->getTopmostIRInstruction();
  auto IRInstrs = getLLVMInstrs();
  for (Instruction *I : reverse(IRInstrs))
    I->insertBefore(BeforeTopI);
  // Update instruction numbering.
  BeforeI->getParent()->assignInstrNumber(this);
}

void SBInstruction::insertAfter(SBInstruction *AfterI) {
  insertInto(AfterI->getParent(), std::next(AfterI->getIterator()));
}

void SBInstruction::insertInto(SBBasicBlock *SBBB,
                               const SBBBIterator &WhereIt) {
  // Maintain instruction numbering
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  BasicBlock *BB = cast<BasicBlock>(ValueAttorney::getValue(SBBB));
  Instruction *BeforeI;
  SBInstruction *SBBeforeI;
  BasicBlock::iterator BeforeIt;
  if (WhereIt != SBBB->end()) {
    SBBeforeI = &*WhereIt;
    BeforeI = SBBeforeI->getTopmostIRInstruction();
    BeforeIt = BeforeI->getIterator();
  } else {
    SBBeforeI = nullptr;
    BeforeI = nullptr;
    BeforeIt = BB->end();
  }
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(SBBeforeI, SBBB, Tracker));
  auto IRInstrs = getLLVMInstrs();
  for (Instruction *I : reverse(IRInstrs))
    I->insertInto(BB, BeforeIt);
  // Update instruction numbering.
  SBBB->assignInstrNumber(this);
}

SBBasicBlock *SBInstruction::getSuccessor(unsigned Idx) const {
  return cast<SBBasicBlock>(
      Ctxt.getSBValue(cast<Instruction>(Val)->getSuccessor(Idx)));
}

#ifndef NDEBUG
void SBInstruction::dump(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dump().";
}
void SBInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBInstruction::dumpVerbose(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dumpVerbose().";
}
void SBInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

SBConstant::SBConstant(Constant *C, SBContext &SBCtxt)
    : SBUser(ClassID::Constant, C, SBCtxt) {}

bool SBConstant::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Constant ||
         From->getSubclassID() == ClassID::Function;
}
#ifndef NDEBUG
void SBConstant::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBConstant::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBConstant::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBConstant::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

bool SBArgument::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Argument;
}

#ifndef NDEBUG
void SBArgument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void SBArgument::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void SBArgument::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBArgument::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBArgument::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

SBValue *SBCmpInstruction::create(CmpInst::Predicate Pred, SBValue *LHS,
                                  SBValue *RHS, SBInstruction *InsertBefore,
                                  SBContext &SBCtxt, const Twine &Name,
                                  MDNode *FPMathTag) {
  Value *LHSIR = ValueAttorney::getValue(LHS);
  Value *RHSIR = ValueAttorney::getValue(RHS);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<CmpInst>(NewV))
    return SBCtxt.createSBCmpInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBCmpInstruction::create(CmpInst::Predicate Pred, SBValue *LHS,
                                  SBValue *RHS, SBBasicBlock *InsertAtEnd,
                                  SBContext &SBCtxt, const Twine &Name,
                                  MDNode *FPMathTag) {
  Value *LHSIR = ValueAttorney::getValue(LHS);
  Value *RHSIR = ValueAttorney::getValue(RHS);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<CmpInst>(NewV))
    return SBCtxt.createSBCmpInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBCmpInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Cmp;
}

#ifndef NDEBUG
void SBCmpInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBCmpInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBCmpInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBCmpInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBStoreInstruction *SBStoreInstruction::create(SBValue *V, SBValue *Ptr,
                                               MaybeAlign Align,
                                               SBInstruction *InsertBefore,
                                               SBContext &SBCtxt) {
  Value *ValIR = ValueAttorney::getValue(V);
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = SBCtxt.createSBStoreInstruction(NewSI);
  return NewSBI;
}
SBStoreInstruction *SBStoreInstruction::create(SBValue *V, SBValue *Ptr,
                                               MaybeAlign Align,
                                               SBBasicBlock *InsertAtEnd,
                                               SBContext &SBCtxt) {
  Value *ValIR = ValueAttorney::getValue(V);
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = SBCtxt.createSBStoreInstruction(NewSI);
  return NewSBI;
}

bool SBStoreInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Store;
}

SBValue *SBStoreInstruction::getValueOperand() const {
  return Ctxt.getSBValue(cast<StoreInst>(Val)->getValueOperand());
}

SBValue *SBStoreInstruction::getPointerOperand() const {
  return Ctxt.getSBValue(cast<StoreInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void SBStoreInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBStoreInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBStoreInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBStoreInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBLoadInstruction *SBLoadInstruction::create(Type *Ty, SBValue *Ptr,
                                             MaybeAlign Align,
                                             SBInstruction *InsertBefore,
                                             SBContext &SBCtxt,
                                             const Twine &Name) {
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = SBCtxt.createSBLoadInstruction(NewLI);
  return NewSBI;
}

SBLoadInstruction *SBLoadInstruction::create(Type *Ty, SBValue *Ptr,
                                             MaybeAlign Align,
                                             SBBasicBlock *InsertAtEnd,
                                             SBContext &SBCtxt,
                                             const Twine &Name) {
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = SBCtxt.createSBLoadInstruction(NewLI);
  return NewSBI;
}

bool SBLoadInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Load;
}

SBValue *SBLoadInstruction::getPointerOperand() const {
  return Ctxt.getSBValue(cast<LoadInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void SBLoadInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBLoadInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBLoadInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBLoadInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBCastInstruction::create(Type *Ty, Opcode Op, SBValue *Operand,
                                   SBInstruction *InsertBefore,
                                   SBContext &SBCtxt, const Twine &Name) {
  Value *IROperand = ValueAttorney::getValue(Operand);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<CastInst>(NewV))
    return SBCtxt.createSBCastInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBCastInstruction::create(Type *Ty, Opcode Op, SBValue *Operand,
                                   SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                                   const Twine &Name) {
  Value *IROperand = ValueAttorney::getValue(Operand);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<CastInst>(NewV))
    return SBCtxt.createSBCastInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBCastInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Cast;
}

#ifndef NDEBUG
void SBCastInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBCastInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBCastInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBCastInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBPHINode::create(Type *Ty, unsigned NumReservedValues,
                           SBInstruction *InsertBefore, SBContext &SBCtxt,
                           const Twine &Name) {
  Instruction *InsertBeforeIR = InsertBefore->getTopmostIRInstruction();
  PHINode *NewPHI =
      PHINode::Create(Ty, NumReservedValues, Name, InsertBeforeIR);
  return SBCtxt.createSBPHINode(NewPHI);
}

SBValue *SBPHINode::create(Type *Ty, unsigned NumReservedValues,
                           SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                           const Twine &Name) {
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  PHINode *NewPHI = PHINode::Create(Ty, NumReservedValues, Name, InsertAtEndIR);
  return SBCtxt.createSBPHINode(NewPHI);
}

bool SBPHINode::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::PHI;
}

#ifndef NDEBUG
void SBPHINode::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBPHINode::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBPHINode::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBPHINode::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBSelectInstruction::create(SBValue *Cond, SBValue *True,
                                     SBValue *False,
                                     SBInstruction *InsertBefore,
                                     SBContext &SBCtxt, const Twine &Name) {
  Value *IRCond = ValueAttorney::getValue(Cond);
  Value *IRTrue = ValueAttorney::getValue(True);
  Value *IRFalse = ValueAttorney::getValue(False);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<SelectInst>(NewV))
    return SBCtxt.createSBSelectInstruction(NewSI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBSelectInstruction::create(SBValue *Cond, SBValue *True,
                                     SBValue *False, SBBasicBlock *InsertAtEnd,
                                     SBContext &SBCtxt, const Twine &Name) {
  Value *IRCond = ValueAttorney::getValue(Cond);
  Value *IRTrue = ValueAttorney::getValue(True);
  Value *IRFalse = ValueAttorney::getValue(False);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<SelectInst>(NewV))
    return SBCtxt.createSBSelectInstruction(NewSI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBSelectInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Select;
}

#ifndef NDEBUG
void SBSelectInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBSelectInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBSelectInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBSelectInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBBinaryOperator::create(SBInstruction::Opcode Op, SBValue *LHS,
                                  SBValue *RHS, SBInstruction *InsertBefore,
                                  SBContext &SBCtxt, const Twine &Name) {
  Value *IRLHS = ValueAttorney::getValue(LHS);
  Value *IRRHS = ValueAttorney::getValue(RHS);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewV)) {
    return SBCtxt.createSBBinaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBBinaryOperator::create(SBInstruction::Opcode Op, SBValue *LHS,
                                  SBValue *RHS, SBBasicBlock *InsertAtEnd,
                                  SBContext &SBCtxt, const Twine &Name) {
  Value *IRLHS = ValueAttorney::getValue(LHS);
  Value *IRRHS = ValueAttorney::getValue(RHS);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewV)) {
    return SBCtxt.createSBBinaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBBinaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS, SBValue *CopyFrom,
    SBInstruction *InsertBefore, SBContext &SBCtxt, const Twine &Name) {
  SBValue *NewV = create(Op, LHS, RHS, InsertBefore, SBCtxt, Name);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  if (isa<SBBinaryOperator>(NewV))
    cast<BinaryOperator>(ValueAttorney::getValue(NewV))
        ->copyIRFlags(IRCopyFrom);
  return NewV;
}

SBValue *SBBinaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS, SBValue *CopyFrom,
    SBBasicBlock *InsertAtEnd, SBContext &SBCtxt, const Twine &Name) {
  SBValue *NewV = create(Op, LHS, RHS, InsertAtEnd, SBCtxt, Name);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  if (isa<SBBinaryOperator>(NewV))
    cast<BinaryOperator>(ValueAttorney::getValue(NewV))
        ->copyIRFlags(IRCopyFrom);
  return NewV;
}

bool SBBinaryOperator::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::BinOp;
}

#ifndef NDEBUG
void SBBinaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBBinaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBBinaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBBinaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBUnaryOperator::createWithCopiedFlags(SBInstruction::Opcode Op,
                                                SBValue *OpV, SBValue *CopyFrom,
                                                SBInstruction *InsertBefore,
                                                SBContext &SBCtxt,
                                                const Twine &Name) {
  Value *IROpV = ValueAttorney::getValue(OpV);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBUnaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBUnaryOperator::createWithCopiedFlags(SBInstruction::Opcode Op,
                                                SBValue *OpV, SBValue *CopyFrom,
                                                SBBasicBlock *InsertAtEnd,
                                                SBContext &SBCtxt,
                                                const Twine &Name) {
  Value *IROpV = ValueAttorney::getValue(OpV);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBUnaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBUnaryOperator::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::UnOp;
}

#ifndef NDEBUG
void SBUnaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBUnaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBUnaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBUnaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBReturnInstruction::create(SBValue *RetVal,
                                     SBInstruction *InsertBefore,
                                     SBContext &SBCtxt) {
  Value *LLVMRet = ValueAttorney::getValue(RetVal);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  ReturnInst *NewV;
  if (RetVal != nullptr)
    NewV = Builder.CreateRet(LLVMRet);
  else
    NewV = Builder.CreateRetVoid();
  if (auto *NewRI = dyn_cast<ReturnInst>(NewV))
    return SBCtxt.createSBReturnInstruction(NewRI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBReturnInstruction::create(SBValue *RetVal, SBBasicBlock *InsertAtEnd,
                                     SBContext &SBCtxt) {
  Value *LLVMRet =
      RetVal != nullptr ? ValueAttorney::getValue(RetVal) : nullptr;
  BasicBlock *LLVMInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  ReturnInst *NewV;
  if (RetVal != nullptr)
    NewV = Builder.CreateRet(LLVMRet);
  else
    NewV = Builder.CreateRetVoid();
  if (auto *NewRI = dyn_cast<ReturnInst>(NewV))
    return SBCtxt.createSBReturnInstruction(NewRI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBReturnInstruction::getReturnValue() const {
  auto *LLVMRetVal = cast<ReturnInst>(Val)->getReturnValue();
  return LLVMRetVal != nullptr ? Ctxt.getSBValue(LLVMRetVal) : nullptr;
}

#ifndef NDEBUG
void SBReturnInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBReturnInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBReturnInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBReturnInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBCallInstruction *SBCallInstruction::create(FunctionType *FTy, SBValue *Func,
                                             ArrayRef<SBValue *> Args,
                                             SBBasicBlock::iterator WhereIt,
                                             SBBasicBlock *WhereBB,
                                             SBContext &SBCtxt,
                                             const Twine &NameStr) {
  Value *LLVMFunc = ValueAttorney::getValue(Func);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostIRInstruction());
  else
    Builder.SetInsertPoint(SBBasicBlockAttorney::getBB(WhereBB));
  SmallVector<Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (SBValue *Arg : Args)
    LLVMArgs.push_back(ValueAttorney::getValue(Arg));
  CallInst *NewCI = Builder.CreateCall(FTy, LLVMFunc, LLVMArgs, NameStr);
  return SBCtxt.createSBCallInstruction(NewCI);
}

SBCallInstruction *SBCallInstruction::create(FunctionType *FTy, SBValue *Func,
                                             ArrayRef<SBValue *> Args,
                                             SBInstruction *InsertBefore,
                                             SBContext &SBCtxt,
                                             const Twine &NameStr) {
  return SBCallInstruction::create(FTy, Func, Args, InsertBefore->getIterator(),
                                   InsertBefore->getParent(), SBCtxt, NameStr);
}

SBCallInstruction *SBCallInstruction::create(FunctionType *FTy, SBValue *Func,
                                             ArrayRef<SBValue *> Args,
                                             SBBasicBlock *InsertAtEnd,
                                             SBContext &SBCtxt,
                                             const Twine &NameStr) {
  return SBCallInstruction::create(FTy, Func, Args, InsertAtEnd->end(),
                                   InsertAtEnd, SBCtxt, NameStr);
}

#ifndef NDEBUG
void SBCallInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBCallInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBCallInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBCallInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBGetElementPtrInstruction::create(Type *Ty, SBValue *Ptr,
                                            ArrayRef<SBValue *> IdxList,
                                            SBBasicBlock::iterator WhereIt,
                                            SBBasicBlock *WhereBB,
                                            SBContext &SBCtxt,
                                            const Twine &NameStr) {
  Value *LLVMPtr = ValueAttorney::getValue(Ptr);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostIRInstruction());
  else
    Builder.SetInsertPoint(SBBasicBlockAttorney::getBB(WhereBB));
  SmallVector<Value *> LLVMIdxList;
  LLVMIdxList.reserve(IdxList.size());
  for (SBValue *Idx : IdxList)
    LLVMIdxList.push_back(ValueAttorney::getValue(Idx));
  Value *NewV = Builder.CreateGEP(Ty, LLVMPtr, LLVMIdxList, NameStr);
  if (auto *NewGEP = dyn_cast<GetElementPtrInst>(NewV))
    return SBCtxt.createSBGetElementPtrInstruction(NewGEP);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBGetElementPtrInstruction::create(Type *Ty, SBValue *Ptr,
                                            ArrayRef<SBValue *> IdxList,
                                            SBInstruction *InsertBefore,
                                            SBContext &SBCtxt,
                                            const Twine &NameStr) {
  return SBGetElementPtrInstruction::create(
      Ty, Ptr, IdxList, InsertBefore->getIterator(), InsertBefore->getParent(),
      SBCtxt, NameStr);
}

SBValue *SBGetElementPtrInstruction::create(Type *Ty, SBValue *Ptr,
                                            ArrayRef<SBValue *> IdxList,
                                            SBBasicBlock *InsertAtEnd,
                                            SBContext &SBCtxt,
                                            const Twine &NameStr) {
  return SBGetElementPtrInstruction::create(
      Ty, Ptr, IdxList, InsertAtEnd->end(), InsertAtEnd, SBCtxt, NameStr);
}

#ifndef NDEBUG
void SBGetElementPtrInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBGetElementPtrInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBGetElementPtrInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBGetElementPtrInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

#ifndef NDEBUG
void SBOpaqueInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBOpaqueInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBOpaqueInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBOpaqueInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBInsertElementInstruction::create(SBValue *Vec, SBValue *NewElt,
                                            SBValue *Idx,
                                            SBInstruction *InsertBefore,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMVec = ValueAttorney::getValue(Vec);
  Value *LLVMNewElt = ValueAttorney::getValue(NewElt);
  Value *LLVMIdx = ValueAttorney::getValue(Idx);
  Instruction *LLVMInsertBefore =
      cast<Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  Value *NewV = Builder.CreateInsertElement(LLVMVec, LLVMNewElt, LLVMIdx, Name);
  if (auto *NewInsert = dyn_cast<InsertElementInst>(NewV))
    return SBCtxt.createSBInsertElementInstruction(NewInsert);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBInsertElementInstruction::create(SBValue *Vec, SBValue *NewElt,
                                            SBValue *Idx,
                                            SBBasicBlock *InsertAtEnd,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMVec = ValueAttorney::getValue(Vec);
  Value *LLVMNewElt = ValueAttorney::getValue(NewElt);
  Value *LLVMIdx = ValueAttorney::getValue(Idx);
  BasicBlock *LLVMInsertAtEnd =
      cast<BasicBlock>(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  Value *NewV = Builder.CreateInsertElement(LLVMVec, LLVMNewElt, LLVMIdx, Name);
  if (auto *NewInsert = dyn_cast<InsertElementInst>(NewV))
    return SBCtxt.createSBInsertElementInstruction(NewInsert);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

#ifndef NDEBUG
void SBInsertElementInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBInsertElementInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBInsertElementInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBInsertElementInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBExtractElementInstruction::create(SBValue *Vec, SBValue *Idx,
                                             SBInstruction *InsertBefore,
                                             SBContext &SBCtxt,
                                             const Twine &Name) {
  Value *LLVMVec = ValueAttorney::getValue(Vec);
  Value *LLVMIdx = ValueAttorney::getValue(Idx);
  Instruction *LLVMInsertBefore =
      cast<Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  Value *NewV = Builder.CreateExtractElement(LLVMVec, LLVMIdx, Name);
  if (auto *NewExtract = dyn_cast<ExtractElementInst>(NewV))
    return SBCtxt.createSBExtractElementInstruction(NewExtract);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBExtractElementInstruction::create(SBValue *Vec, SBValue *Idx,
                                             SBBasicBlock *InsertAtEnd,
                                             SBContext &SBCtxt,
                                             const Twine &Name) {
  Value *LLVMVec = ValueAttorney::getValue(Vec);
  Value *LLVMIdx = ValueAttorney::getValue(Idx);
  BasicBlock *LLVMInsertAtEnd =
      cast<BasicBlock>(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  Value *NewV = Builder.CreateExtractElement(LLVMVec, LLVMIdx, Name);
  if (auto *NewExtract = dyn_cast<ExtractElementInst>(NewV))
    return SBCtxt.createSBExtractElementInstruction(NewExtract);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

#ifndef NDEBUG
void SBExtractElementInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBExtractElementInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBExtractElementInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBExtractElementInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBShuffleVectorInstruction::create(SBValue *V1, SBValue *V2,
                                            SBValue *Mask,
                                            SBInstruction *InsertBefore,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMV1 = ValueAttorney::getValue(V1);
  Value *LLVMV2 = ValueAttorney::getValue(V2);
  Value *LLVMMask = ValueAttorney::getValue(Mask);
  Instruction *LLVMInsertBefore =
      cast<Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, LLVMMask, Name);
  if (auto *NewShuffleVec = dyn_cast<ShuffleVectorInst>(NewV))
    return SBCtxt.createSBShuffleVectorInstruction(NewShuffleVec);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBShuffleVectorInstruction::create(SBValue *V1, SBValue *V2,
                                            SBValue *Mask,
                                            SBBasicBlock *InsertAtEnd,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMV1 = ValueAttorney::getValue(V1);
  Value *LLVMV2 = ValueAttorney::getValue(V2);
  Value *LLVMMask = ValueAttorney::getValue(Mask);
  BasicBlock *LLVMInsertAtEnd =
      cast<BasicBlock>(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, LLVMMask, Name);
  if (auto *NewShuffleVec = dyn_cast<ShuffleVectorInst>(NewV))
    return SBCtxt.createSBShuffleVectorInstruction(NewShuffleVec);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBShuffleVectorInstruction::create(SBValue *V1, SBValue *V2,
                                            ArrayRef<int> Mask,
                                            SBInstruction *InsertBefore,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMV1 = ValueAttorney::getValue(V1);
  Value *LLVMV2 = ValueAttorney::getValue(V2);
  Instruction *LLVMInsertBefore =
      cast<Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, Mask, Name);
  if (auto *NewShuffleVec = dyn_cast<ShuffleVectorInst>(NewV))
    return SBCtxt.createSBShuffleVectorInstruction(NewShuffleVec);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBShuffleVectorInstruction::create(SBValue *V1, SBValue *V2,
                                            ArrayRef<int> Mask,
                                            SBBasicBlock *InsertAtEnd,
                                            SBContext &SBCtxt,
                                            const Twine &Name) {
  Value *LLVMV1 = ValueAttorney::getValue(V1);
  Value *LLVMV2 = ValueAttorney::getValue(V2);
  BasicBlock *LLVMInsertAtEnd =
      cast<BasicBlock>(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, Mask, Name);
  if (auto *NewShuffleVec = dyn_cast<ShuffleVectorInst>(NewV))
    return SBCtxt.createSBShuffleVectorInstruction(NewShuffleVec);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

#ifndef NDEBUG
void SBShuffleVectorInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBShuffleVectorInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBShuffleVectorInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBShuffleVectorInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void SBFunction::detachFromLLVMIR() {
  for (SBBasicBlock &SBBB : *this)
    SBBB.detachFromLLVMIR();
  // Detach the actual SBFunction.
  Ctxt.detach(this);
}

#ifndef NDEBUG
void SBFunction::dumpNameAndArgs(raw_ostream &OS) const {
  Function *F = getFunction();
  OS << *getType() << " @" << F->getName() << "(";
  auto NumArgs = F->arg_size();
  for (auto [Idx, Arg] : enumerate(F->args())) {
    auto *SBArg = cast_or_null<SBArgument>(Ctxt.getSBValue(&Arg));
    if (SBArg == nullptr)
      OS << "NULL";
    else
      SBArg->printAsOperand(OS);
    if (Idx + 1 < NumArgs)
      OS << ", ";
  }
  OS << ")";
}
void SBFunction::dump(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  Function *F = getFunction();
  BasicBlock &LastBB = F->back();
  for (BasicBlock &BB : *F) {
    auto *SBBB = cast_or_null<SBBasicBlock>(Ctxt.getSBValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      OS << *SBBB;
    if (&BB != &LastBB)
      OS << "\n";
  }
  OS << "}\n";
}
void SBFunction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBFunction::dumpVerbose(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  for (BasicBlock &BB : *getFunction()) {
    auto *SBBB = cast_or_null<SBBasicBlock>(Ctxt.getSBValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      SBBB->dumpVerbose(OS);
    OS << "\n";
  }
  OS << "}\n";
}
void SBFunction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

SBContext::SBContext(LLVMContext &LLVMCtxt)
    : LLVMCtxt(LLVMCtxt), LLVMIRBuilder(LLVMCtxt, ConstantFolder()) {}

std::unique_ptr<SBValue> SBContext::detachValue(Value *V) {
  std::unique_ptr<SBValue> Erased;
  auto It = LLVMValueToSBValueMap.find(V);
  if (It != LLVMValueToSBValueMap.end()) {
    auto *Val = It->second.release();
    Erased = std::unique_ptr<SBValue>(Val);
    LLVMValueToSBValueMap.erase(It);
  }
  MultiInstrMap.erase(V);
  return Erased;
}

SBValue *SBContext::getSBValue(Value *V) const {
  // In the common case we should find the value in LLVMValueToSBValueMap.
  auto It = LLVMValueToSBValueMap.find(V);
  if (It != LLVMValueToSBValueMap.end())
    return It->second.get();
  // Instrs that map to multiple IR Instrs (like Packs) use a second map.
  auto It2 = MultiInstrMap.find(V);
  if (It2 != MultiInstrMap.end()) {
    Value *Key = It2->second;
    assert(Key != V && "Bad entry in MultiInstrMap!");
    return getSBValue(Key);
  }
  return nullptr;
}

SBConstant *SBContext::getSBConstant(Constant *C) const {
  return cast_or_null<SBConstant>(getSBValue(C));
}

SBConstant *SBContext::getOrCreateSBConstant(Constant *C) {
  auto Pair = LLVMValueToSBValueMap.insert({C, nullptr});
  auto It = Pair.first;
  if (Pair.second) {
    It->second = std::unique_ptr<SBConstant>(new SBConstant(C, *this));
    return cast<SBConstant>(It->second.get());
  }
  return cast<SBConstant>(It->second.get());
}

std::unique_ptr<SBValue> SBContext::detach(SBValue *SBV) {
  if (auto *SBI = dyn_cast<SBInstruction>(SBV))
    SBI->detachExtras();
#ifndef NDEBUG
  switch (SBV->getSubclassID()) {
  case SBValue::ClassID::Constant:
    llvm_unreachable("Can't detach a constant!");
    break;
  case SBValue::ClassID::User:
    llvm_unreachable("Can't detach a user!");
    break;
  default:
    break;
  }
#endif
  Value *V = ValueAttorney::getValue(SBV);
  return detachValue(V);
}

SBValue *SBContext::registerSBValue(std::unique_ptr<SBValue> &&SBVPtr) {
  auto &Tracker = getTracker();
  if (Tracker.tracking() && isa<SBInstruction>(SBVPtr.get()))
    Tracker.track(std::make_unique<CreateAndInsertInstr>(
        cast<SBInstruction>(SBVPtr.get()), Tracker));

  assert(SBVPtr->getSubclassID() != SBValue::ClassID::User &&
         "Can't register a user!");
  SBValue *V = SBVPtr.get();
  Value *Key = ValueAttorney::getValue(V);
  LLVMValueToSBValueMap[Key] = std::move(SBVPtr);
  // For multi-LLVM-Instruction SBInstructrions we also need to map the rest.
  if (auto *I = dyn_cast<SBInstruction>(V)) {
    auto LLVMInstrs = I->getLLVMInstrs();
    for (auto *InternalI : drop_begin(LLVMInstrs))
      MultiInstrMap[InternalI] = Key;

    if (!DontNumberInstrs)
      // Number the instruction
      I->getParent()->assignInstrNumber(I);
    runInsertInstrCallbacks(I);
  }
  return V;
}

void SBContext::createMissingConstantOperands(Value *V) {
  // Create SandboxIR for all new constant operands.
  if (User *U = dyn_cast<User>(V)) {
    for (Value *Op : U->operands()) {
      if (auto *ConstOp = dyn_cast<Constant>(Op))
        getOrCreateSBConstant(ConstOp);
    }
  }
}

// Store
SBStoreInstruction *SBContext::getSBStoreInstruction(StoreInst *SI) const {
  return cast_or_null<SBStoreInstruction>(getSBValue(SI));
}

SBStoreInstruction *SBContext::createSBStoreInstruction(StoreInst *SI) {
  assert(SI->getParent() != nullptr && "Detached!");
  assert(getSBStoreInstruction(SI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBStoreInstruction>(new SBStoreInstruction(SI, *this));
  return cast<SBStoreInstruction>(registerSBValue(std::move(NewPtr)));
}

SBStoreInstruction *SBContext::getOrCreateSBStoreInstruction(StoreInst *SI) {
  if (auto *SBSI = getSBStoreInstruction(SI))
    return SBSI;
  return createSBStoreInstruction(SI);
}

// Load
SBLoadInstruction *SBContext::getSBLoadInstruction(LoadInst *LI) const {
  return cast_or_null<SBLoadInstruction>(getSBValue(LI));
}

SBLoadInstruction *SBContext::createSBLoadInstruction(LoadInst *LI) {
  assert(getSBLoadInstruction(LI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBLoadInstruction>(new SBLoadInstruction(LI, *this));
  return cast<SBLoadInstruction>(registerSBValue(std::move(NewPtr)));
}

SBLoadInstruction *SBContext::getOrCreateSBLoadInstruction(LoadInst *LI) {
  if (auto *SBLI = getSBLoadInstruction(LI))
    return SBLI;
  return createSBLoadInstruction(LI);
}

// Cast
SBCastInstruction *SBContext::getSBCastInstruction(CastInst *CI) const {
  return cast_or_null<SBCastInstruction>(getSBValue(CI));
}

SBCastInstruction *SBContext::createSBCastInstruction(CastInst *CI) {
  assert(getSBCastInstruction(CI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBCastInstruction>(new SBCastInstruction(CI, *this));
  return cast<SBCastInstruction>(registerSBValue(std::move(NewPtr)));
}

SBCastInstruction *SBContext::getOrCreateSBCastInstruction(CastInst *CI) {
  if (auto *SBCI = getSBCastInstruction(CI))
    return SBCI;
  return createSBCastInstruction(CI);
}

// PHI
SBPHINode *SBContext::getSBPHINode(PHINode *PHI) const {
  return cast_or_null<SBPHINode>(getSBValue(PHI));
}

SBPHINode *SBContext::createSBPHINode(PHINode *PHI) {
  assert(getSBPHINode(PHI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBPHINode>(new SBPHINode(PHI, *this));
  return cast<SBPHINode>(registerSBValue(std::move(NewPtr)));
}

SBPHINode *SBContext::getOrCreateSBPHINode(PHINode *PHI) {
  if (auto *SBPHI = getSBPHINode(PHI))
    return SBPHI;
  return createSBPHINode(PHI);
}

// Select
SBSelectInstruction *SBContext::getSBSelectInstruction(SelectInst *SI) const {
  return cast_or_null<SBSelectInstruction>(getSBValue(SI));
}

SBSelectInstruction *SBContext::createSBSelectInstruction(SelectInst *SI) {
  assert(getSBSelectInstruction(SI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBSelectInstruction>(new SBSelectInstruction(SI, *this));
  return cast<SBSelectInstruction>(registerSBValue(std::move(NewPtr)));
}

SBSelectInstruction *SBContext::getOrCreateSBSelectInstruction(SelectInst *SI) {
  if (auto *SBSI = getSBSelectInstruction(SI))
    return SBSI;
  return createSBSelectInstruction(SI);
}

// BinaryOperator
SBBinaryOperator *SBContext::getSBBinaryOperator(BinaryOperator *BO) const {
  return cast_or_null<SBBinaryOperator>(getSBValue(BO));
}

SBBinaryOperator *SBContext::createSBBinaryOperator(BinaryOperator *BO) {
  assert(getSBBinaryOperator(BO) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBBinaryOperator>(new SBBinaryOperator(BO, *this));
  return cast<SBBinaryOperator>(registerSBValue(std::move(NewPtr)));
}

SBBinaryOperator *SBContext::getOrCreateSBBinaryOperator(BinaryOperator *BO) {
  if (auto *SBBO = getSBBinaryOperator(BO))
    return SBBO;
  return createSBBinaryOperator(BO);
}

// UnaryOperator
SBUnaryOperator *SBContext::getSBUnaryOperator(UnaryOperator *UO) const {
  return cast_or_null<SBUnaryOperator>(getSBValue(UO));
}

SBUnaryOperator *SBContext::createSBUnaryOperator(UnaryOperator *UO) {
  assert(getSBUnaryOperator(UO) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBUnaryOperator>(new SBUnaryOperator(UO, *this));
  return cast<SBUnaryOperator>(registerSBValue(std::move(NewPtr)));
}

SBUnaryOperator *SBContext::getOrCreateSBUnaryOperator(UnaryOperator *UO) {
  if (auto *SBUO = getSBUnaryOperator(UO))
    return SBUO;
  return createSBUnaryOperator(UO);
}

// Cmp
SBCmpInstruction *SBContext::getSBCmpInstruction(CmpInst *CI) const {
  return cast_or_null<SBCmpInstruction>(getSBValue(CI));
}

SBCmpInstruction *SBContext::createSBCmpInstruction(CmpInst *CI) {
  assert(getSBCmpInstruction(CI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBCmpInstruction>(new SBCmpInstruction(CI, *this));
  return cast<SBCmpInstruction>(registerSBValue(std::move(NewPtr)));
}

SBCmpInstruction *SBContext::getOrCreateSBCmpInstruction(CmpInst *CI) {
  if (auto *SBCI = getSBCmpInstruction(CI))
    return SBCI;
  return createSBCmpInstruction(CI);
}

SBValue *SBContext::getOrCreateSBValue(Value *V) {
  return getOrCreateSBValueInternal(V, 0);
}

SBValue *SBContext::getOrCreateSBValueInternal(Value *V, int Depth, User *U) {
  assert(Depth < 666 && "Infinite recursion?");
  // TODO: Use switch-case with subclass IDs instead of `if`.
  if (auto *C = dyn_cast<Constant>(V)) {
    // Globals may be self-referencing, like @bar = global [1 x ptr] [ptr @bar].
    // Avoid infinite loops by early returning once we detect a loop.
    if (isa<GlobalValue>(C)) {
      if (Depth == 0)
        VisitedConstants.clear();
      if (!VisitedConstants.insert(C).second)
        return nullptr; //  recursion loop!
    }
    for (Value *COp : C->operands())
      getOrCreateSBValueInternal(COp, Depth + 1, C);
    return getOrCreateSBConstant(C);
  }
  if (auto *Arg = dyn_cast<Argument>(V)) {
    return getOrCreateSBArgument(Arg);
  }
  if (auto *BB = dyn_cast<BasicBlock>(V)) {
    assert(isa<BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getSBValue(BB))
      return SBBB;
    // TODO: return a SBOpaqueValue
    return nullptr;
  }
  assert(isa<Instruction>(V) && "Expected Instruction");
  switch (cast<Instruction>(V)->getOpcode()) {
  case Instruction::PHI:
    return getOrCreateSBPHINode(cast<PHINode>(V));
  case Instruction::ExtractElement:
    return getOrCreateSBValueFromExtractElement(cast<ExtractElementInst>(V),
                                                Depth);
  case Instruction::ExtractValue:
    goto opaque_label;
  case Instruction::InsertElement:
    return getOrCreateSBValueFromInsertElement(cast<InsertElementInst>(V),
                                               Depth);
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast:
    return getOrCreateSBCastInstruction(cast<CastInst>(V));
  case Instruction::FCmp:
  case Instruction::ICmp:
    return getOrCreateSBCmpInstruction(cast<CmpInst>(V));
  case Instruction::Select:
    return getOrCreateSBSelectInstruction(cast<SelectInst>(V));
  case Instruction::FNeg:
    return getOrCreateSBUnaryOperator(cast<UnaryOperator>(V));
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return getOrCreateSBBinaryOperator(cast<BinaryOperator>(V));
  case Instruction::Load:
    return getOrCreateSBLoadInstruction(cast<LoadInst>(V));
  case Instruction::Store:
    return getOrCreateSBStoreInstruction(cast<StoreInst>(V));
  case Instruction::GetElementPtr:
    return getOrCreateSBGetElementPtrInstruction(cast<GetElementPtrInst>(V));
  case Instruction::Call:
    return getOrCreateSBCallInstruction(cast<CallInst>(V));
  case Instruction::ShuffleVector:
    return getOrCreateSBValueFromShuffleVector(cast<ShuffleVectorInst>(V),
                                               Depth);
  case Instruction::Ret:
    return getOrCreateSBReturnInstruction(cast<ReturnInst>(V));
  default:
  opaque_label:
    return getOrCreateSBOpaqueInstruction(cast<Instruction>(V));
  }
}

void SBBasicBlock::renumberInstructions() {
  int64_t Num = 0;
  for (SBInstruction &IRef : *this) {
    InstrNumberMap[&IRef] = Num;
    Num += InstrNumberingStep;
  }
}

void SBBasicBlock::assignInstrNumber(SBInstruction *I) {
  int64_t Num;
  assert(I->getParent() && "Expected a parent block!");
  if (I->getNextNode() == nullptr) {
    // Inserting at the end of the block.
    if (empty())
      Num = 0;
    else {
      assert(I->getPrevNode() != nullptr && "Handle by empty()");
      auto PrevNum = getInstrNumber(I->getPrevNode());
      assert(PrevNum < std::numeric_limits<decltype(PrevNum)>::max() -
                           InstrNumberingStep &&
             "You're gonna need a bigger boat!");
      Num = PrevNum + InstrNumberingStep;
    }
  } else if (I->getPrevNode() == nullptr) {
    // Inserting at the beginning of the block.
    SBInstruction *NextI = I->getNextNode();
    assert(NextI != nullptr &&
           "Should've been handled by `if (I->getNextNode() == null)`");
    auto NextNum = getInstrNumber(NextI);
    assert(NextNum > std::numeric_limits<decltype(NextNum)>::min() +
                         InstrNumberingStep &&
           "You're gonna need a bigger boat!");
    Num = NextNum - InstrNumberingStep;
  } else {
    // Inserting between two instructions.
    auto GetNum = [this](SBInstruction *I) -> std::optional<int64_t> {
      auto *NextI = I->getNextNode();
      auto *PrevI = I->getPrevNode();
      assert(NextI != nullptr && PrevI != nullptr && "Expected next and prev");
      int64_t NextNum = getInstrNumber(NextI);
      int64_t PrevNum = getInstrNumber(PrevI);
      int64_t NewNum = (PrevNum + NextNum) / 2;
      bool LargeEnoughGap = NewNum != PrevNum && NewNum != NextNum;
      if (!LargeEnoughGap)
        return std::nullopt;
      return NewNum;
    };
    auto NumOpt = GetNum(I);
    if (!NumOpt) {
      renumberInstructions();
      NumOpt = GetNum(I);
      assert(NumOpt && "Expected a large enough gap after renumbering");
    }
    Num = *NumOpt;
  }
  InstrNumberMap[I] = Num;
}

void SBBasicBlock::removeInstrNumber(SBInstruction *I) {
  InstrNumberMap.erase(I);
}

bool SBBasicBlock::classof(const SBValue *From) {
  return From->getSubclassID() == SBValue::ClassID::Block;
}

void SBBasicBlock::buildSBBasicBlockFromIR(BasicBlock *BB) {
  for (Instruction &IRef : reverse(*BB)) {
    Instruction *I = &IRef;
    SBValue *SBV = Ctxt.getOrCreateSBValue(I);
    for (auto [OpIdx, Op] : enumerate(I->operands())) {
      // For now Unpacks only have a single operand.
      if (OpIdx > 0 && isa<SBInstruction>(SBV) &&
          cast<SBInstruction>(SBV)->getNumOperands() == 1)
        continue;
      // Skip instruction's label operands
      if (isa<BasicBlock>(Op))
        continue;
      // Skip metadata for now
      if (isa<MetadataAsValue>(Op))
        continue;
      // Skip asm
      if (isa<InlineAsm>(Op))
        continue;
      Ctxt.getOrCreateSBValue(Op);
    }
  }
  // Instruction numbering
  renumberInstructions();
#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  verify();
#endif
}

SBBasicBlock::iterator SBBasicBlock::getFirstNonPHIIt() {
  Instruction *FirstI = cast<BasicBlock>(Val)->getFirstNonPHI();
  return FirstI == nullptr
             ? begin()
             : cast<SBInstruction>(Ctxt.getSBValue(FirstI))->getIterator();
}

SBBasicBlock::SBBasicBlock(BasicBlock *BB, SBContext &SBCtxt)
    : SBValue(ClassID::Block, BB, SBCtxt) {}

SBBasicBlock::~SBBasicBlock() {
  Ctxt.destroyingBB(*this);
  // This BB is now gone, so there is no need for the BB-specific callbacks.
  Ctxt.RemoveInstrCallbacksBB.erase(this);
  Ctxt.InsertInstrCallbacksBB.erase(this);
  Ctxt.MoveInstrCallbacksBB.erase(this);
}

SBFunction *SBBasicBlock::getParent() const {
  auto *BB = cast<BasicBlock>(Val);
  auto *F = BB->getParent();
  if (F == nullptr)
    // Detached
    return nullptr;
  return Ctxt.getSBFunction(F);
}

SBBasicBlock::iterator SBBasicBlock::begin() const {
  BasicBlock *BB = cast<BasicBlock>(Val);
  BasicBlock::iterator It = BB->begin();
  if (!BB->empty()) {
    auto *SBV = Ctxt.getSBValue(&*BB->begin());
    assert(SBV != nullptr && "No SandboxIR for BB->begin()!");
    auto *SBI = cast<SBInstruction>(SBV);
    unsigned Num = SBI->getNumOfIRInstrs();
    assert(Num >= 1u && "Bad getNumOfIRInstrs()");
    It = std::next(It, Num - 1);
  }
  return iterator(BB, It, &Ctxt);
}

void SBBasicBlock::detach() {
  // We are detaching bottom-up because detaching some SandboxIR
  // Instructions require non-detached operands.
  // Note: we are in the process of detaching from the underlying BB, so we
  //       can't rely on 1-1 mapping between IR and SandboxIR.
  for (Instruction &I : reverse(*cast<BasicBlock>(Val))) {
    if (auto *SI = Ctxt.getSBValue(&I))
      Ctxt.detach(SI);
  }
}

void SBBasicBlock::detachFromLLVMIR() {
  // Detach instructions
  detach();
  // Detach the actual BB
  Ctxt.detach(this);
}

SBArgument *SBContext::getSBArgument(Argument *Arg) const {
  return cast_or_null<SBArgument>(getSBValue(Arg));
}

SBArgument *SBContext::createSBArgument(Argument *Arg) {
  assert(getSBArgument(Arg) == nullptr && "Already exists!");
  auto NewArg = std::unique_ptr<SBArgument>(new SBArgument(Arg, *this));
  return cast<SBArgument>(registerSBValue(std::move(NewArg)));
}

SBArgument *SBContext::getOrCreateSBArgument(Argument *Arg) {
  // TODO: Try to avoid two lookups in getOrCreate functions.
  if (auto *TArg = getSBArgument(Arg))
    return TArg;
  return createSBArgument(Arg);
}

SBValue *
SBContext::getSBValueFromExtractElement(ExtractElementInst *ExtractI) const {
  return getSBValue(ExtractI);
}
SBValue *
SBContext::getOrCreateSBValueFromExtractElement(ExtractElementInst *ExtractI,
                                                int Depth) {
  if (auto *SBV = getSBValueFromExtractElement(ExtractI))
    return SBV;
  return createSBValueFromExtractElement(ExtractI, Depth);
}

SBValue *
SBContext::getSBValueFromInsertElement(InsertElementInst *InsertI) const {
  return getSBValue(InsertI);
}
SBValue *
SBContext::getOrCreateSBValueFromInsertElement(InsertElementInst *InsertI,
                                               int Depth) {
  if (auto *SBV = getSBValueFromInsertElement(InsertI))
    return SBV;
  return createSBValueFromInsertElement(InsertI, Depth);
}

SBValue *
SBContext::getSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI) const {
  return getSBValue(ShuffleI);
}

SBValue *
SBContext::getOrCreateSBValueFromShuffleVector(ShuffleVectorInst *ShuffleI,
                                               int Depth) {
  if (auto *SBV = getSBValueFromShuffleVector(ShuffleI))
    return SBV;
  return createSBValueFromShuffleVector(ShuffleI, Depth);
}

SBOpaqueInstruction *SBContext::getSBOpaqueInstruction(Instruction *I) const {
  return cast_or_null<SBOpaqueInstruction>(getSBValue(I));
}

SBOpaqueInstruction *SBContext::createSBOpaqueInstruction(Instruction *I) {
  assert(getSBOpaqueInstruction(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBOpaqueInstruction>(new SBOpaqueInstruction(I, *this));
  return cast<SBOpaqueInstruction>(registerSBValue(std::move(NewPtr)));
}

SBOpaqueInstruction *SBContext::getOrCreateSBOpaqueInstruction(Instruction *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBOpaqueInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBOpaqueInstruction(I);
}

SBInsertElementInstruction *
SBContext::getSBInsertElementInstruction(InsertElementInst *I) const {
  return cast_or_null<SBInsertElementInstruction>(getSBValue(I));
}

SBInsertElementInstruction *
SBContext::createSBInsertElementInstruction(InsertElementInst *I) {
  assert(getSBInsertElementInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBInsertElementInstruction>(
      new SBInsertElementInstruction(I, *this));
  return cast<SBInsertElementInstruction>(registerSBValue(std::move(NewPtr)));
}

SBInsertElementInstruction *
SBContext::getOrCreateSBInsertElementInstruction(InsertElementInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBInsertElementInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBInsertElementInstruction(I);
}

SBExtractElementInstruction *
SBContext::getSBExtractElementInstruction(ExtractElementInst *I) const {
  return cast_or_null<SBExtractElementInstruction>(getSBValue(I));
}

SBExtractElementInstruction *
SBContext::createSBExtractElementInstruction(ExtractElementInst *I) {
  assert(getSBExtractElementInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBExtractElementInstruction>(
      new SBExtractElementInstruction(I, *this));
  return cast<SBExtractElementInstruction>(registerSBValue(std::move(NewPtr)));
}

SBExtractElementInstruction *
SBContext::getOrCreateSBExtractElementInstruction(ExtractElementInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBExtractElementInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBExtractElementInstruction(I);
}

SBShuffleVectorInstruction *
SBContext::getSBShuffleVectorInstruction(ShuffleVectorInst *I) const {
  return cast_or_null<SBShuffleVectorInstruction>(getSBValue(I));
}

SBShuffleVectorInstruction *
SBContext::createSBShuffleVectorInstruction(ShuffleVectorInst *I) {
  assert(getSBShuffleVectorInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBShuffleVectorInstruction>(
      new SBShuffleVectorInstruction(I, *this));
  return cast<SBShuffleVectorInstruction>(registerSBValue(std::move(NewPtr)));
}

SBShuffleVectorInstruction *
SBContext::getOrCreateSBShuffleVectorInstruction(ShuffleVectorInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBShuffleVectorInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBShuffleVectorInstruction(I);
}

SBReturnInstruction *SBContext::getSBReturnInstruction(ReturnInst *I) const {
  return cast_or_null<SBReturnInstruction>(getSBValue(I));
}

SBReturnInstruction *SBContext::createSBReturnInstruction(ReturnInst *I) {
  assert(getSBReturnInstruction(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBReturnInstruction>(new SBReturnInstruction(I, *this));
  return cast<SBReturnInstruction>(registerSBValue(std::move(NewPtr)));
}

SBReturnInstruction *SBContext::getOrCreateSBReturnInstruction(ReturnInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBReturnInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBReturnInstruction(I);
}

SBCallInstruction *SBContext::getSBCallInstruction(CallInst *I) const {
  return cast_or_null<SBCallInstruction>(getSBValue(I));
}

SBCallInstruction *SBContext::createSBCallInstruction(CallInst *I) {
  assert(getSBCallInstruction(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBCallInstruction>(new SBCallInstruction(I, *this));
  return cast<SBCallInstruction>(registerSBValue(std::move(NewPtr)));
}

SBCallInstruction *SBContext::getOrCreateSBCallInstruction(CallInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBCallInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBCallInstruction(I);
}

SBGetElementPtrInstruction *
SBContext::getSBGetElementPtrInstruction(GetElementPtrInst *I) const {
  return cast_or_null<SBGetElementPtrInstruction>(getSBValue(I));
}

SBGetElementPtrInstruction *
SBContext::createSBGetElementPtrInstruction(GetElementPtrInst *I) {
  assert(getSBGetElementPtrInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBGetElementPtrInstruction>(
      new SBGetElementPtrInstruction(I, *this));
  return cast<SBGetElementPtrInstruction>(registerSBValue(std::move(NewPtr)));
}

SBGetElementPtrInstruction *
SBContext::getOrCreateSBGetElementPtrInstruction(GetElementPtrInst *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBGetElementPtrInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBGetElementPtrInstruction(I);
}

SBBasicBlock *SBContext::getSBBasicBlock(BasicBlock *BB) const {
  return cast_or_null<SBBasicBlock>(getSBValue(BB));
}

SBBasicBlock *SBContext::createSBBasicBlock(BasicBlock *BB) {
  DontNumberInstrs = true;
  assert(getSBBasicBlock(BB) == nullptr && "Already exists!");
  auto NewBBPtr = std::unique_ptr<SBBasicBlock>(new SBBasicBlock(BB, *this));
  auto *SBBB = cast<SBBasicBlock>(registerSBValue(std::move(NewBBPtr)));
  // Create SandboxIR for BB's body.
  SBBB->buildSBBasicBlockFromIR(BB);

  // Run hook.
  createdSBBasicBlock(*SBBB);

  DontNumberInstrs = false;
  return SBBB;
}

SBFunction *SBContext::getSBFunction(Function *F) const {
  return cast_or_null<SBFunction>(getSBValue(F));
}

SBFunction *SBContext::createSBFunction(Function *F, bool CreateBBs) {
  assert(getSBFunction(F) == nullptr && "Already exists!");
  auto NewFPtr = std::unique_ptr<SBFunction>(new SBFunction(F, *this));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateSBArgument(&Arg);
  // Create BBs.
  if (CreateBBs) {
    for (auto &BB : *F)
      createSBBasicBlock(&BB);
  }
  auto *SBF = cast<SBFunction>(registerSBValue(std::move(NewFPtr)));
  return SBF;
}

SBInstruction &SBBasicBlock::front() const {
  auto *BB = cast<BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<SBInstruction>(getContext().getSBValue(&*BB->begin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

SBInstruction &SBBasicBlock::back() const {
  auto *BB = cast<BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<SBInstruction>(getContext().getSBValue(&*BB->rbegin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

#ifndef NDEBUG
void SBBasicBlock::verify() {
  auto *BB = cast<BasicBlock>(Val);
  for (Instruction &IRef : *BB) {
    // Check that all llvm instructions in BB have a corresponding SBValue.
    assert(getContext().getSBValue(&IRef) != nullptr && "No SBValue for IRef!");
  }

  SBInstruction *LastI = nullptr;
  unsigned CntInstrs = 0;
  for (SBInstruction &IRef : *this) {
    ++CntInstrs;
    // Check instrunction numbering.
    if (LastI != nullptr)
      assert(LastI->comesBefore(&IRef) && "Broken instruction numbering!");
    LastI = &IRef;
  }
  assert(InstrNumberMap.size() == CntInstrs &&
         "Forgot to add/remove instrs from map?");

  // Note: we are not simply doing bool HaveSandboxIRForWholeFn = getParent()
  // because there is the corner case of @function operands of constants.
  SBValue *SBF = Ctxt.getSBValue(BB->getParent());
  bool HaveSandboxIRForWholeFn = SBF != nullptr && isa<SBFunction>(SBF);
  // Check operand/user consistency.
  for (const SBInstruction &SBI : *this) {
    Value *V = ValueAttorney::getValue(&SBI);
    assert(!isa<BasicBlock>(V) && "Broken SBBasicBlock construction!");

    // Note: This is expensive for packs, so skip based on num of LLVM instrs.
    if (SBI.getNumOfIRInstrs() < 16) {
      for (auto [OpIdx, Use] : enumerate(SBI.operands())) {
        SBValue *Op = Use;
        if (HaveSandboxIRForWholeFn) {
          Value *LLVMOp = SBUseAttorney::getLLVMUse(Use)->get();
          if (isa<Instruction>(LLVMOp) || isa<Constant>(LLVMOp))
            assert(Op != nullptr && "Null instruction/constant operands are "
                                    "not allowed when we have "
                                    "SandboxIR for the whole function");
        }
        if (Op == nullptr)
          continue;
        // Op could be an operand of a ConstantVector. We don't model this.
        assert((isa<Constant>(ValueAttorney::getValue(Op)) ||
                find(Op->users(), &SBI) != Op->users().end()) &&
               "If Op is SBI's operand, then SBI should be in Op's users.");
        // Count how many times Op is found in operands.
        unsigned CntOpEdges = 0;
        for_each(SBI.operands(), [&CntOpEdges, Op](SBValue *TmpOp) {
          if (TmpOp == Op)
            ++CntOpEdges;
        });
        if (CntOpEdges > 1 && !isa<SBConstant>(Op)) {
          // Check that Op has `CntOp` users matching `SBI`.
          unsigned CntUserEdges = 0;
          for_each(Op->users(), [&CntUserEdges, &SBI](SBUser *User) {
            if (User == &SBI)
              ++CntUserEdges;
          });
          assert(
              CntOpEdges == CntUserEdges &&
              "Broken IR! User edges count doesn't match operand edges count!");
        }
      }
    }
    for (auto *User : SBI.users()) {
      if (User == nullptr)
        continue;
      assert(find(User->operands(), &SBI) != User->operands().end() &&
             "If User is in SBI's users, then SBI should be in User's "
             "operands.");
    }
  }

  SBInstruction *LastNonPHI = nullptr;
  // Checks opcodes and other.
  for (SBInstruction &SBI : *this) {
    if (LLVM_UNLIKELY(SBI.isPad())) {
      assert(&SBI == &*getFirstNonPHIIt() &&
             "{Landing,Catch,Cleanup}Pad Instructions must be the non-PHI!");
    }
    if (isa<SBPHINode>(&SBI)) {
      if (LastNonPHI != nullptr) {
        errs() << "SBPHIs not grouped at top of BB!\n";
        errs() << SBI << "\n";
        errs() << *LastNonPHI << "\n";
        llvm_unreachable("Broken SandboxIR");
      }
    } else {
      LastNonPHI = &SBI;
    }
  }

  // Check that we only have a single SBValue for every constant.
  DenseMap<Value *, const SBValue *> Map;
  for (const SBValue &SBV : *this) {
    Value *V = ValueAttorney::getValue(&SBV);
    if (isa<Constant>(V)) {
      auto Pair = Map.insert({V, &SBV});
      if (!Pair.second) {
        auto It = Pair.first;
        assert(&SBV == It->second &&
               "Expected a unique SBValue for each LLVM IR constant!");
      }
    }
  }
}

void SBBasicBlock::verifyLLVMIR() const {
  // Check that all llvm instructions in BB have a corresponding SBValue.
  auto *BB = cast<BasicBlock>(Val);
  Instruction *LastI = nullptr;
  for (Instruction &IRef : *BB) {
    Instruction *I = &IRef;
    for (Value *Op : I->operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (OpI == nullptr)
        continue;
      if (OpI->getParent() != BB)
        continue;
      if (!isa<PHINode>(I) && !OpI->comesBefore(I)) {
        errs() << "Instruction does not dominate uses!\n";
        errs() << *Op << " " << Op << "\n";
        errs() << *I << " " << I << "\n";
        errs() << "\n";

        errs() << "SBValues:\n";
        auto *SBOp = Ctxt.getSBValue(Op);
        if (SBOp != nullptr)
          errs() << *SBOp << " " << SBOp << "\n";
        else
          errs() << "No SBValue for Op\n";
        auto *SBI = Ctxt.getSBValue(I);
        if (SBI != nullptr)
          errs() << *SBI << " " << SBI << "\n";
        else
          errs() << "No SBValue for I\n";
        llvm_unreachable("Instruction does not dominate uses!");
      }
    }

    if (LastI != nullptr && isa<PHINode>(I) && !isa<PHINode>(LastI)) {
      errs() << "PHIs not grouped at top of BB!\n";
      errs() << *LastI << " " << LastI << "\n";
      errs() << *I << " " << I << "\n";
      llvm_unreachable("PHIs not grouped at top of BB!\n");
    }
    LastI = I;
  }
}

void SBBasicBlock::dumpVerbose(raw_ostream &OS) const {
  for (const auto &SBI : reverse(*this)) {
    SBI.dumpVerbose(OS);
    OS << "\n";
  }
}
void SBBasicBlock::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBBasicBlock::dump(raw_ostream &OS) const {
  BasicBlock *BB = cast<BasicBlock>(Val);
  const auto &Name = BB->getName();
  OS << Name;
  if (!Name.empty())
    OS << ":\n";
  // If there are Instructions in the BB that are not mapped to SandboxIR, then
  // use a crash-proof dump.
  if (any_of(*BB, [this](Instruction &I) {
        return Ctxt.getSBValue(&I) == nullptr;
      })) {
    OS << "<Crash-proof mode!>\n";
    DenseSet<SBInstruction *> Visited;
    for (Instruction &IRef : *BB) {
      SBValue *SBV = Ctxt.getSBValue(&IRef);
      if (SBV == nullptr)
        OS << IRef << " *** No SandboxIR ***\n";
      else {
        auto *SBI = dyn_cast<SBInstruction>(SBV);
        if (SBI == nullptr)
          OS << IRef << " *** Not a SBInstruction!!! ***\n";
        else {
          if (Visited.insert(SBI).second)
            OS << *SBI << "\n";
        }
      }
    }
  } else {
    for (auto &SBI : *this) {
      SBI.dump(OS);
      OS << "\n";
    }
  }
}
void SBBasicBlock::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBBasicBlock::dumpInstrs(SBValue *SBV, int Num) const {
  auto *SBI = dyn_cast<SBInstruction>(SBV);
  if (SBI == nullptr) {
    dbgs() << "SBV == null!\n";
    return;
  }
  auto *FromI = SBI;
  for (auto Cnt : seq<int>(0, Num)) {
    (void)Cnt;
    auto *PrevI = FromI->getPrevNode();
    if (PrevI == nullptr)
      break;
    FromI = PrevI;
  }
  auto *ToI = SBI;
  for (auto Cnt : seq<int>(0, Num)) {
    (void)Cnt;
    auto *NextI = ToI->getNextNode();
    if (NextI == nullptr)
      break;
    ToI = NextI;
  }
  for (SBInstruction *I = FromI, *E = ToI->getNextNode(); I != E;
       I = I->getNextNode())
    dbgs() << *I << "\n";
}
#endif

SandboxIRTracker &SBBasicBlock::getTracker() { return Ctxt.getTracker(); }

SBInstruction *SBBasicBlock::getTerminator() const {
  auto *TerminatorV = Ctxt.getSBValue(cast<BasicBlock>(Val)->getTerminator());
  return cast_or_null<SBInstruction>(TerminatorV);
}

SBBasicBlock::iterator::pointer
SBBasicBlock::iterator::getSBI(BasicBlock::iterator It) const {
  SBInstruction *SBI = cast_or_null<SBInstruction>(SBCtxt->getSBValue(&*It));
  assert((!SBI || cast<Instruction>(ValueAttorney::getValue(SBI)) == &*It) &&
         "It should always point at the bottom IR instruction of a "
         "SBInstruction!");
  return SBI;
}

SBContext::RemoveCBTy *SBContext::registerRemoveInstrCallback(RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterRemoveInstrCallback(RemoveCBTy *CB) {
  auto It = find_if(RemoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != RemoveInstrCallbacks.end()) &&
         "Callback not registered!");
  if (It != RemoveInstrCallbacks.end())
    RemoveInstrCallbacks.erase(It);
}

SBContext::InsertCBTy *SBContext::registerInsertInstrCallback(InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterInsertInstrCallback(InsertCBTy *CB) {
  auto It = find_if(InsertInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != InsertInstrCallbacks.end() && "Callback not registered!");
  InsertInstrCallbacks.erase(It);
}

SBContext::MoveCBTy *SBContext::registerMoveInstrCallback(MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterMoveInstrCallback(MoveCBTy *CB) {
  auto It = find_if(MoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != MoveInstrCallbacks.end() && "Callback not registered!");
  MoveInstrCallbacks.erase(It);
}

SBContext::RemoveCBTy *
SBContext::registerRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterRemoveInstrCallbackBB(SBBasicBlock &BB,
                                                RemoveCBTy *CB) {
  auto MapIt = RemoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != RemoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != Vec.end()) && "Callback not registered!");
  if (It != Vec.end())
    Vec.erase(It);
}

SBContext::InsertCBTy *
SBContext::registerInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterInsertInstrCallbackBB(SBBasicBlock &BB,
                                                InsertCBTy *CB) {
  auto MapIt = InsertInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != InsertInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

SBContext::MoveCBTy *SBContext::registerMoveInstrCallbackBB(SBBasicBlock &BB,
                                                            MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy *CB) {
  auto MapIt = MoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != MoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

void SBContext::quickFlush() {
  InQuickFlush = true;

  RemoveInstrCallbacks.clear();
  InsertInstrCallbacks.clear();
  MoveInstrCallbacks.clear();

  RemoveInstrCallbacksBB.clear();
  InsertInstrCallbacksBB.clear();
  MoveInstrCallbacksBB.clear();

  LLVMValueToSBValueMap.clear();
  MultiInstrMap.clear();
  InQuickFlush = false;
}

void SBContext::runRemoveInstrCallbacks(SBInstruction *SBI) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : RemoveInstrCallbacks)
    (*CBPtr)(SBI);

  auto *BB = SBI->getParent();
  auto It = RemoveInstrCallbacksBB.find(BB);
  if (It != RemoveInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI);
  }
}

void SBContext::runInsertInstrCallbacks(SBInstruction *SBI) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : InsertInstrCallbacks)
    (*CBPtr)(SBI);

  auto *BB = SBI->getParent();
  auto It = InsertInstrCallbacksBB.find(BB);
  if (It != InsertInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI);
  }
}

void SBContext::runMoveInstrCallbacks(SBInstruction *SBI, SBBasicBlock &SBBB,
                                      const SBBBIterator &WhereIt) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : MoveInstrCallbacks)
    (*CBPtr)(SBI, SBBB, WhereIt);

  auto It = MoveInstrCallbacksBB.find(&SBBB);
  if (It != MoveInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI, SBBB, WhereIt);
  }
}
