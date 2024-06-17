set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  )

add_llvm_unittest(SandboxIRTests
  SandboxIRTest.cpp
  SandboxIRTrackerTest.cpp
  DmpVectorTest.cpp
  )

set_property(TARGET SandboxIRTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
