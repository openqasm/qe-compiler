//===- QUIRToAER.cpp - Convert QUIR to AER --------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements passes for converting QUIR to AER
//
//===----------------------------------------------------------------------===//
#include "Conversion/QUIRToAer.h"
#include "Conversion/TypeConversion.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "SimulatorUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

#include <exception>

namespace qssc::targets::simulator::conversion {

namespace {

class AerStateWrapper {
public:
  AerStateWrapper(Value mem) : mem(mem) {}
  
  Value access(OpBuilder& builder) const {
    return builder.create<LLVM::LoadOp>(
        builder.getUnknownLoc(), mem, /*alignment=*/8);
  }

private:
  Value mem;
};
  
std::map<std::string, LLVM::LLVMFuncOp> aerFuncTable;

// TODO: Take care of errors, nan and inf.
using AngleTable = std::map<double, Value>;
using QubitTable = std::map<int, Value>;

// TODO: We should map some IDs (attributes?) to values
class ValueTable {
public:
  static void registerAngle(double angle, Value v) {
    assert(!std::isnan(angle) && !std::isinf(angle));
    angleTable[angle] = v;
  }
  
  static void registerQubit(int qid, Value v) {
    qubitTable[qid] = v;
  }
  
  static Value lookupAngle(double angle) {
    return angleTable.at(angle);
  }
  
  static Value lookupQubit(int qid) {
    return qubitTable.at(qid);
  }
  
  static QubitTable getQubits() {
    return qubitTable;
  }
  
  static void clear() {
    angleTable.clear();
    qubitTable.clear();
  }
  
  static void clearQubitTable() {
    qubitTable.clear();
  }

private:
  static AngleTable angleTable;
  static QubitTable qubitTable;
};
AngleTable ValueTable::angleTable;
QubitTable ValueTable::qubitTable;

void buildQubitTable(ModuleOp moduleOp, AerStateWrapper wrapper) {
  ValueTable::clearQubitTable();
  OpBuilder builder(moduleOp);
  
  
  moduleOp.walk([&](quir::DeclareQubitOp declOp){
    const int width = declOp.getType().dyn_cast<quir::QubitType>().getWidth();
    if(width != 1) throw std::runtime_error(""); // TODO
    
    builder.setInsertionPointAfter(declOp);
    const auto widthAttr = builder.getIntegerAttr(
        builder.getI64Type(), width);
    auto constOp = builder.create<arith::ConstantOp>(
        declOp->getLoc(), builder.getI64Type(), widthAttr);
    auto aerState = wrapper.access(builder);
    auto alloc = builder.create<LLVM::CallOp>(
        declOp->getLoc(),
        aerFuncTable.at("aer_allocate_qubits"),
        ValueRange{aerState, constOp});

    ValueTable::registerQubit(*declOp.id(), alloc.getResult(0));
  });
  
  // TODO
  // Assume that qubit declaration with the biggest id is called after any other declaration
  auto qubitTable = ValueTable::getQubits();
  if(!qubitTable.empty()) {
    auto declOp = qubitTable.rbegin()->second.getDefiningOp();
    builder.setInsertionPointAfter(declOp);
    auto aerState = wrapper.access(builder);
    builder.create<LLVM::CallOp>(
        declOp->getLoc(), aerFuncTable.at("aer_state_initialize"), aerState);
  }
}

}

// Assume qcs.init is called before all quir.declare_qubit operations
struct QCSInitConversionPat : public OpConversionPattern<qcs::SystemInitOp> {
  explicit QCSInitConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1), aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(qcs::SystemInitOp initOp,
                                qcs::SystemInitOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    
    // global string values for aer configuration
    std::map<std::string, mlir::Value> globals;
    const auto config_strs = {
        "method", "statevector",
        "device", "CPU",
        "precision", "double"};
    for(auto config_str : config_strs) {
      const auto var_name = std::string("aer_conf_") + config_str;
      const auto with_null = config_str + std::string("\0", 1);
      globals[config_str] = LLVM::createGlobalString(
        initOp->getLoc(), rewriter, var_name,
        with_null, LLVM::Linkage::Private);
    }
    // configure
    // aer_state_configure(state, "method", "statevector")
    // aer_state_configure(state, "device", "CPU")
    // aer_state_configure(state, "precision", "double")
    auto state = aerState.access(rewriter);
    rewriter.create<LLVM::CallOp>(
        initOp->getLoc(), aerFuncTable.at("aer_state_configure"),
        ValueRange{state, globals["method"], globals["statevector"]});
    rewriter.create<LLVM::CallOp>(
        initOp->getLoc(), aerFuncTable.at("aer_state_configure"),
        ValueRange{state, globals["device"], globals["CPU"]});
    rewriter.create<LLVM::CallOp>(
        initOp->getLoc(), aerFuncTable.at("aer_state_configure"),
        ValueRange{state, globals["precision"], globals["double"]});
    rewriter.eraseOp(initOp);
    return success();
  }
  
private:
  AerStateWrapper aerState;
};

struct RemoveQCSShotInitConversionPat : public OpConversionPattern<qcs::ShotInitOp> {
  explicit RemoveQCSShotInitConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1)
  {}
  
  LogicalResult matchAndRewrite(qcs::ShotInitOp initOp,
                                qcs::ShotInitOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(initOp);
    return success();
  }
};

struct FinalizeConversionPat : public OpConversionPattern<qcs::SystemFinalizeOp> {
  explicit FinalizeConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(qcs::SystemFinalizeOp finOp,
                                qcs::SystemFinalizeOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointAfter(finOp);
    rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(),
                                  aerFuncTable.at("aer_state_finalize"),
                                  aerState.access(rewriter));
    rewriter.eraseOp(finOp);
    return success();
  }
  
private:
  AerStateWrapper aerState;
};

struct BuiltinUopConversionPat : public OpConversionPattern<quir::Builtin_UOp> {
  explicit BuiltinUopConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::Builtin_UOp op,
                                quir::Builtin_UOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    // qubit, angle{1, 2, 3}
    assert(op.getOperands().size() == 4);

    std::vector<Value> args;
    args.emplace_back(aerState.access(rewriter));
    { // qubit
      const auto qID = quir::lookupQubitId(op.target());
      assert(qID);
      args.emplace_back(ValueTable::lookupQubit(*qID));
    }
    for(auto val : {op.theta(), op.phi(), op.lambda()}) {
      auto angleOp = dyn_cast<quir::ConstantOp>(val.getDefiningOp());
      const double angle = angleOp.value().dyn_cast<quir::AngleAttr>().getValue().convertToDouble();
      args.emplace_back(ValueTable::lookupAngle(angle));
    }
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), aerFuncTable.at("aer_apply_u3"), args);
    rewriter.eraseOp(op);
    return success();
  }
  
private:
  AerStateWrapper aerState;
};

struct BuiltinCXConversionPat : public OpConversionPattern<quir::BuiltinCXOp> {
  explicit BuiltinCXConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::BuiltinCXOp op,
                                quir::BuiltinCXOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    auto qID1 = quir::lookupQubitId(op->getOperand(0));
    auto qID2 = quir::lookupQubitId(op->getOperand(1));
    if(qID1 && qID2) {
      auto q1 = ValueTable::lookupQubit(*qID1);
      auto q2 = ValueTable::lookupQubit(*qID2);
      auto state = aerState.access(rewriter);
      rewriter.create<LLVM::CallOp>(
          op->getLoc(),
          aerFuncTable.at("aer_apply_cx"),
          ValueRange{state, q1, q2});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
  
private:
  AerStateWrapper aerState;
};

struct MeasureOpConversionPat : public OpConversionPattern<quir::MeasureOp> {
  explicit MeasureOpConversionPat(
      MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::MeasureOp op,
                                quir::MeasureOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    // TODO: Should not create a new array each time.
    if(auto qID = quir::lookupQubitId(op->getOperand(0))) {
      auto context = op->getContext();
      const auto i64Type = IntegerType::get(context, 64);
      const unsigned arrSize = 1; // TODO
      const IntegerAttr arraySizeAttr = rewriter.getIntegerAttr(i64Type, arrSize);
      const unsigned int align = 8; // TODO
      const auto qubit = ValueTable::lookupQubit(*qID);
      auto arrSizeOp = rewriter.create<arith::ConstantOp>(
          op->getLoc(), i64Type, arraySizeAttr);
      auto qubitArr = rewriter.create<LLVM::AllocaOp>(
          op->getLoc(),
          LLVM::LLVMPointerType::get(i64Type),
          arrSizeOp, align);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), qubit, qubitArr);
      auto state = aerState.access(rewriter);
      auto meas = rewriter.create<LLVM::CallOp>(
          op->getLoc(),
          aerFuncTable.at("aer_apply_measure"),
          ValueRange{state, qubitArr.getResult(), arrSizeOp});
      rewriter.replaceOp(op, meas.getResult(0));
      return success();
    }
    return failure();
  }
  
private:
  AerStateWrapper aerState;
};

struct AngleConversionPat : public OpConversionPattern<quir::ConstantOp> {
  explicit AngleConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern(typeConverter, ctx, 1)
  {}
  
  LogicalResult matchAndRewrite(quir::ConstantOp op,
                               quir::ConstantOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override
  {
    if(auto angleAttr = op.value().dyn_cast<quir::AngleAttr>()) {
      rewriter.setInsertionPointAfter(op);
      const auto angle = angleAttr.getValue().convertToDouble();
      const auto fType = rewriter.getF64Type();
      FloatAttr fAttr = rewriter.getFloatAttr(fType, angle);
      auto constOp = rewriter.create<arith::ConstantOp>(op->getLoc(), fType, fAttr);
      rewriter.replaceOp(op, {constOp});
      // TODO: We must build angle table before applying QUIRToAerPass
      ValueTable::registerAngle(angle, constOp);
    }
    return success();
  }
};

template <typename Op>
struct RemoveConversionPat : public OpConversionPattern<Op> {
  using Adaptor = typename Op::Adaptor;
  
  explicit RemoveConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern<Op>(typeConverter, ctx, 1)
  {}
  
  LogicalResult matchAndRewrite(Op op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// TDOO
// Probably I should implement this pattern as another pass
struct FunctionConversionPat : public OpConversionPattern<mlir::FuncOp> {
  explicit FunctionConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, AerStateWrapper aerState)
    : OpConversionPattern(typeConverter, ctx, 1), aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(mlir::FuncOp funcOp, mlir::FuncOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Assume: funcOp != mainFunc
    rewriter.eraseOp(funcOp);
    return success();
  }
  
private:
  AerStateWrapper aerState;
};


void conversion::QUIRToAERPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::AffineDialect, arith::ArithmeticDialect>();
}

void QUIRToAERPass::runOnOperation(SimulatorSystem &system) {
  ModuleOp moduleOp = getOperation();

  // First remove all arguments from synchronization ops
  moduleOp->walk([](qcs::SynchronizeOp synchOp) {
    synchOp.qubitsMutable().assign(ValueRange({}));
  });

  AerTypeConverter typeConverter;
  auto *context = &getContext();
  ConversionTarget target(*context);

  target.addLegalDialect<arith::ArithmeticDialect, LLVM::LLVMDialect,
                         mlir::AffineDialect, memref::MemRefDialect,
                         scf::SCFDialect, StandardOpsDialect,
                         mlir::pulse::PulseDialect>();
  // Since we are converting QUIR -> AER/LLVM, make QUIR illegal.
  // Further, because OQ3 and QCS ops are migrated from QUIR, make them also
  // illegal.
  target.addIllegalDialect<qcs::QCSDialect, oq3::OQ3Dialect>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType());
  });

  RewritePatternSet patterns(context);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<RemoveQCSShotInitConversionPat,
               RemoveConversionPat<quir::DeclareQubitOp>,
               RemoveConversionPat<oq3::DeclareVariableOp>,
               RemoveConversionPat<oq3::CastOp>,
               RemoveConversionPat<oq3::VariableAssignOp>,
               RemoveConversionPat<oq3::CBitAssignBitOp>,
               AngleConversionPat>(
      context, typeConverter);
  
  // Aer initialization
  declareAerFunctions(moduleOp);
  auto aerState = [&]() -> AerStateWrapper {
    auto mainFunc = mlir::quir::getMainFunction(moduleOp);
    OpBuilder builder(mainFunc);

    auto mainBody = &mainFunc->getRegion(0).getBlocks().front();
    builder.setInsertionPointToStart(mainBody);
    const auto i8Type = IntegerType::get(context, 8);
    const auto i64Type = IntegerType::get(context, 64);
    auto arrSizeOp = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), i64Type,
        builder.getIntegerAttr(i64Type, 1));
    auto alloca = builder.create<LLVM::AllocaOp>(
        builder.getUnknownLoc(),
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(i8Type)),
        arrSizeOp, /*alignment=*/8); // alignment must be 8
    auto call = builder.create<LLVM::CallOp>(
        builder.getUnknownLoc(),
        aerFuncTable.at("aer_state"),
        ValueRange{}).getResult(0);
    builder.create<LLVM::StoreOp>(builder.getUnknownLoc(), call, alloca);
    return AerStateWrapper(alloca);
  }();
  // Must build qubit table before applying any conversion
  buildQubitTable(moduleOp, aerState);
  
  patterns.add<QCSInitConversionPat,
               FinalizeConversionPat,
               BuiltinUopConversionPat,
               BuiltinCXConversionPat,
               MeasureOpConversionPat,
               FunctionConversionPat>(
    context, typeConverter, aerState);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    llvm::outs() << "Failed applyPartialConversion\n";
  }
} // QUIRToStdPass::runOnOperation()

llvm::StringRef QUIRToAERPass::getArgument() const {
  return "simulator-oq3-to-aer";
}

llvm::StringRef QUIRToAERPass::getDescription() const {
  return "Convert OQ3 ops to aer";
}

void QUIRToAERPass::declareAerFunctions(ModuleOp moduleOp) {
  using LLVM::LLVMFunctionType;

  aerFuncTable.clear();

  OpBuilder builder(moduleOp);

  auto registerFunc = [&](const char* name, LLVMFunctionType ty) {
    const auto loc = builder.getUnknownLoc();
    const auto f = builder.create<LLVM::LLVMFuncOp>(loc, name, ty);
    aerFuncTable.insert({name, f});
  };

  auto context = moduleOp->getContext();
  builder.setInsertionPointToStart(moduleOp.getBody());
  // common types
  const auto voidType = LLVM::LLVMVoidType::get(context);
  const auto i8Type = IntegerType::get(context, 8);
  const auto i64Type = IntegerType::get(context, 64);
  const auto f64Type = Float64Type::get(context);
  const auto aerStateType = LLVM::LLVMPointerType::get(i8Type);
  const auto strType = LLVM::LLVMPointerType::get(i8Type);
  // @aer_state(...) -> i8*
  const auto aerStateFunType = LLVMFunctionType::get(aerStateType, {}, true);
  registerFunc("aer_state", aerStateFunType);
  // @aer_state_configure(i8* noundef, i8* noundef, i8* noundef) -> void
  const auto aerStateConfigureType = LLVMFunctionType::get(voidType,
                                                           {strType, strType, strType});
  registerFunc("aer_state_configure", aerStateConfigureType);
  // @aer_allocate_qubits(i8* noundef, i64 noundef) -> i64
  const auto aerAllocQubitsType = LLVMFunctionType::get(i64Type,
                                                        {aerStateType, i64Type});
  registerFunc("aer_allocate_qubits", aerAllocQubitsType);
  // @aer_state_initialize(i8*) -> i8*
  const auto aerStateInitType = LLVMFunctionType::get(aerStateType, {aerStateType});
  registerFunc("aer_state_initialize", aerStateInitType);
  // @aer_apply_u3(i8* noundef, i64 noundef, i64 noundef, i64 noundef) -> void
  const auto aerApplyU3Type = LLVMFunctionType::get(
      voidType, {aerStateType, i64Type, f64Type, f64Type, f64Type});
  registerFunc("aer_apply_u3", aerApplyU3Type);
  // @aer_apply_cx(i8* noundef, i64 noundef, i64 noundef) -> void
  const auto aerApplyCXType = LLVMFunctionType::get(
      voidType, {aerStateType, i64Type, i64Type});
  registerFunc("aer_apply_cx", aerApplyCXType);
  // @aer_apply_measure(i8* noundef, i64* noundef, i64 noundef) -> i64
  const auto aerMeasType = LLVMFunctionType::get(
      i64Type, {aerStateType, LLVM::LLVMPointerType::get(i64Type), i64Type});
  registerFunc("aer_apply_measure", aerMeasType);
  // @aer_state_finalize(i8* noundef) -> void
  const auto aerStateFinalizeType = LLVMFunctionType::get(voidType, aerStateType);
  registerFunc("aer_state_finalize", aerStateFinalizeType);
}

} // namespace qssc::targets::simulator::conversion
