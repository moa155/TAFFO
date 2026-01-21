#pragma once
#include "CodeInterpreter.hpp"
#include "VRALogger.hpp"
#include "RangedRecurrences.hpp"

#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <memory>

namespace taffo {

class Range;

enum FollowingPathResponse {
    ENQUE_BLOCK, NO_ENQUE, LOOP_JOIN, LOOP_FORK
};

struct VRALoopInfo {
    llvm::Loop* L;
    u_int64_t TripCount;
    llvm::Value* InductionVariable;

    llvm::SmallVector<llvm::BasicBlock*> bbFlow;

    bool isInvariant(const llvm::Value* V) { return L->isLoopInvariant(V); }

    /// @brief check if all blocks of the loop are visited
    /// @return true is whole visited (all latches are visited), false otherwise
    bool isEntirelyVisited() {
        llvm::SmallVector<llvm::BasicBlock*, 4> latches;
        L->getLoopLatches(latches);
        
        for (auto latch : latches) {
            if (std::find(bbFlow.begin(), bbFlow.end(), latch) == bbFlow.end()) return false;
        }
        return true;
    }

    VRALoopInfo() : L(nullptr) {}
    VRALoopInfo(llvm::Loop* L) : L(L) {}
};

struct VRARecurrenceInfo {
    const llvm::Value* root;
    llvm::SmallVector<llvm::Value*> chain;

    llvm::SmallVector<llvm::Function*> depsOnFn;
    llvm::SmallVector<llvm::Value*> depsOnRR;

    llvm::DenseMap<llvm::Value*, std::shared_ptr<RangedRecurrence>> RRs;
    std::shared_ptr<Range> lastRange;
    u_int64_t lastRangeComputedAt = 0;
};

//todo: manca lo scope corrente
struct VRAFunctionInfo {
    llvm::Function* F;
    llvm::SmallVector<llvm::BasicBlock*> bbFlow;
    llvm::DenseMap<llvm::Loop*, VRALoopInfo> loops;
    llvm::DenseMap<llvm::Value*, VRARecurrenceInfo> RRs;
    FunctionScope scope;

    std::shared_ptr<Range> lastRange;
    llvm::DenseMap<llvm::Argument*, Range> lastRangeArgs;

    llvm::DominatorTree* DT;
    llvm::LoopInfo* LI;
    llvm::ScalarEvolution* SE;

    VRAFunctionInfo(): F(nullptr) {}
    VRAFunctionInfo(llvm::Function* F, llvm::ModuleAnalysisManager& MAM): F(F) {
        auto& FAM = MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(*F->getParent()).getManager();
        LI = &(FAM.getResult<llvm::LoopAnalysis>(*F));
        DT = &(FAM.getResult<llvm::DominatorTreeAnalysis>(*F));
        SE = &(FAM.getResult<llvm::ScalarEvolutionAnalysis>(*F));

        //todo: argument handlers??
    }
};

class ModuleInterpreter {
public:

    std::shared_ptr<AnalysisStore> getStoreForValue(const llvm::Value* V) const;
    std::shared_ptr<AnalysisStore> getGlobalStore() const { return GlobalStore; }
    std::shared_ptr<AnalysisStore> getFunctionStore() const {
        if (curFn.empty())
            return nullptr;
        auto It = FNs.find(curFn.back());
        if (It == FNs.end())
            return nullptr;
        return It->second.scope.FunctionStore;
    }
    llvm::ModuleAnalysisManager& getMAM() const { return MAM; }

    void interpret();

    ModuleInterpreter(llvm::Module& M, llvm::ModuleAnalysisManager& MAM, std::shared_ptr<AnalysisStore> GlobalStore): M(M), GlobalStore(GlobalStore), curFn(), MAM(MAM), FNs() {}

protected:

    // 1) PRESEED METHODS
    void preSeed();
    void interpretFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore = nullptr);
    FollowingPathResponse followPath(VRAFunctionInfo info, llvm::BasicBlock* src, llvm::BasicBlock* dst, llvm::SmallVector<llvm::Loop*> nesting) const;
    void interpretCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer, llvm::Instruction* I);
    void updateSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer, llvm::Instruction* TermInstr, unsigned SuccIdx);

    // 2) INSPECTION PHASE METHODS
    void inspect();
    void handlePHIChain(llvm::Loop* L, const llvm::PHINode* PHI, VRARecurrenceInfo& VRI);
    void handleStoreChain(llvm::Loop* L, const llvm::StoreInst* Store, VRARecurrenceInfo& VRI);

    // 3) ASSEMBLING METHODS
    void assemble();
    bool isFakeRecurrence(VRARecurrenceInfo& VRI);
    bool isAffineRecurrence(VRARecurrenceInfo& VRI);
    bool isDeltaAffineRecurrence(VRARecurrenceInfo& VRI);
    bool isGeometricRecurrence(VRARecurrenceInfo& VRI);
    bool isDeltaGeometricRecurrence(VRARecurrenceInfo& VRI);
    void fallbackRecurrence(VRARecurrenceInfo& VRI);
    // add here new recurrences...

    // 4) TRIP COUNT METHODS
    void tripCount();

    // 5) PROPAGATION METHODS
    void propagate();

private:

    llvm::Module& M;
    std::shared_ptr<AnalysisStore> GlobalStore;
    llvm::SmallVector<llvm::Function*, 4U> curFn;    //current function scope
    llvm::ModuleAnalysisManager& MAM;

    llvm::DenseMap<llvm::Function*, VRAFunctionInfo> FNs;

};

}
