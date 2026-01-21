#include "ValueRangeAnalysisPass.hpp"
#include "ModuleInterpreter.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

#include <cassert>
#include <deque>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;

namespace taffo {

static bool isLoopLatch(const llvm::Loop* L, llvm::BasicBlock* candidate) {
    llvm::SmallVector<llvm::BasicBlock*, 4> latches;
    L->getLoopLatches(latches);

    auto it = std::find(latches.begin(), latches.end(), candidate);
    return it != latches.end();
}

static bool isLoopExit(const llvm::Loop* L, llvm::BasicBlock* candidate) {
    llvm::SmallVector<llvm::BasicBlock*, 4> exits;
    L->getExitBlocks(exits);

    auto it = std::find(exits.begin(), exits.end(), candidate);
    return it != exits.end();
}

void ModuleInterpreter::interpret() {

    preSeed();
    if (FNs.size() == 0) return;

    inspect();

    size_t iteration = 1;
    size_t changes = 0;
    do {

        assemble();

        //todo: check at least one loop without trip count
        if (true) {
            tripCount();
        }

        propagate();


        ++iteration;
        if (MaxPropagation && iteration > MaxPropagation) {
            LLVM_DEBUG(tda::log() << "Propagation interrupted: after " << MaxPropagation << " iteration(s) no fixed point reached\n");
            break;
        }
    } while (changes != 0);

}

//==================================================================================================
//======================= PRESEEDING METHODS =======================================================
//==================================================================================================

void ModuleInterpreter::preSeed() {

    bool FoundVisitableFunction = false;
    for (Function& F : M) {
        if (!F.empty() && (TaffoInfo::getInstance().isStartingPoint(F)) && F.getName() == "main") {

            
            interpretFunction(&F);

            FoundVisitableFunction = true;
        }
    }

    if (!FoundVisitableFunction) {
        LLVM_DEBUG(tda::log() << " No visitable functions found.\n");
    }

}

void ModuleInterpreter::interpretFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore) {

    if (FNs.count(F)) {
        LLVM_DEBUG(tda::log() << "FN["<<F->getName()<<"] already interpreted\n");
        return;
    }

    auto InsertRes = FNs.try_emplace(F, VRAFunctionInfo(F, MAM));
    VRAFunctionInfo& VFI = InsertRes.first->second;

    if (!FunctionStore)
        FunctionStore = GlobalStore->newFnStore(*this);
    VFI.scope = FunctionScope(FunctionStore);
    curFn.push_back(F);

    llvm::BasicBlock* EntryBlock = &F->getEntryBlock();
    llvm::SmallPtrSet<llvm::BasicBlock*, 4U> VisitedSuccs;
    std::deque<llvm::BasicBlock*> worklist;
    llvm::SmallVector<llvm::Loop*> curLoop;

    worklist.push_back(EntryBlock);
    VFI.scope.BBAnalyzers[EntryBlock] = GlobalStore->newInstructionAnalyzer(*this);

    while (!worklist.empty()) {
        llvm::BasicBlock* curBlock = worklist.front();
        worklist.pop_front();

        auto CAIt = VFI.scope.BBAnalyzers.find(curBlock);
        assert(CAIt != VFI.scope.BBAnalyzers.end());
        std::shared_ptr<CodeAnalyzer> CurAnalyzer = CAIt->second;


        DEBUG_WITH_TYPE(GlobalStore->getLogger()->getDebugType(), GlobalStore->getLogger()->logBasicBlock(curBlock));
        for (llvm::Instruction& I : *curBlock) {
            if (CurAnalyzer->requiresInterpretation(&I)) {
                interpretCall(CurAnalyzer, &I);
            } else {
                //CurAnalyzer->analyzeInstruction(&I);
            }
        }

        if (curLoop.empty()) {
            LLVM_DEBUG(tda::log() << "BB["<<curBlock->getName()<<"] marked as visited\n");
            VFI.bbFlow.push_back(curBlock);
        } else {
            LLVM_DEBUG(tda::log() << "BB["<<curBlock->getName()<<"] marked as visited into the loop " << curLoop.back()->getName() << "\n");
            VFI.loops[curLoop.back()].bbFlow.push_back(curBlock);
        }

        llvm::Instruction* Term = curBlock->getTerminator();
        VisitedSuccs.clear();
        for (unsigned NS = 0; NS < Term->getNumSuccessors(); ++NS) {
            llvm::BasicBlock* nextBlock = Term->getSuccessor(NS);

            // Needed just for terminators where the same successor appears twice
            if (VisitedSuccs.count(nextBlock)) continue;
            else VisitedSuccs.insert(nextBlock);

            LLVM_DEBUG(tda::log() << "FN["<<F->getName()<<"] >> Follow path "<<curBlock->getName()<<" ==> "<<nextBlock->getName()<<": ");
            switch(followPath(VFI, curBlock, nextBlock, curLoop)) {
                case FollowingPathResponse::ENQUE_BLOCK: {
                    LLVM_DEBUG(tda::log() << "ENQUE_BLOCK.\n");
                    worklist.push_front(nextBlock);
                    break;
                }
                case FollowingPathResponse::LOOP_FORK: {
                    LLVM_DEBUG(tda::log() << "LOOP_FORK. New VRALoopInfo created, new context, header enqueued ");

                    // add also loop header to trace when go into loop into propagate phase()
                    if (curLoop.empty()) {
                        LLVM_DEBUG(tda::log() << "added loop header to bbFlow\n");
                        VFI.bbFlow.push_back(nextBlock);
                    } else {
                        LLVM_DEBUG(tda::log() << "added loop header to bbFlow of the loop " << curLoop.back()->getName() << "\n");
                        VFI.loops[curLoop.back()].bbFlow.push_back(nextBlock);
                    }

                    //here nextBlock is the new loop header
                    Loop* dstLoop = VFI.LI->getLoopFor(nextBlock);
                    curLoop.push_back(dstLoop);
                    VFI.loops.try_emplace(dstLoop, VRALoopInfo(dstLoop));
                    worklist.push_front(nextBlock);

                    break;
                }
                case FollowingPathResponse::LOOP_JOIN: {
                    LLVM_DEBUG(tda::log() << "LOOP_JOIN. Old context restored, exits enqueued.\n");
                    llvm::Loop* dstLoop = curLoop.back();
                    curLoop.pop_back();

                    SmallVector<BasicBlock*, 4> exits;
                    dstLoop->getExitBlocks(exits);
                    for (BasicBlock* EB : exits) {
                        LLVM_DEBUG(tda::log() << "enqueuing loop exit " << EB->getName() << "\n");
                        worklist.push_back(EB);
                    }

                    break;
                }
                default: {
                    LLVM_DEBUG(tda::log() << "NO ACTION.\n");
                    break;
                }
            }

            updateSuccessorAnalyzer(CurAnalyzer, Term, NS);
        }
        GlobalStore->convexMerge(*CurAnalyzer);
    }
    curFn.pop_back();
}

FollowingPathResponse ModuleInterpreter::followPath(VRAFunctionInfo info, llvm::BasicBlock* src, llvm::BasicBlock* dst, llvm::SmallVector<llvm::Loop*> nesting) const {

    llvm::Loop* srcLoop = info.LI->getLoopFor(src);
    llvm::Loop* dstLoop = info.LI->getLoopFor(dst);

    if (srcLoop && isLoopExit(srcLoop, dst)) return FollowingPathResponse::NO_ENQUE;

    llvm::SmallVector<llvm::BasicBlock *> curFlow;
    if (!srcLoop) curFlow = info.bbFlow;                                                                               // flow preso dal corpo della funzione
    else if (nesting.back() == srcLoop) curFlow = info.loops.lookup(srcLoop).bbFlow;                         //flow da analizzare preso dal loop corrente
    
    for (BasicBlock* pred : predecessors(dst)) {

        // do not analyze header predecessors if latch
        llvm::Loop* predLoop = info.LI->getLoopFor(pred);
        if (predLoop && isLoopLatch(predLoop, pred) && dst == predLoop->getHeader()) continue;
        if (dstLoop && predLoop == dstLoop->getParentLoop() && dstLoop->getHeader() == dst) continue;               // blocco prima del loop più esterno per forza visitato

        // typically loop exits path
        if (predLoop && dstLoop && info.loops.count(predLoop) && dstLoop == predLoop->getParentLoop()) {
            VRALoopInfo loopInfo = info.loops.lookup(predLoop);
            if (!loopInfo.isEntirelyVisited()) {
                LLVM_DEBUG(tda::log() << "pred loop " << predLoop->getName() << " is not entirely visited yet\n");
                return FollowingPathResponse::NO_ENQUE;
            }
        }

        auto it = std::find(curFlow.begin(), curFlow.end(), pred);
        if (it == curFlow.end()) {
            LLVM_DEBUG(tda::log() << "pred block " << pred->getName() << " is not visited yet\n");
            return FollowingPathResponse::NO_ENQUE;
        }
    }
    
    // percorso da esterno a nuovo loop
    if (dstLoop && dstLoop->getHeader() == dst && srcLoop != dstLoop) {
        if (!info.loops.count(dstLoop)) {
        return FollowingPathResponse::LOOP_FORK;
        }
    }
    
    // latch del loop che punta al suo header:
    if (dstLoop && dstLoop->getHeader() == dst && srcLoop == dstLoop && isLoopLatch(srcLoop, src)) {
        // torna loop join se il loop risulta completamente visitato, altrimenti no_enque
        LLVM_DEBUG(tda::log() << "path latch -> header (same loop) ");
        return info.loops.lookup(srcLoop).isEntirelyVisited() ? FollowingPathResponse::LOOP_JOIN : FollowingPathResponse::NO_ENQUE;
    }
    
    // path standard: se non visitato permetti, altrimenti evita
    auto it = std::find(curFlow.begin(), curFlow.end(), dst);
    return it == curFlow.end() ? FollowingPathResponse::ENQUE_BLOCK : FollowingPathResponse::NO_ENQUE;
}

void ModuleInterpreter::updateSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer, llvm::Instruction* TermInstr, unsigned SuccIdx) {
  llvm::DenseMap<llvm::BasicBlock*, std::shared_ptr<CodeAnalyzer>>& BBAnalyzers = FNs[curFn.back()].scope.BBAnalyzers;
  llvm::BasicBlock* SuccBB = TermInstr->getSuccessor(SuccIdx);

  std::shared_ptr<CodeAnalyzer> SuccAnalyzer;
  auto SAIt = BBAnalyzers.find(SuccBB);
  if (SAIt == BBAnalyzers.end()) {
    SuccAnalyzer = CurrentAnalyzer->clone();
    BBAnalyzers[SuccBB] = SuccAnalyzer;
  }
  else {
    SuccAnalyzer = SAIt->second;
    SuccAnalyzer->convexMerge(*CurrentAnalyzer);
  }

  CurrentAnalyzer->setPathLocalInfo(SuccAnalyzer, TermInstr, SuccIdx);
}


//==================================================================================================
//======================= INSPECTION METHODS =======================================================
//==================================================================================================


}
