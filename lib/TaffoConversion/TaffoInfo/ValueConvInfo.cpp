#include "ValueConvInfo.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

std::string ValueConvInfo::toString() const {
  std::stringstream ss;
  ss << "{ ";
  ss << "oldType: " << (oldType ? oldType->toString() : "null") << ", ";
  if (constant)
    ss << "constant, ";
  if (conversionDisabled)
    ss << "disabled, ";
  else if (newType)
    ss << "newType: " << *newType << ", ";
  if (isConverted)
    ss << "converted, ";
  if (forceType)
    ss << "forceType, ";
  ss << (isArgumentPlaceholder ? "argPlaceholder, " : "");
  ss << (isBacktrackingNode ? "backtracking, " : "");
  if (isRoot)
    ss << "root, ";
  else {
    ss << "roots: { ";
    bool first = true;
    for (Value* root : roots) {
      if (!first)
        ss << ", ";
      // NOTE: parentheses are required because operator<< binds tighter than
      // the ternary ?:. Without them the name would never actually reach `ss`.
      ss << (root->getName().empty() ? std::string("<unnamed>") : root->getName().str());
      first = false;
    }
    ss << " }";
  }
  ss << " }";
  return ss.str();
}
