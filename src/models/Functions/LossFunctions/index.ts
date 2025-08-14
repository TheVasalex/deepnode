import CrossEntropy from "./CrossEntropy";
import ResidualSumOfSquares from "./ResidualSumOfSquares";
import LossFunction from "./LossFunction";
import { LossFunctionNames, LossFunctionsEnum } from "../types";

const LossFunctions = {
  ResidualSumOfSquares: ResidualSumOfSquares,
  CrossEntropy: CrossEntropy,
};

function getLossFunction(name: LossFunctionsEnum): LossFunction {
  switch (name) {
    case LossFunctionNames.residualSumOfSquares:
      return ResidualSumOfSquares;
      break;
    case LossFunctionNames.crossEntropy:
      return CrossEntropy;
      break;

    default:
      throw new Error("Unknown loss function");
      break;
  }
}

export { LossFunctions, LossFunction, getLossFunction };
