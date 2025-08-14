import Passthrough from "./Passthrough";
import ReLU from "./ReLU";
import Sigmoid from "./Sigmoid";
import SoftMax from "./SoftMax";
import Tanh from "./Tanh";
import ActivationFunction from "./ActivationFunction";
import { ActivationFunctionNames, ActivationFunctionsEnum } from "../types";

const ActivationFunctions = {
  None: Passthrough,
  ReLU: ReLU,
  Sigmoid: Sigmoid,
  Tanh: Tanh,
  SoftMax: SoftMax,
};

function getActivationFunction(name: ActivationFunctionsEnum): ActivationFunction {
  switch (name) {
    case ActivationFunctionNames.none:
      return Passthrough;
      break;
    case ActivationFunctionNames.relu:
      return ReLU;
      break;
    case ActivationFunctionNames.softmax:
      return SoftMax;
      break;
    case ActivationFunctionNames.sigmoid:
      return Sigmoid;
      break;
    case ActivationFunctionNames.tanh:
      return Tanh;
      break;

    default:
      throw new Error("Unknown activation function");
      break;
  }
}

export { ActivationFunctions, ActivationFunction, getActivationFunction };
