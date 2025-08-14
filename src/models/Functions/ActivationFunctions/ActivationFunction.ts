import { Matrix } from "../../Matrix";
import { ActivationFunctionsEnum } from "../types";

abstract class ActivationFunction {
  abstract calculate(input: Matrix): Matrix;

  abstract calculateDerivative(input: Matrix): Matrix;

  abstract getName(): ActivationFunctionsEnum;
}

export default ActivationFunction