import { Matrix } from "../../Matrix";
import { ActivationFunctionNames } from "../types";
import ActivationFunction from "./ActivationFunction";

class Passthrough extends ActivationFunction {
  calculate(input: Matrix) {
    return input;
  }

  calculateDerivative(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    return new Matrix(inputRowCount, inputCollumnCount, 1);
  }

  getName() {
    return ActivationFunctionNames.none;
  }
}

export default new Passthrough()