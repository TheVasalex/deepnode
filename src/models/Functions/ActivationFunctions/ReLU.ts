import { Matrix } from "../../Matrix";
import { ActivationFunctionNames } from "../types";
import ActivationFunction from "./ActivationFunction";

class ReLU extends ActivationFunction {
  calculate(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const inputValue = input.get(i, j);
        const output = inputValue > 0 ? inputValue : 0;
        result.set(i, j, output);
      }
    }
    return result;
  }

  calculateDerivative(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const inputValue = input.get(i, j);
        const output = inputValue > 0 ? 1 : 0;
        result.set(i, j, output);
      }
    }
    return result;
  }

  getName() {
    return ActivationFunctionNames.relu;
  }
}

export default new ReLU()