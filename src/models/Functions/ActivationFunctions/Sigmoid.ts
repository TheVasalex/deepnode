import { Matrix } from "../../Matrix";
import { ActivationFunctionNames } from "../types";
import ActivationFunction from "./ActivationFunction";

class Sigmoid extends ActivationFunction {
  calculate(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const inputValue = input.get(i, j);
        const output = 1 / (1 + Math.exp(inputValue * -1));
        result.set(i, j, output);
      }
    }
    return result;
  }

  calculateDerivative(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    const sigmoidResult = this.calculate(input);
    const [sigmoidRowCount, sigmoidCollumnCount] = sigmoidResult.getDimensions();

    for (let i = 0; i < sigmoidRowCount; i++) {
      for (let j = 0; j < sigmoidCollumnCount; j++) {
        const sigmoidValue = sigmoidResult.get(i, j);
        const output = sigmoidValue * (1 - sigmoidValue);
        result.set(i, j, output);
      }
    }

    return result;
  }

  getName() {
    return ActivationFunctionNames.sigmoid;
  }
}

export default new Sigmoid()