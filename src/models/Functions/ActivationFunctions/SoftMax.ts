import { Matrix, Vector } from "../../Matrix";
import { ActivationFunctionNames } from "../types";
import ActivationFunction from "./ActivationFunction";

class SoftMax extends ActivationFunction {
  calculate(input: Vector) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = input.clone();
    let epsilons = 0;

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const value = input.get(i, j);
        epsilons = epsilons + Math.exp(value);
      }
    }

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const value = input.get(i, j);
        const output = Math.exp(value) / epsilons;
        result.set(i, j, output);
      }
    }
    return result;
  }

  calculateDerivative(input: Vector) {
    const softmaxResult = this.calculate(input);
    if (softmaxResult.getOrientation() == "horizontal") softmaxResult.transpose();
    const [softmaxRowCount, softmaxCollumnCount] = softmaxResult.getDimensions();
    const result = new Matrix(softmaxRowCount, softmaxRowCount, 0);

    for (let i = 0; i < softmaxRowCount; i++) {
      for (let j = 0; j < softmaxRowCount; j++) {
        const output = softmaxResult.get(i, 0) * ((i === j ? 1 : 0) - softmaxResult.get(j, 0));
        result.set(i, j, output);
      }
    }

    return result;
  }

  getName() {
    return ActivationFunctionNames.softmax;
  }
}

export default new SoftMax()