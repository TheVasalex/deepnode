import { Matrix } from "../../Matrix";
import { ActivationFunctionNames } from "../types";
import ActivationFunction from "./ActivationFunction";

class Tanh extends ActivationFunction {
  calculate(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    for (let i = 0; i < inputRowCount; i++) {
      for (let j = 0; j < inputCollumnCount; j++) {
        const inputValue = input.get(i, j);
        const exponent = Math.exp(inputValue); // e^x
        const negativeExponent = 1 / exponent; // e^-x
        const output = (exponent - negativeExponent) / (exponent + negativeExponent);
        result.set(i, j, output);
      }
    }
    return result;
  }

  calculateDerivative(input: Matrix) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    const result = new Matrix(inputRowCount, inputCollumnCount, 0);

    const tanhResult = this.calculate(input);
    const [tanhRowCount, tanhCollumnCount] = tanhResult.getDimensions();

    for (let i = 0; i < tanhRowCount; i++) {
      for (let j = 0; j < tanhCollumnCount; j++) {
        const tanhValue = tanhResult.get(i, j);
        const output = 1 - tanhValue * tanhValue;
        result.set(i, j, output);
      }
    }

    return result;
  }

  getName() {
    return ActivationFunctionNames.tanh;
  }
}


export default new Tanh()