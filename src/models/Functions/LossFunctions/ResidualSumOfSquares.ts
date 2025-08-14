import { Matrix, Vector } from "../../Matrix";
import { LossFunctionNames } from "../types";
import LossFunction from "./LossFunction";

class ResidualSumOfSquares extends LossFunction {
  calculate(predictions: Vector, labels: Vector) {
    const [predictionRowCount, predictionCollumnCount] = predictions.getDimensions();
    const [trueRowCount, trueCollumnCount] = labels.getDimensions();

    if (predictionRowCount != trueRowCount || predictionCollumnCount != trueCollumnCount) {
      throw new Error("Matrices dimensions must match for residual sum of squares");
    }
    const result = new Matrix(predictionRowCount, predictionCollumnCount, 0) as Vector;

    for (let i = 0; i < predictionRowCount; i++) {
      for (let j = 0; j < predictionCollumnCount; j++) {
        const predictedValue = predictions.get(i, j);
        const label = labels.get(i, j);

        const loss = (label - predictedValue) * (label - predictedValue);
        result.set(i, j, loss);
      }
    }
    return result;
  }

  calculateDerivative(predictions: Vector, labels: Vector) {
    const [predictionRowCount, predictionCollumnCount] = predictions.getDimensions();
    const [trueRowCount, trueCollumnCount] = labels.getDimensions();

    if (predictionRowCount != trueRowCount || predictionCollumnCount != trueCollumnCount) {
      throw new Error("Matrices dimensions must match for residual squares");
    }
    const result = new Matrix(predictionRowCount, predictionCollumnCount, 0) as Vector;

    for (let i = 0; i < predictionRowCount; i++) {
      for (let j = 0; j < predictionCollumnCount; j++) {
        const predictedValue = predictions.get(i, j);
        const label = labels.get(i, j);

        const derivative = -2 * (label - predictedValue);
        result.set(i, j, derivative);
      }
    }
    return result;
  }

  getName() {
    return LossFunctionNames.residualSumOfSquares;
  }
}

export default new ResidualSumOfSquares()