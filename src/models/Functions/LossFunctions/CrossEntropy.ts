import { Matrix, Vector } from "../../Matrix";
import { LossFunctionNames } from "../types";
import LossFunction from "./LossFunction";

class CrossEntropy extends LossFunction {
  calculate(predictions: Vector, labels: Vector) {
    const [predictionRowCount, predictionCollumnCount] = predictions.getDimensions();
    const [trueRowCount, trueCollumnCount] = labels.getDimensions();

    if (predictionRowCount != trueRowCount || predictionCollumnCount != trueCollumnCount) {
      throw new Error("Matrices dimensions must match for cross entropy");
    }
    const result = new Matrix(predictionRowCount, predictionCollumnCount, 0) as Vector;

    for (let i = 0; i < predictionRowCount; i++) {
      for (let j = 0; j < predictionCollumnCount; j++) {
        let predictedValue = predictions.get(i, j);
        const label = labels.get(i, j);

        // Ensure that predictedValue is within the range [epsilon, 1 - epsilon] to avoid log(0)
        const epsilon = 1e-15;
        predictedValue = Math.max(epsilon, Math.min(1 - epsilon, predictedValue));

        const loss = -(label * Math.log(predictedValue));
        result.set(i, j, loss);
      }
    }
    return result;
  }

  calculateDerivative(predictions: Vector, labels: Vector) {
    const [predictionRowCount, predictionCollumnCount] = predictions.getDimensions();
    const [trueRowCount, trueCollumnCount] = labels.getDimensions();

    if (predictionRowCount != trueRowCount || predictionCollumnCount != trueCollumnCount) {
      throw new Error("Matrix dimensions must match for cross entropy");
    }
    const result = new Matrix(predictionRowCount, predictionCollumnCount, 0) as Vector;

    for (let i = 0; i < predictionRowCount; i++) {
      for (let j = 0; j < predictionCollumnCount; j++) {
        let predictedValue = predictions.get(i, j);
        const label = labels.get(i, j);

        // Ensure that predictedValue is within the range [epsilon, 1 - epsilon] to avoid label/0
        const epsilon = 1e-15;
        predictedValue = Math.max(epsilon, Math.min(1 - epsilon, predictedValue));

        const derivative = -(label / predictedValue);
        result.set(i, j, derivative);
      }
    }
    return result;
  }

  getName() {
    return LossFunctionNames.crossEntropy;
  }
}

export default new CrossEntropy()
