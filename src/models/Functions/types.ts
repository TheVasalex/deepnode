import { Matrix, Vector } from "../Matrix";

export interface ActivationFunction {
  new (): void; // Constructor signature
  calculate(input: Matrix): Matrix;
  calculateDerivative(input: Matrix): Matrix;
  getName(): ActivationFunctionsEnum
}

export interface LossFunction {
  new (): void; // Constructor signature
  calculate(predictions: Vector, labels: Vector): Vector;
  calculateDerivative(predictions: Vector, labels: Vector): Vector;
  getName(): LossFunctionsEnum
}

export const ActivationFunctionNames = {
  none: "none",
  relu: "relu",
  sigmoid: "sigmoid",
  tanh: "tanh",
  softmax: "softmax"
} as const;
export type ActivationFunctionsEnum = (typeof ActivationFunctionNames)[keyof typeof ActivationFunctionNames];

export const LossFunctionNames = {
  residualSumOfSquares: "rss",
  crossEntropy: "ce"
} as const;
export type LossFunctionsEnum = (typeof LossFunctionNames)[keyof typeof LossFunctionNames];
