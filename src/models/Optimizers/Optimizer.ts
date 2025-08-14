import Network from "../Network";
import { Matrix, Vector } from "../Matrix";
import { OptimizersEnum } from "./types";

abstract class Optimizer {
  weightGradients: Matrix[] = new Array<Matrix>();
  biasGradients: Vector[] = new Array<Vector>();

  abstract optimize(network: Network, trainInputs: Vector[], trainLabels: Vector[]): void;
  protected abstract calculateGradients(network: Network, trainInput: Vector, trainLabel: Vector): void;
  protected abstract applyGradients(network: Network): void;
  protected abstract resetGradients(): void;
  abstract getName(): OptimizersEnum;
}

export default Optimizer;
