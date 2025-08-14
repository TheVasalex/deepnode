import { Vector } from "../../Matrix";
import { LossFunctionsEnum } from "../types";

abstract class LossFunction {
  abstract calculate(prediction: Vector, label: Vector): Vector;

  abstract calculateDerivative(prediction: Vector, label: Vector): Vector;

  abstract getName(): LossFunctionsEnum;
}

export default LossFunction