import { Matrix } from "../Matrix";
import { Initializer } from "./types";

class Zeroes extends Initializer {
  initialize(matrix: Matrix, inputCount: number, outputCount: number): void {
    matrix.fill(0);
  }
}

export default new Zeroes()