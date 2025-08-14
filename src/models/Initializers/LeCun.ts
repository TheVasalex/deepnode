import { Matrix } from "../Matrix";
import { Initializer } from "./types";

class LeCun extends Initializer {
    initialize(matrix: Matrix, inputCount: number, outputCount: number): void {
        const [rowCount, collumnCount] = matrix.getDimensions()
        const scale = Math.sqrt(1 / inputCount); // Yann LeCun formula
        for (let i = 0; i < rowCount; i++) {
            for (let j = 0; j < collumnCount; j++) {
              const randomValue = super.boxMullerRandom(0,scale)
              matrix.set(i, j, randomValue);
            }
          }
    }
    
}

export default new LeCun()