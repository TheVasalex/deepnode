import { Matrix } from "../Matrix";

export abstract class Initializer {
    boxMullerRandom(mean = 0, stdDev = 1) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); // Convert [0,1) to (0,1)
        while (v === 0) v = Math.random(); // Convert [0,1) to (0,1)
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return (z * stdDev) + mean;
    }
    abstract initialize(matrix: Matrix, inputCount: number, outputCount: number): void;
}