/**
 * A 2D Matrix.
 */
export class Matrix {
  protected matrix: number[][];
  protected rowCount: number;
  protected collumnCount: number;
  /**
   * @param rowCount Default = 1
   * @param collumnCount Default = 1
   * @param defaultValue The value which the Matrix will be filled upon creation. Default = 0
   */
  constructor(rowCount: number = 1, collumnCount: number = 1, defaultValue: number = 0) {
    this.rowCount = rowCount <= 0 ? 1 : rowCount;
    this.collumnCount = collumnCount <= 0 ? 1 : collumnCount;
    this.matrix = new Array(this.rowCount);

    for (let i = 0; i < rowCount; i++) {
      this.matrix[i] = new Array(this.collumnCount);
      for (let j = 0; j < collumnCount; j++) {
        this.matrix[i][j] = defaultValue;
      }
    }
  }

  public get(row: number, collumn: number) {
    if (row < 0 || collumn < 0 || row > this.rowCount - 1 || collumn > this.collumnCount - 1) {
      throw new Error("Matrix limits exceeded. Cannot get value");
    }
    return this.matrix[row][collumn];
  }

  public set(row: number, collumn: number, value: number) {
    if (typeof value != "number") throw new Error("Invalid value type. Only numbers are allowed to be set");
    if (row < 0 || collumn < 0 || row > this.rowCount - 1 || collumn > this.collumnCount - 1) {
      throw new Error("Matrix limits exceeded. Cannot set value");
    }
    this.matrix[row][collumn] = value;
    return this;
  }

  public getDimensions() {
    return [this.rowCount, this.collumnCount];
  }

  /**
   * Create a copy of the Matrix, which can be mutated seperately.
   */
  public clone() {
    return new Matrix().fromArray(this.matrix);
  }

  /**
   * Print the Matrix to the console for debugging purposes.
   */
  public print() {
    console.table(this.matrix);
  }

  public fromArray(matrixArray: number[] | number[][]) {
    if (!Array.isArray(matrixArray) || matrixArray.length === 0) throw new Error("Invalid input. Make sure it's a valid number array");

    let numberStreak = 0;
    let childItemCount = -1;
    for (let i = 0; i < matrixArray.length; i++) {
      if (numberStreak >= 0 && typeof matrixArray[i] === "number") {
        numberStreak++;
        continue;
      }

      const child = matrixArray[i];
      if (Array.isArray(child)) {
        numberStreak = -1;
        if (child.length === 0) throw new Error("Invalid input. Make sure it's a valid number array");
        if (childItemCount === -1) childItemCount = child.length;
        if (childItemCount === child.length) {
          const areNumbers = child.every((item) => typeof item === "number");
          if (areNumbers) continue;
        }
        throw new Error("Invalid input. Make sure it's a valid number array");
      }
    }
    const is1D = numberStreak > 0;

    const inputArray = (is1D ? [matrixArray] : matrixArray) as number[][];
    const rowCount = inputArray.length;
    const collumnCount = inputArray[0].length;

    this.matrix = new Array(rowCount);

    for (let i = 0; i < rowCount; i++) {
      this.matrix[i] = new Array(collumnCount);
      for (let j = 0; j < collumnCount; j++) {
        this.matrix[i][j] = inputArray[i][j];
      }
    }
    this.rowCount = rowCount;
    this.collumnCount = collumnCount;
    return this;
  }

  public toArray() {
    return this.matrix;
  }

  public getOrientation() {
    if (this.rowCount > this.collumnCount) return "vertical";
    if (this.rowCount < this.collumnCount) return "horizontal";
    return "square";
  }

  public getSum() {
    let sum = 0;
    for (let i = 0; i < this.rowCount; i++) {
      for (let j = 0; j < this.collumnCount; j++) {
        sum += this.get(i, j);
      }
    }
    return sum;
  }
  public getAvg() {
    return this.getSum() / (this.rowCount * this.collumnCount);
  }

  public fill(value: number) {
    for (let i = 0; i < this.rowCount; i++) {
      for (let j = 0; j < this.collumnCount; j++) {
        this.set(i, j, value);
      }
    }
    return this
  }

  // Math Operations

  /**
 * Transform rows to collumns and collumns to rows.
 * Mutates the Matrix.
 */
  public transpose() {
    const clone = this.clone();
    const [rowCount, collumnCount] = clone.getDimensions();
    this.matrix = [];
    for (let j = 0; j < this.collumnCount; j++) {
      this.matrix.push(new Array(this.rowCount));
      for (let i = 0; i < this.rowCount; i++) {
        this.matrix[j][i] = clone.get(i, j);
      }
    }
    this.rowCount = collumnCount;
    this.collumnCount = rowCount;
    return this;
  }

  public add(operand: Matrix | number) {
    if (operand instanceof Matrix) return this.addMatrix(operand);
    if (typeof operand != "number") throw new Error("Invalid operand type");
    const resultMatrix = new Matrix(this.rowCount, this.collumnCount, 0);
    for (let i = 0; i < this.rowCount; i++) {
      for (let j = 0; j < this.collumnCount; j++) {
        const currentValue = this.get(i, j);
        resultMatrix.set(i, j, currentValue + operand);
      }
    }
    return resultMatrix;
  }

  protected addMatrix(otherMatrix: Matrix) {
    const dimA = this.getDimensions();
    const dimB = otherMatrix.getDimensions();

    if (dimA[0] !== dimB[0] || dimA[1] !== dimB[1]) {
      throw new Error(`Matrices dimensions must match for addition. DimA: ${dimA.toString()} DimB: ${dimB.toString()}`);
    }
    const resultMatrix = new Matrix(dimA[0], dimA[1], 0);

    for (let i = 0; i < dimA[0]; i++) {
      for (let j = 0; j < dimA[1]; j++) {
        const result = this.get(i, j) + otherMatrix.get(i, j);
        resultMatrix.set(i, j, result);
      }
    }
    return resultMatrix;
  }

  public dot(operand: Matrix | number) {
    if (operand instanceof Matrix) return this.dotMatrix(operand);
    if (typeof operand != "number") throw new Error("Invalid operand type");
    const resultMatrix = new Matrix(this.rowCount, this.collumnCount, 0);
    for (let i = 0; i < this.rowCount; i++) {
      for (let j = 0; j < this.collumnCount; j++) {
        const currentValue = this.get(i, j);
        resultMatrix.set(i, j, currentValue * operand);
      }
    }
    return resultMatrix;
  }

  protected dotMatrix(otherMatrix: Matrix) {
    const dimA = this.getDimensions();
    const dimB = otherMatrix.getDimensions();
    if (dimA[1] !== dimB[0]) {
      throw new Error(`Matrices dimensions are incompatible for multiplication. DimA: ${dimA.toString()} DimB: ${dimB.toString()}`);
    }
    const resultMatrix = new Matrix(dimA[0], dimB[1], 0);

    for (let i = 0; i < dimA[0]; i++) {
      for (let j = 0; j < dimA[1]; j++) {
        for (let k = 0; k < dimB[1]; k++) {
          const buffer = resultMatrix.get(i, k);
          const product = this.get(i, j) * otherMatrix.get(j, k);
          resultMatrix.set(i, k, buffer + product);
        }
      }
    }
    return resultMatrix;
  }

  public dotHadamard(otherMatrix: Matrix) {
    const [inputRowCount, inputCollumnCount] = otherMatrix.getDimensions();
    if (inputRowCount != this.rowCount || inputCollumnCount != this.collumnCount) {
      throw new Error(
        `Matrices dimensions are incompatible for hadamard product. DimA: ${this.rowCount},${this.collumnCount} DimB: ${inputRowCount},${inputCollumnCount}`
      );
    }
    const resultMatrix = new Matrix(this.rowCount, this.collumnCount, 0);
    for (let i = 0; i < this.rowCount; i++) {
      for (let j = 0; j < this.collumnCount; j++) {
        const hadamardProduct = this.get(i, j) * otherMatrix.get(i, j);
        resultMatrix.set(i, j, hadamardProduct);
      }
    }
    return resultMatrix;
  }
}
/**
 * A 1 x n or n x 1 Matrix.
 * By default it's n x 1
 */
export class Vector extends Matrix {
  /**
   * Create a collumn vector
   * @param rowCount Default = 1
   * @param defaultValue The value which the Vector will be filled upon creation. Default = 0
   */
  constructor(rowCount: number = 1, defaultValue: number = 0) {
    super(rowCount, 1, defaultValue);
  }
  /**
   * Create a copy of the Vector, which can be mutated seperately.
   */
  public clone(): Vector {
    return super.clone() as Vector;
  }
}
