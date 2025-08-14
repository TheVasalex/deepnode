import { ActivationFunctionsEnum } from "../Functions"

export type LayerData = {
    weights: number[][]
    biases: number[][]
    activationFunctionName: ActivationFunctionsEnum
    inputs: number[][]
    weightedInputs: number[][]
    outputs: number[][]
    neuronCount: number
    inputCount: number
}