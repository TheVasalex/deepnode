import { ActivationFunction } from "../Functions";
import { getActivationFunction } from "../Functions/ActivationFunctions";
import { Initiliazer as Initializer, Initializers as Initializers } from "../Initializers";
import { Matrix, Vector } from "../Matrix";
import { LayerData } from "./types";

export default class Layer {
  public weights: Matrix; // neuronCount x inputCount dimensions
  public biases: Vector; // neuronCount x 1 dimensions
  public activationFunction: ActivationFunction;
  public inputs: Vector; //inputCount x 1 dimensions
  public weightedInputs: Vector; // neuronCount x 1 dimensions
  public outputs: Vector; // neuronCount x 1 dimensions
  public neuronCount: number;
  public inputCount: number; // It's also previous layer's neuronCount

  constructor(neuronCount: number, inputCount: number, activationFunction: ActivationFunction) {
    this.weights = new Matrix(neuronCount, inputCount);
    this.biases = new Vector(neuronCount);
    this.activationFunction = activationFunction;
    this.inputs = new Vector(inputCount);
    this.weightedInputs = new Vector(neuronCount);
    this.outputs = new Vector(neuronCount);
    this.neuronCount = neuronCount;
    this.inputCount = inputCount;
  }

  initializeParameters(weightInitializer: Initializer = Initializers.Glorot, biasInitializer: Initializer = Initializers.Zeroes) {
    weightInitializer.initialize(this.weights, this.inputCount, this.neuronCount);
    biasInitializer.initialize(this.biases, this.inputCount, this.neuronCount);
  }

  calculateOutput(input: Vector) {
    /**
     * The forward pass is the process by which a neural network computes its predictions based on its parameters, specifically the weights and biases.
     * During this procedure, each layer calculates its weighted input and applies an activation function to simulate the behavior of biological neural networks.
     * For a single neuron, the weighted input is given by: weightedInput = [(weight1 * input1) + (weight2 * input2) + ... + (weightN * inputN)] + bias.
     * In this context, matrices are used to represent the various parameters, making it crucial to maintain consistent dimensions throughout the computations.
     * weights Matrix has neuronCount x inputCount dimensions, biases Vector has neuronCount x 1 dimensions, inputs Vector has inputCount x 1 dimensions
     * weightedInputs and output Vectors must have neuronCount x 1 dimensions, as each element corresponds to the output of a single neuron in the layer.
     */
    this.inputs = input; // inputCount x 1
    this.weightedInputs = this.weights.dot(this.inputs).add(this.biases); //neuronCount x 1 because: (weights Matrix * inputs Vector) + biases Vector => (neuronCount x inputCount) * (inputCount x 1) + (neuronCount x 1) [dimensions] => (neuronCount x 1) + (neuronCount x 1)
    this.outputs = this.activationFunction.calculate(this.weightedInputs); // neuronCount x 1
    return this.outputs;
  }

  calculateGradients(outputDerivatives: Vector) {
    /**
     * outputDerivatives Vector has neuronCount x 1 dimensions
     * inputs Vector has inputCount x 1 dimesions
     * weightGradients Matrix has neuronCount x inputCount dimensions
     */

    const activationDerivatives = this.activationFunction.calculateDerivative(this.weightedInputs); // neuronCount x 1
    const [activationDerivativesRows, activationDerivativesCollumns] = activationDerivatives.getDimensions();

    if (activationDerivativesRows === activationDerivativesCollumns && activationDerivativesRows > 1) {
      // In this case activationDerivatives is a Jacobian Matrix
      // activationDerivatives has neuronCount x neuronCount dimensions
      // outputDerivatives has neuronCount x 1 dimensions
      outputDerivatives = activationDerivatives.dot(outputDerivatives); // neuronCount x 1
    } else {
      // activationDerivatives has neuronCount x 1 dimensions
      // outputDerivatives has neuronCount x 1 dimensions
      outputDerivatives = activationDerivatives.dotHadamard(outputDerivatives); // neuronCount x 1
    }

    const weightGradient = outputDerivatives.dot(this.inputs.clone().transpose()); // neuronCount x inputCount
    const biasGradient = outputDerivatives.clone(); // neuronCount x 1

    const nextOutputDerivatives = this.weights.clone().transpose().dot(outputDerivatives); // inputCount x 1 (Note: inputCount is the previous layer's neuronCount)

    return [weightGradient, biasGradient, nextOutputDerivatives];
  }

  import(layerData: LayerData) {
    this.weights = new Matrix().fromArray(layerData.weights);
    this.biases = new Vector().fromArray(layerData.biases);
    this.activationFunction = getActivationFunction(layerData.activationFunctionName);
    this.inputs = new Vector().fromArray(layerData.inputs);
    this.weightedInputs = new Vector().fromArray(layerData.weightedInputs);
    this.outputs = new Vector().fromArray(layerData.outputs);
    this.neuronCount = layerData.neuronCount;
    this.inputCount = layerData.inputCount;

    return this
  }

  export() {
    const layerData: LayerData = {
      weights: this.weights.toArray(),
      biases: this.biases.toArray(),
      activationFunctionName: this.activationFunction.getName(),
      inputs: this.inputs.toArray(),
      weightedInputs: this.weightedInputs.toArray(),
      outputs: this.outputs.toArray(),
      neuronCount: this.neuronCount,
      inputCount: this.inputCount,
    };
    return layerData;
  }
}
