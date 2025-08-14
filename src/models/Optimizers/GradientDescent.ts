import { getTimeString } from "../../Utilities";
import { Matrix, Vector } from "../Matrix";
import Network from "../Network";
import Optimizer from "./Optimizer";
import { GradientDescentOpts, OptimizerNames, OptimizersEnum } from "./types";

class GradientDescent extends Optimizer {
  learnRate: number;
  momentumFactor: number;
  batchSize: number;
  targetEpoch: number;
  targetScore: number;
  weightGradients: Matrix[] = new Array();
  biasGradients: Vector[] = new Array();
  weightVelocities: Matrix[] = new Array();
  biasVelocities: Vector[] = new Array();

  constructor({ learnRate = 0.001, momentumFactor = 0, batchSize = 1, targetEpoch = 100, targetScore = -1 }: GradientDescentOpts) {
    super();
    this.learnRate = learnRate;
    this.momentumFactor = momentumFactor;
    this.batchSize = batchSize;
    this.targetEpoch = targetEpoch;
    this.targetScore = targetScore;
  }

  optimize(network: Network, trainInputs: Vector[], trainLabels: Vector[]): void {
    // Initialize Matrices
    this.weightGradients = new Array(network.layers.length);
    this.biasGradients = new Array(network.layers.length);

    this.weightVelocities = new Array(network.layers.length);
    this.biasVelocities = new Array(network.layers.length);

    for (let i = 0; i < network.layers.length; i++) {
      this.weightGradients[i] = new Matrix(network.layers[i].neuronCount, network.layers[i].inputCount);
      this.biasGradients[i] = new Vector(network.layers[i].neuronCount);

      this.weightVelocities[i] = new Matrix(network.layers[i].neuronCount, network.layers[i].inputCount);
      this.biasVelocities[i] = new Vector(network.layers[i].neuronCount);
    }

    // Optimize
    let currentEpoch = 1;
    let currentScore = 9999;
    while ((this.targetEpoch != -1 && currentEpoch < this.targetEpoch) || currentScore > this.targetScore) {
      // An epoch starts here
      for (let i = 0; i < trainInputs.length; i++) {
        this.calculateGradients(network, trainInputs[i], trainLabels[i]);
        if (i % this.batchSize === 0 || i === trainInputs.length - 1) {
          this.applyGradients(network);
          this.resetGradients();
        }
      }
      currentScore = network.evaluate(trainInputs, trainLabels);
      console.log(`[${getTimeString()}] [Epoch ${currentEpoch}] [Score ${currentScore}]`);
      // An epoch ends here
      currentEpoch++;
    }
    console.log(`[${getTimeString()}] Training Completed!`);
  }

  protected calculateGradients(network: Network, trainInput: Vector, trainLabel: Vector): void {
    /**
     * The gradient is a fundamental mathematical concept used to quantify how small changes in a parameter influence the output of a function.
     * In the context of neural networks, it helps determine how slight adjustments to weights and biases affect the output of the loss function, also known as the total error.
     * To compute these gradients, one must answer a key question "how does a change in a given weight/bias impact the total error?"
     * The relationship can be broken down into a sequence of dependencies:
     * A weight/bias influences the weighted input, which in turn affects the layer's output, ultimately impacting the total error.
     * There is a slight difference in how gradients are calculated depending on the layer in question.
     * The last layer of the neural network influences the total error directly, while all the other layer's output, influences the weighted input of the next layer.
     * Due to this structure, the computation of gradients begins at the last layer and propagates backward through the network until the first layer, a process known as backpropagation.
     * Mathematically, this process is described by the chain rule of calculus. Each step in this chain corresponds to a partial derivative.
     * So, for the last layer:
     * (1) "The weight/bias affects the weighted input" = The partial derivative of the weighted input with respect to the weight.
     * (2) "which affects the layer's output" = The partial derivative of the layer's output with respect to the weighted input.
     * (3) "which affects the total Error" = The partial derivative of the total error with respect to the layer's output.
     * (1) * (2) * (3) = The partial derivative of the total error with respect to the weight/bias = how a small change in the weight/bias, affects the total error.
     */

    const out = network.predict(trainInput);
    const lastLayerIndex = network.layers.length - 1;
    const lastLayer = network.layers[lastLayerIndex];
    let outputDerivatives = new Vector(lastLayer.neuronCount);

    const lossDerivatives = network.lossFunction.calculateDerivative(lastLayer.outputs, trainLabel);
    const activationDerivatives = lastLayer.activationFunction.calculateDerivative(lastLayer.weightedInputs);
    const [activationDerivativesRows, activationDerivativesCollumns] = activationDerivatives.getDimensions();

    /**
     * Certain activation functions (such as Softmax), depend on all inputs to determine the output.
     * Consequently, their derivatives form a matrix where each element represents the partial derivative of one input with respect to another.
     * This matrix is called Jacobian matrix.
     * This is the reason why a check is needed to determine the activationDerivatives type.
     * Ultimately calculating the outputDerivatives, by using the suitable matrix operation for each case.
     */

    if (activationDerivativesRows === activationDerivativesCollumns && activationDerivativesRows > 1) {
      // In this case activationDerivatives is a Jacobian Matrix
      // activationDerivatives has neuronCount x neuronCount dimensions
      // lossDerivatives has neuronCount x 1 dimensions
      outputDerivatives = activationDerivatives.dot(lossDerivatives); // neuronCount x 1
    } else {
      // activationDerivatives has neuronCount x 1 dimensions
      // lossDerivatives has neuronCount x 1 dimensions
      outputDerivatives = activationDerivatives.dotHadamard(lossDerivatives); // neuronCount x 1
    }

    // Calculate Gradients for the last layer

    // outputDerivatives has neuronCount x 1 dimensions
    // inputs has inputCount x 1 dimensions, so a transpose is required
    const lastWeightGradient = outputDerivatives.dot(lastLayer.inputs.clone().transpose()); // neuronCount x inputCount
    const lastBiasGradient = outputDerivatives.clone(); // neuronCount x 1

    this.weightGradients[lastLayerIndex] = this.weightGradients[lastLayerIndex].add(lastWeightGradient);
    this.biasGradients[lastLayerIndex] = this.biasGradients[lastLayerIndex].add(lastBiasGradient);

    /**
     * In order for all the other layers to calculate their gradients, they need to know how their output affects the next layer's weighted input.
     * outputDerivatives already includes the information about how the next layer's (in this case, last layer) weightedInput affects the total error.
     * By multipling the last layer's weights with the outputDerivatives Vector (chain rule) , we find out how the previous layer's output affects the total error (the partial derivative of the total error with respect to the previous layer's output)
     */

    // outputDerivatives has neuronCount x 1 dimensions
    // weights has neuronCount x inputCount dimensions, so a transpose is required
    let nextOutputDerivatives = lastLayer.weights.clone().transpose().dot(outputDerivatives); // inputCount x 1 (Note: inputCount is the previous layer's neuronCount)

    // Calculate Gradients for all the other layers
    for (let i = 1; i < network.layers.length; i++) {
      const index = network.layers.length - i - 1;
      const currentLayer = network.layers[index];
      const [weightGradient, biasGradient, newOutputDerivatives] = currentLayer.calculateGradients(nextOutputDerivatives);
      nextOutputDerivatives = newOutputDerivatives

      this.weightGradients[index] = this.weightGradients[index].add(weightGradient);
      this.biasGradients[index] = this.biasGradients[index].add(biasGradient);
    }
  }

  protected applyGradients(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      /**
       * parameterVelocity = (momentumFactor * parameterVelocity) - (learnRate * parameterGradient)
       * parameter = parameter + parameterVelocity
       */

      const weightVelocityFactor1 = this.weightVelocities[i].dot(this.momentumFactor);
      const weightVelocityFactor2 = this.weightGradients[i].dot(-1 * this.learnRate * (1 / this.batchSize));
      this.weightVelocities[i] = weightVelocityFactor1.add(weightVelocityFactor2);

      const biasVelocityFactor1 = this.biasVelocities[i].dot(this.momentumFactor);
      const biasVelocityFactor2 = this.biasGradients[i].dot(-1 * this.learnRate * (1 / this.batchSize));
      this.biasVelocities[i] = biasVelocityFactor1.add(biasVelocityFactor2);

      network.layers[i].weights = network.layers[i].weights.add(this.weightVelocities[i]);
      network.layers[i].biases = network.layers[i].biases.add(this.biasVelocities[i]);
    }
  }

  protected resetGradients(): void {
    // weightGradients and biasGradients have the same length
    for (let i = 0; i < this.weightGradients.length; i++) {
      this.weightGradients[i].fill(0);
      this.biasGradients[i].fill(0);
    }
  }

  getName(): OptimizersEnum {
    return OptimizerNames.gradientDescent
  }
}

export default GradientDescent