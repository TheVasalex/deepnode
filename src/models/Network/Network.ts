import { ActivationFunction, ActivationFunctions, LossFunction, LossFunctions } from "../Functions";
import { getLossFunction } from "../Functions/LossFunctions";
import { Initiliazer, Initializers } from "../Initializers";
import Layer from "../Layer";
import { LayerData } from "../Layer/types";
import { Vector } from "../Matrix";
import { NetworkData } from "./types";

export default class Network {
  public layers: Layer[] = new Array();
  public lossFunction: LossFunction = LossFunctions.ResidualSumOfSquares;
  constructor() {}

  addLayer(neuronCount: number, activationFunction: ActivationFunction) {
    const previousLayer = this.layers[this.layers.length - 1];
    let prevNeuronCount = 0;
    if (previousLayer != undefined) {
      prevNeuronCount = previousLayer.neuronCount;
      if (previousLayer.inputCount == 0) this.layers = new Array();
    }
    const layer = new Layer(neuronCount, prevNeuronCount, activationFunction);
    this.layers.push(layer);
  }

  setLossFunction(lossFunction: LossFunction) {
    this.lossFunction = lossFunction;
  }

  initializeParameters(weightInitiliazer: Initiliazer = Initializers.Glorot, biasInitiliazer: Initiliazer = Initializers.Zeroes) {
    for (const layer of this.layers) {
      layer.initializeParameters(weightInitiliazer, biasInitiliazer);
    }
  }

  predict(input: Vector) {
    const [inputRowCount, inputCollumnCount] = input.getDimensions();
    if (inputRowCount != 1 && inputCollumnCount != 1) throw new Error(`Invalid input for prediction. Provide a Vector`);
    let output = input.clone();
    if (output.getOrientation() === "horizontal") output.transpose();

    for (let i = 0; i < this.layers.length; i++) {
      output = this.layers[i].calculateOutput(output);
    }
    return output;
  }

  evaluate(inputs: Vector[], labels: Vector[]) {
    const scores = new Vector(inputs.length);
    for (let i = 0; i < inputs.length; i++) {
      const prediction = this.predict(inputs[i]);
      const score = this.lossFunction.calculate(prediction, labels[i]).getSum(); // total error
      scores.set(i, 0, score);
    }
    return scores.getAvg();
  }

  import(networkData: NetworkData) {
    this.lossFunction = getLossFunction(networkData.lossFunctionName)
    this.layers = new Array<Layer>()
    for (const layerData of networkData.layers) {
      const layer = new Layer(0,0,ActivationFunctions.None).import(layerData)
      this.layers.push(layer)
    }
    return this
  }

  export() {
    const networkData: NetworkData = {
      layers: new Array<LayerData>(),
      lossFunctionName: this.lossFunction.getName(),
    };
    this.layers.forEach((layer) => networkData.layers.push(layer.export()));
    return networkData
  }
}
