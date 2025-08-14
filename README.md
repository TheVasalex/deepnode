# DeepNode | Thesis Project

<div align="center">
  <img height="200" src="https://github.com/TheVasalex/deepnode/blob/main/images/uth.jpg"  />
</div>

DeepNode is a comprehensive, object-oriented implementation of neural network concepts built with Node.js and TypeScript. Instead of relying on high-level, optimized libraries like TensorFlow or PyTorch, this project was engineered to build the entire stack, from matrix operations to backpropagation, from the ground up in a modular library format. This approach provides unparalleled clarity into the mathematics powering machine learning, making it an exceptional educational tool and a solid foundation for custom ML algorithms.

## Abstract
The modern era, is defined by a technological revolution driven by the emergence of
autonomous systems, operated through machine learning and artificial intelligence. A
foundational element of these fields is neural networks. A thorough understanding of their
operational mechanisms, is critically important for proper utilization and the development
of improved technologies that will shape the future. To this end, a customized library has
been implemented, aiming to create versatile neural networks, using TypeScript alongside the Node.js runtime. During development, the mathematical principles underlying
neural networks were detailed, including calculus concepts such as linear and non-linear
functions, derivatives, partial derivatives, the chain rule, and linear algebra topics like
matrices and their properties. The examination focused, on how these mathematical concepts are combined in neural network algorithms, leading to the development of concepts
like neurons and network layers, weights, biases, activation and loss functions and the
process of training a neural network through backpropagation and gradient descent optimization algorithms. Subsequently, a software architecture plan was created, emphasizing the importance of object-oriented programming, in such a project and the role of each
object model forming the library. The development process, highlighted the significance
of a deep understanding of the operational mechanisms of neural networks, focusing on
making informed decisions, performing preliminary dataset analysis, and understanding
the consequences of negligence. Issues, such as exploding and vanishing gradients were
identified.

## Technical Presentation
![Overview](https://github.com/TheVasalex/deepnode/blob/main/images/high-level-overview.png)

The Matrix and Vector classes are the fundamental classes of the library, which will implement all the necessary methods for performing linear algebra operations using matrices and vectors, respectively. The Vector class will be a child of the Matrix class through inheritance.

![Matrix](https://github.com/TheVasalex/deepnode/blob/main/images/matrix.png)

One of the fundamental building blocks of a neural network is its layered structure. A neural network is composed of multiple layers, each containing a number of neurons that act as computational units. Every neuron is connected to neurons in the following layer through weighted connections. These weights determine the importance of the signals passed between neurons. The role of a layer is to manage these connections and perform the computations required to produce the output of each neuron, which together form the output of the layer. In the library, all of these properties and operations will be handled internally by the Layer class, eliminating the need to implement separate logic for each neuron. Consequently, this class serves as the core component of the library.

![Layer](https://github.com/TheVasalex/deepnode/blob/main/images/layer.png)

All layers will be orchestrated via the Network class, which is the entity that the user can adjust and control.

![Network](https://github.com/TheVasalex/deepnode/blob/main/images/network.png)

Parameter initialization is a crucial step in designing a machine learning model, so the library should support it. To ensure modularity and maintainability, the Initializer class is defined as an abstract class, that specifies the interface each initializer must implement. This approach, allows multiple initialization strategies to be added without affecting the rest of the codebase.

Currently, the supported Initializers are:
- LeCun
- He
- Glorot
- Zeroes

![Initializer](https://github.com/TheVasalex/deepnode/blob/main/images/initializer.png)

Key concept of the data processing is the Activation Function. This function, is a special mathematical function, which is used to determine the activation amount of a neuron, attempting to simulate the way that a biological neurons functions. Since, there are several functions available, for different problem solving scenarios, a central parent abstract class is used to describe the methods.

Currently, the supported Activation Functions are:
- ReLU
- Sigmoid
- Tanh
- SoftMax
- None

![ActivationFunction](https://github.com/TheVasalex/deepnode/blob/main/images/activationFunction.png)

The same concept applies to the Loss Function. The difference is, that the Loss Function is used during training to evaluate the model's error.
The LossFunction is defined at the Network level (see examples below).

Currently, the supported Loss Functions are:
- SSR (Sum of Squared Residuals)
- CrossEntropy (Categorical Cross Entropy)

![LossFunction](https://github.com/TheVasalex/deepnode/blob/main/images/lossFunction.png)

Generaly, the goal of the training procedure, is to minimize the output of the Loss Function, while avoiding issues such as overfitting. This is achieved, by updating the network parameters in a specific way which is called optimization. There are various optimization techniques available, so the abstract Optimizer class is defined to describe the methods which need to be implemented by any optimizer, without interfering with previous or future algorithms.

Currently supported Optimizer is: Gradient Descent (With Momentum Options)

![Optimizer](https://github.com/TheVasalex/deepnode/blob/main/images/optimizer.png)


## Usage
### Training Example
```javascript
const {ActivationFunctions, LossFunctions , Initializers , Network, Optimizers, parseCSV} = require("deepnode")
const fs = require("node:fs")

const neuralNetwork = new Network();

neuralNetwork.addLayer(2, ActivationFunctions.None);

neuralNetwork.addLayer(6, ActivationFunctions.Sigmoid);
neuralNetwork.addLayer(6, ActivationFunctions.Sigmoid);

neuralNetwork.addLayer(1, ActivationFunctions.Tanh);

neuralNetwork.setLossFunction(LossFunctions.ResidualSumOfSquares);
neuralNetwork.initializeParameters(Initializers.He, Initializers.Zeroes);

const optimiser = new Optimizers.GradientDescent({
  learnRate: 0.1,
  batchSize: 4,
  targetEpoch: -1, // Until targetScore is reached
  targetScore: 0.0005,
  momentumFactor: 0,
});

const [inputs, labels] = parseCSV("...\\dataset.csv", ["input1", "input2"], ["output"]);

optimiser.optimize(neuralNetwork, inputs, labels);
const checkpoint = neuralNetwork.export();
const fileName = `checkpoint-${new Date().getTime()}.model.json`
fs.writeFileSync(fileName, JSON.stringify(checkpoint, null, 4), { encoding: "utf-8" });
```
### Prediction Example
```javascript
const { Network, parseCSV } = require("deepnode");
const fs = require("node:fs");

const neuralNetwork = new Network();

const [inputs, labels] = parseCSV("...\\dataset.csv", ["input1", "input2"], ["output"]);

const fileName = "checkpoint.model.json"
const model = JSON.parse(fs.readFileSync(fileName, { encoding: "utf-8" }));
neuralNetwork.import(model);
console.log(`Model loaded: ${fileName}`)

for (let i = 0; i < inputs.length; i++) {
  const networkOutput = neuralNetwork.predict(inputs[i]);
  const predictedValue = networkOutput.toArray()
  console.log(`Input Value: ${inputs[i].toArray()}. Label Value: ${labels[i].toArray()}. Network Prediction: ${predictedValue}`);
}

const score = neuralNetwork.evaluate(inputs, labels);
console.log(`Score: ${score}`);
```
The library has IntelliSense support for TypeScript or a compatible IDE