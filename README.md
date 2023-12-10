# Reinforcement Learning Snake Game from Scratch in C accelerated by CUDA

## Demonstration

## Introduction

## Game Modes
- **Normal Mode:** Allows User to Play the Game Normally
- **Training Mode:** Allows user to see the AI Agent being trained in realtime.
- **AI Mode:** Allows user to select a pretrained agent and see it play the game.

## Game Class
- Attributes:
    - `int** Board`: 2D array to track of the snake, walls, and food.
    - `int* SnakeX`: 1D aray to track of the x-positions of the snake.
    - `int* SnakeY`: 1D array to track of the y-positions of the snake.
    - `int FoodX`: Integer to track of the x-position of the food.
    - `int FoodY`: Integer to track of the y-position of the food.
    - `bool State`: Boolean to track whether the last move ended the game or not.
- Methods:
    - `void Initialize()`: Initializes Game State.
    - `int UserInput()`: Gets input from user.
    - `int AgentInput()`: Gets input from agent.
    - `void Logic()`: Executes Game Logic.
    - `void Render()`: Renders the Board and Score.
    - `float* GenerateFeatures()`: Generates features for training and inference.
    - `float* GenerateTargets()`: Generates targets for training.
    - `void ShortTermTraining()`: Trains the Agent every single move, do not uses GPU acceleration.
    - `void LongTermTraining()`: Trains the Agent every specified number of moves, uses GPU acceleration.
<img src="mermaid_outputs/game_class.png" alt="game_class" width="200">

## Layer Class
- Attributes:
    - `int InputSize`: The number fo inputs in the layer.
    - `int OutputSize`: The number of outputs from the layer.
    - `float** Weights`: 2D array representing the layer's weights (one weight for every combination of an input and output in the layer).
    - `float* Biases`: 1D array representing the layer's biases (one bias for every output in the layer.
    - `float (*ActivationFunction)(float)`: Function pointer to the activation function.
- Methods:
    - `float* ForwardPass(float* input)`: Performs the forward pass of the layer.
    - `void BackwardPass(float* input, float* gradients)`: Performs the backward pass of the layer.
<img src="mermaid_outputs/layer_class.png" alt="layer_class" width="370">

## Neural Network Class
- Attributes:
    - `Layer* Layers`: An array of Layers that make up the neural network.
    - `float (*LossFunction)(float, float)`: A function pointer to the loss function.
    - `float LearningRate`: The learning rate used in training.
- Methods:
    - `void AddLayer(Layer layer)`: Adds a layer to the neural network.
    - `void TrainCPU(float* inputs, float*targets)`: Trains the Neural Network using the CPU (works for one example).
    - `void TrainGPU(float* inputs, float* targets, int num_samples)`: Trains the neural network using the GPU (works for multiple examples). 
    - float* Predict(float* input): Makes prediction gieen a set of inputs.
    - `void Save(const char* filename)`: Saves weights and biases of the neural network to a csv file.
    - `void Load(const char* filename, int* numWeights, int* numBiases))`: Loads the weights and biases from a csv file given a specification of weights and biases for each layer.
<img src="mermaid_outputs/neural_network_class.png" alt="neural_network_class" width="450">



## Deep Reinforcement Learning
Bellman Equations

$$Q_{\text{new}}(s, a) = Q(s,a) + \alpha [R(s, a) \gamma {Q'}_{\text{max}}(s', a') - Q(s,a)]]$$

where $Q_{new}(s,a)$ is the new $Q$ value for a given state and action.

## Hardware Used
- Saturn Cloud... Google Colab, etc???

## References
- Patric Loeber's PyGame inspiration
- CUDA course by...
