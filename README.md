# Reinforcement Learning Snake Game from Scratch in C accelerated by CUDA

## Demonstration

## Introduction

## Game Modes
- **Normal Mode:** Allows User to Play the Game Normally
- **Training Mode:** Allows user to see the AI Agent being trained in realtime.
- **AI Mode:** Allows user to select a pretrained agent and see it play the game.

## Game Class
```mermaid
classDiagram
    class Game {
        int** Board
        int* SnakeX
        int* SnakeY
        int FoodX
        int FoodY
        bool State
        void Initialize()
        int UserInput()
        int AgentInput()
        void Logic()
        void Render()
        float* GenerateFeatures()
        float* GenerateTargets()
        void ShortTermTraining()
        void LongTermTraining()
    }
```

## Layer Class
```mermaid
classDiagram
    class Layer {
        int InputSize
        int OutputSize
        float** Weights
        float* Biases
        float (*ActivationFunction)(float)
        float* ForwardPass(float* input)
        void BackwardPass(float* input, float* gradients)
    }
```

## Neural Network Class
```mermaid
classDiagram
    class NeuralNetwork {
        Layer* Layers
        float (*LossFunction)(float predicted, float actual)
        float LearningRate
        void AddLayer(Layer layer)
        void TrainCPU(float* inputs, float* targets)
        void TrainGPU(float* inputs, float* targets, int num_samples)
        float* Predict(float* input)
        void Save(const char* filename)
        void Load(const char* filename, int* numWeights, int* numBiases)
    }
```

## Deep Reinforcement Learning
Bellman Equations

$$Q_{\text{new}}(s, a) = Q(s, a) + \alpha \left[R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

- $Q_{\text{new}}(s, a)$ is the new $Q$ value for a given state-action pair.
- $Q(s, a)$ is the current $Q$ value given the current state-action pair.
- $\alpha$ is the learning rate.
- $R(s, a)$ is the reward received after taking action $a$ in state $s$.
- $\gamma$ is the discount factor.
- $\max_{a'} Q(s', a')$ is the maximum expected future reward observed at the new state $s'$, across all possible actions $a'$.

```mermaid
graph LR
    subgraph Input Layer
        I1(Input 1)
        I2(Input 2)
        I3(Input 3)
    end

    subgraph Hidden Layer
        H1(Neuron 1)
        H2(Neuron 2)
        H3(Neuron 3)
    end

    subgraph Output Layer
        O1(Output)
    end

    I1 -->|Weight I1H1| H1
    I1 -->|Weight I1H2| H2
    I1 -->|Weight I1H3| H3

    I2 -->|Weight I2H1| H1
    I2 -->|Weight I2H2| H2
    I2 -->|Weight I2H3| H3

    I3 -->|Weight I3H1| H1
    I3 -->|Weight I3H2| H2
    I3 -->|Weight I3H3| H3

    H1 -->|Weight H1O1| O1
    H2 -->|Weight H2O1| O1
    H3 -->|Weight H3O1| O1
```

## Hardware Used
- Saturn Cloud... Google Colab, etc???

## References
- Patric Loeber's PyGame inspiration
- CUDA course by...
