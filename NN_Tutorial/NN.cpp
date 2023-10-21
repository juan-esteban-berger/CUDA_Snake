#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Define the ReLU activation and its derivative
double relu(double x) {
    return std::max(0.0, x);
}

double relu_derivative(double x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

class NeuralNetwork {
private:
    int input_nodes;
    int hidden_nodes1;
    int hidden_nodes2;
    int hidden_nodes3;
    int output_nodes;

    std::vector<std::vector<double>> weights_ih1;
    std::vector<double> bias_h1;

    std::vector<std::vector<double>> weights_h1h2;
    std::vector<double> bias_h2;

    std::vector<std::vector<double>> weights_h2h3;
    std::vector<double> bias_h3;

    std::vector<std::vector<double>> weights_ho;
    std::vector<double> bias_o;

public:
    NeuralNetwork(int input, int hidden1, int hidden2, int hidden3, int output)
    : input_nodes(input), hidden_nodes1(hidden1), hidden_nodes2(hidden2), hidden_nodes3(hidden3), output_nodes(output) {
        srand(time(0));
        
        // Initialize weights and biases with random values for first hidden layer
        weights_ih1 = std::vector<std::vector<double>>(hidden_nodes1, std::vector<double>(input_nodes));
        for (int i = 0; i < hidden_nodes1; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                weights_ih1[i][j] = (rand() % 2000 - 1000) / 1000.0;
            }
        }
        bias_h1 = std::vector<double>(hidden_nodes1, 0);

        // For second hidden layer
        weights_h1h2 = std::vector<std::vector<double>>(hidden_nodes2, std::vector<double>(hidden_nodes1));
        for (int i = 0; i < hidden_nodes2; ++i) {
            for (int j = 0; j < hidden_nodes1; ++j) {
                weights_h1h2[i][j] = (rand() % 2000 - 1000) / 1000.0;
            }
        }
        bias_h2 = std::vector<double>(hidden_nodes2, 0);

        // For third hidden layer
        weights_h2h3 = std::vector<std::vector<double>>(hidden_nodes3, std::vector<double>(hidden_nodes2));
        for (int i = 0; i < hidden_nodes3; ++i) {
            for (int j = 0; j < hidden_nodes2; ++j) {
                weights_h2h3[i][j] = (rand() % 2000 - 1000) / 1000.0;
            }
        }
        bias_h3 = std::vector<double>(hidden_nodes3, 0);

        // For output layer
        weights_ho = std::vector<std::vector<double>>(output_nodes, std::vector<double>(hidden_nodes3));
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes3; ++j) {
                weights_ho[i][j] = (rand() % 2000 - 1000) / 1000.0;
            }
        }
        bias_o = std::vector<double>(output_nodes, 0);
    }

    // Implement backpropagation to train the network
    void train(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        // Feedforward
        std::vector<double> hidden1(hidden_nodes1);
        std::vector<double> hidden2(hidden_nodes2);
        std::vector<double> hidden3(hidden_nodes3);
        std::vector<double> output(output_nodes);

        // Calculate hidden1 outputs
        for (int i = 0; i < hidden_nodes1; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                hidden1[i] += input[j] * weights_ih1[i][j];
            }
            hidden1[i] += bias_h1[i];
            hidden1[i] = relu(hidden1[i]);
        }

        // Calculate hidden2 outputs
        for (int i = 0; i < hidden_nodes2; ++i) {
            for (int j = 0; j < hidden_nodes1; ++j) {
                hidden2[i] += hidden1[j] * weights_h1h2[i][j];
            }
            hidden2[i] += bias_h2[i];
            hidden2[i] = relu(hidden2[i]);
        }

        // Calculate hidden3 outputs
        for (int i = 0; i < hidden_nodes3; ++i) {
            for (int j = 0; j < hidden_nodes2; ++j) {
                hidden3[i] += hidden2[j] * weights_h2h3[i][j];
            }
            hidden3[i] += bias_h3[i];
            hidden3[i] = relu(hidden3[i]);
        }

        // Calculate final outputs
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes3; ++j) {
                output[i] += hidden3[j] * weights_ho[i][j];
            }
            output[i] += bias_o[i];
        }

        // Compute output errors (target - output)
        std::vector<double> output_errors(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            output_errors[i] = target[i] - output[i];
        }

        // Backpropagate errors to hidden layers
        // Calculate hidden3 errors
        std::vector<double> hidden3_errors(hidden_nodes3, 0.0);
        for (int i = 0; i < hidden_nodes3; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                hidden3_errors[i] += output_errors[j] * weights_ho[j][i];
            }
        }

        // Calculate hidden2 errors
        std::vector<double> hidden2_errors(hidden_nodes2, 0.0);
        for (int i = 0; i < hidden_nodes2; ++i) {
            for (int j = 0; j < hidden_nodes3; ++j) {
                hidden2_errors[i] += hidden3_errors[j] * weights_h2h3[j][i];
            }
        }

        // Calculate hidden1 errors
        std::vector<double> hidden1_errors(hidden_nodes1, 0.0);
        for (int i = 0; i < hidden_nodes1; ++i) {
            for (int j = 0; j < hidden_nodes2; ++j) {
                hidden1_errors[i] += hidden2_errors[j] * weights_h1h2[j][i];
            }
        }

        // Update weights & biases
        // Update output weights and biases
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes3; ++j) {
                weights_ho[i][j] += learning_rate * output_errors[i] * hidden3[j];
            }
            bias_o[i] += learning_rate * output_errors[i];
        }

        // Update hidden3 weights and biases
        for (int i = 0; i < hidden_nodes3; ++i) {
            for (int j = 0; j < hidden_nodes2; ++j) {
                weights_h2h3[i][j] += learning_rate * hidden3_errors[i] * relu_derivative(hidden3[i]) * hidden2[j];
            }
            bias_h3[i] += learning_rate * hidden3_errors[i] * relu_derivative(hidden3[i]);
        }

        // Update hidden2 weights and biases
        for (int i = 0; i < hidden_nodes2; ++i) {
            for (int j = 0; j < hidden_nodes1; ++j) {
                weights_h1h2[i][j] += learning_rate * hidden2_errors[i] * relu_derivative(hidden2[i]) * hidden1[j];
            }
            bias_h2[i] += learning_rate * hidden2_errors[i] * relu_derivative(hidden2[i]);
        }

        // Update hidden1 weights and biases
        for (int i = 0; i < hidden_nodes1; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                weights_ih1[i][j] += learning_rate * hidden1_errors[i] * relu_derivative(hidden1[i]) * input[j];
            }
            bias_h1[i] += learning_rate * hidden1_errors[i] * relu_derivative(hidden1[i]);
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        // Calculate outputs for first hidden layer
        std::vector<double> hidden1(hidden_nodes1, 0);
        for (int i = 0; i < hidden_nodes1; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                hidden1[i] += input[j] * weights_ih1[i][j];
            }
            hidden1[i] += bias_h1[i];
            hidden1[i] = relu(hidden1[i]);
        }

        // Calculate outputs for second hidden layer
        std::vector<double> hidden2(hidden_nodes2, 0);
        for (int i = 0; i < hidden_nodes2; ++i) {
            for (int j = 0; j < hidden_nodes1; ++j) {
                hidden2[i] += hidden1[j] * weights_h1h2[i][j];
            }
            hidden2[i] += bias_h2[i];
            hidden2[i] = relu(hidden2[i]);
        }

        // Calculate outputs for third hidden layer
        std::vector<double> hidden3(hidden_nodes3, 0);
        for (int i = 0; i < hidden_nodes3; ++i) {
            for (int j = 0; j < hidden_nodes2; ++j) {
                hidden3[i] += hidden2[j] * weights_h2h3[i][j];
            }
            hidden3[i] += bias_h3[i];
            hidden3[i] = relu(hidden3[i]);
        }

        // Calculate final outputs
        std::vector<double> output(output_nodes, 0);
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes3; ++j) {
                output[i] += hidden3[j] * weights_ho[i][j];
            }
            output[i] += bias_o[i];
            // No activation function for output layer as we want raw Q-values
        }

        return output;
    }
};

double mse_loss(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        sum += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
    return sum / predicted.size();
}

// Generate data based on the known function
std::vector<std::pair<std::vector<double>, std::vector<double>>> generate_data(int num_points, int input_size) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data;

    for (int i = 0; i < num_points; ++i) {
        std::vector<double> input(input_size, 0);
        double board_sum = 0.0;
        
        // Set one-hot encoded positions randomly
        for (int j = 0; j < input_size - 1; ++j) {
            input[j] = (rand() % 2); // either 0 or 1
            board_sum += input[j];
        }
        
        // The last input is the action, set randomly between 0 and 1
        input[input_size - 1] = (rand() % 1000) / 1000.0;

        // Compute Q-values based on board_sum
        std::vector<double> q_values = {2 * board_sum, 3 * board_sum, -1 * board_sum, board_sum + 5};

        data.push_back({input, q_values});
    }

    return data;
}

int main() {
    // Create a neural network
    NeuralNetwork nn(1601, 512, 256, 128, 4);

    // Generate data points
    int num_data_points = 1000;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data = generate_data(num_data_points, 1601);
    
    std::cout << "Generated data:\n";
    for (const auto& point : data) {
        std::cout << "Input: [";
        for (const auto& val : point.first) {
            std::cout << val << ",";
        }
        std::cout << "], Q-values: [";
        for (const auto& val : point.second) {
            std::cout << val << ",";
        }
        std::cout << "]\n";
    }

    // Train the network using the generated data and backpropagation
    int epochs = 1000; // Example number of epochs
    double learning_rate = 0.01; // Example learning rate

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& point : data) {
            std::vector<double> input = point.first;
            std::vector<double> target = point.second;

            std::vector<double> predicted = nn.predict(input);
            total_loss += mse_loss(predicted, target);

            // Train the network using backpropagation
            nn.train(input, target, learning_rate);
        }

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_data_points << "\n";
    }

    return 0;
}
