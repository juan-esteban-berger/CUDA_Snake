#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <random>
#include <ctime>
#include <ncurses.h>
#include <vector>
#include <cstring>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <unistd.h>


////////////////////////////////////////////////////////////////////////
// Variable Definitions
bool gameOver;

// Enum for Direction
enum Direction { UP, DOWN, LEFT, RIGHT };

// Ints for generating the snake's position
int x, y;

// Ints for generating the food's position
int foodX, foodY;

// Int for storing the score
int score;

// Generate Board
const int bd_size = 20;
int board[bd_size][bd_size] = {0};

// Create variable for number of actions before retraining
const int round_actions = 1000;

// Deque to store the Snake
std::deque<std::pair<int, int>> snake;

// Direction of the Snake
Direction dir;

// Int to store the index of the action taken
int actionIndex;

// Global variable to ensure srand is called only once
bool isRandomSeeded = false;

// Global variable for checking if the snake has eaten food
bool hasEatenFood;

// Global variables for features, rewards, and q-values
std::vector<double> features;
std::vector<double> current_features;
std::vector<double> next_features;
double reward = 0.0;
std::vector<double> current_q_values;
std::vector<double> next_q_values;

////////////////////////////////////////////////////////////////////////
// Preliminary Functions
// Setup Function
void setup(){
    if (!isRandomSeeded) {
        srand(time(nullptr));
        isRandomSeeded = true;
    }
    gameOver = false;
    score = 0;
    x = rand() % (bd_size - 2) + 1;
    y = rand() % (bd_size - 2) + 1;
    int randomNum = rand() % 4;
    dir = static_cast<Direction>(randomNum);
    snake.clear();
    snake.push_back({x, y});
}

// Print board during game
void print_board() {
    clear();  // Clear the screen

    for (int i = 0; i < bd_size; i++) {
        for (int j = 0; j < bd_size; j++) {
            if (board[i][j] == 0) {
                mvprintw(i, j*2, "  ");
            } else if (board[i][j] == 1) {
                mvprintw(i, j*2, "#");
            } else if (board[i][j] == 2) {
                mvprintw(i, j*2, "O");
            } else if (board[i][j] == 3) {
                mvprintw(i, j*2, "F");
            } else {
                mvprintw(i, j*2, "E");
            }
        }
    }

    // Print "Score"
    mvprintw(bd_size, 0, "Score: %d", score);

    // Print "Reward"
    mvprintw(bd_size + 1, 0, "Reward: %1.2f", reward);  // Printing the reward

    // Printing features at the end, 20 per line
    int feature_line = bd_size + 2; // Start printing features at this line
    mvprintw(feature_line, 0, "Features:"); // Print "Features:"
    feature_line++; // Move to the next line to start printing features

    int offset = 0; // Starting offset for features printing (no initial offset for the first line of features)
    for (int i = 0; i < features.size(); i++) {
        if (i > 0 && i % 20 == 0) {
            feature_line++; // Move to the next line after printing 20 features
            offset = 0; // Reset offset for the new line
        }
        mvprintw(feature_line, offset, "%d ", static_cast<int>(features[i]));
        offset += 2 + strlen(std::to_string(static_cast<int>(features[i])).c_str()) - 1; // Update offset for the next feature
    }

    // refresh();  // Refresh the screen
}

// Print Board Function outside of ncurses mode once game has ended
void print_board_ended(){
  for (int i=0; i < bd_size; i++){
    for (int j=0; j < bd_size; j++){
      if (board[i][j] == 0){
        std::cout << "  ";
      } else if (board[i][j] == 1){
        std::cout << "# ";
      } else if (board[i][j] == 2){
        std::cout << "O ";
      } else if (board[i][j] == 3){
        std::cout << "F ";
      } else {
        std::cout << "E ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << "Score: " << score << std::endl;

  // Print the reward using the global variable
  std::cout << "Reward: " << reward << std::endl;

  // Print features 20 at a time
  std::cout << "Features: " << std::endl; // Print "Features: " followed by a newline
  for (int i = 0; i < features.size(); i++) {
      std::cout << static_cast<int>(features[i]); // Print feature as integer
      if ((i + 1) % 20 == 0) {
          std::cout << std::endl; // After every 20 features, start a new line
      } else {
          std::cout << " "; // Separate features with a space
      }
  }
  if (features.size() % 20 != 0) { // Add a newline if the last line wasn't complete
      std::cout << std::endl;
  }


}

// Set Wall Function
void set_walls(){
  int count = 0;
  for (int i=0; i < bd_size; i++){
    for (int j=0; j < bd_size; j++){
      if (i == 0 || i == bd_size-1 || j == 0 || j == bd_size-1){
        board[i][j] = 1;
      }
    }
  }
}

// Update Snake Function
void update_snake(){
    // Remove the snake from the board
    for (int i = 0; i < bd_size; i++) {
        for (int j = 0; j < bd_size; j++) {
            if (board[i][j] == 2) {
                board[i][j] = 0;
            }
        }
    }
  // Update the board with snake's current position
  for(std::deque<std::pair<int, int>>::iterator it = snake.begin(); it != snake.end(); ++it) {
      int x = it->first;
      int y = it->second;
      board[y][x] = 2;
  }
}

// Generate Food Function
void generate_food() {
    // Ensure srand is called only once
    if (!isRandomSeeded) {
        srand(time(nullptr));
        isRandomSeeded = true;
    }

    bool foodPlaced = false;

    while (!foodPlaced) {
        // Generate random position for food
        foodX = rand() % (bd_size - 2) + 1;
        foodY = rand() % (bd_size - 2) + 1;

        // Assume this is a valid position for now
        bool isValidPosition = true;

        // Check if the generated position is on the snake's body
        for (const auto& part : snake) {
            if (part.first == foodX && part.second == foodY) {
                isValidPosition = false;
                break;
            }
        }

        // If the position is valid and not on the walls, place the food there
        if (isValidPosition && board[foodY][foodX] == 0) {
            board[foodY][foodX] = 3;  // Set the food position on the board
            foodPlaced = true;
        }
    }
}

// Input Function
void Input(){
    int ch;
    ch = getch();
    switch(ch){
        case 'w':
            if(dir != DOWN || snake.size() == 1)  // If moving UP, prevent moving DOWN if there's a snake
                dir = UP;
            break;
        case 'a':
            if(dir != RIGHT || snake.size() == 1)  // If moving LEFT, prevent moving RIGHT if there's a snake
                dir = LEFT;
            break;
        case 's':
            if(dir != UP || snake.size() == 1)  // If moving DOWN, prevent moving UP if there's a snake
                dir = DOWN;
            break;
        case 'd':
            if(dir != LEFT || snake.size() == 1)  // If moving RIGHT, prevent moving LEFT if there's a snake
                dir = RIGHT;
            break;
        default:
            break;
    }
    refresh();
}

int AI_Input(const std::vector<double>& output) {
    int maxIndex = 0;
    double maxValue = output[0];
    for (int i = 1; i < output.size(); ++i) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            maxIndex = i;
        }
    }

    Direction newDir;
    switch(maxIndex) {
        case 0: newDir = UP; break;
        case 1: newDir = RIGHT; break;
        case 2: newDir = DOWN; break;
        case 3: newDir = LEFT; break;
    }

    // Allow moving in opposite direction if snake's size is 1
    if (newDir == UP && (dir != DOWN || snake.size() == 1)) dir = UP;
    else if (newDir == RIGHT && (dir != LEFT || snake.size() == 1)) dir = RIGHT;
    else if (newDir == DOWN && (dir != UP || snake.size() == 1)) dir = DOWN;
    else if (newDir == LEFT && (dir != RIGHT || snake.size() == 1)) dir = LEFT;

    refresh();

    return maxIndex; // Return the index of the chosen action
}

// Logic Function
bool Logic(){
    int prevX = x;  // Store the previous x before updating
    int prevY = y;

    // Potential new positions
    int potentialX = x;
    int potentialY = y;

    // Adjusting direction of the snake's head based on input
    switch(dir){
        case UP:
            potentialY--;
            break;
        case LEFT:
            potentialX--;
            break;
        case DOWN:
            potentialY++;
            break;
        case RIGHT:
            potentialX++;
            break;
        default:
            break;
    }

    // Check if the potential position is a wall (value 1 in board)
    if(board[potentialY][potentialX] == 1){
        gameOver = true;
    }

    // Check if the snake would collide with itself
    // We'll start from the second element to skip the head during the check
    for (auto it = snake.begin(); it != snake.end(); ++it) {
        if (it->first == potentialX && it->second == potentialY) {
            gameOver = true;
            break;
        }
    }

    // Bool to check if the snake has eaten food
    bool hasEatenFood = false;

    // Only update the positions if the game isn't over
    if (!gameOver) {
        x = potentialX;
        y = potentialY;
        snake.push_front({x, y});   // Add new head position to the front of the snake
        snake.pop_back();           // Remove last segment of the snake

        // Check if the snake has eaten the food
        if (x == foodX && y == foodY) {
            score++;
            generate_food(); // Corrected from GenerateFood
            snake.push_back({prevX, prevY});  // Add a new segment to the snake
            // Set hasEatenFood to true
            hasEatenFood = true;
        }
    }

    return hasEatenFood;

}

// Function to generate a vector with the features for deep reinforcement learning
std::vector<double> get_features() {
    // The total number of features is bd_size * bd_size (board state) + 11 (initial features)
    std::vector<double> features(bd_size * bd_size + 15, 0.0);

    // Flatten the board into a 1D vector and place it at the beginning of features.
    for (int i = 0; i < bd_size; i++) {
        for (int j = 0; j < bd_size; j++) {
            double cellValue = static_cast<double>(board[i][j]);
            // Set the feature to -1 if the cell contains food, otherwise keep the value
            // Then, add 1 to all values to avoid negative inputs for ReLU
            features[i * bd_size + j] = (cellValue == 3.0) ? -1.0 : cellValue;
            features[i * bd_size + j] += 1.0;
        }
    }

    // The number of features representing the board state
    int board_state_size = bd_size * bd_size;

    // Get the head of the snake
    int headX = snake.front().first;
    int headY = snake.front().second;

    // Food is above the snake
    if (foodY < headY) {
        features[board_state_size] = 1.0;
    } else {
        features[board_state_size] = 0.0;
    }

    // Food is below the snake
    if (foodY > headY) {
        features[board_state_size + 1] = 1.0;
    } else {
        features[board_state_size + 1] = 0.0;
    }

    // Food is to the left of the snake
    if (foodX < headX) {
        features[board_state_size + 2] = 1.0;
    } else {
        features[board_state_size + 2] = 0.0;
    }

    // Food is to the right of the snake
    if (foodX > headX) {
        features[board_state_size + 3] = 1.0;
    } else {
        features[board_state_size + 3] = 0.0;
    }

    // Distance from top wall
    features[board_state_size + 4] = headY;

    // Distance from bottom wall
    features[board_state_size + 5] = bd_size - 1 - headY;

    // Distance from left wall
    features[board_state_size + 6] = headX;

    // Distance from right wall
    features[board_state_size + 7] = bd_size - 1 - headX;

    // X distance from food
    features[board_state_size + 8] = std::abs(foodX - headX);

    // Y distance from food
    features[board_state_size + 9] = std::abs(foodY - headY);

    // Checking if the current direction of the snake is up
    if (dir == UP) {
        features[board_state_size + 10] = 1.0;
    } else {
        features[board_state_size + 10] = 0.0;
    }

    // Checking if the current direction of the snake is right
    if (dir == RIGHT) {
        features[board_state_size + 11] = 1.0;
    } else {
        features[board_state_size + 11] = 0.0;
    }

    // Checking if the current direction of the snake is down
    if (dir == DOWN) {
        features[board_state_size + 12] = 1.0;
    } else {
        features[board_state_size + 12] = 0.0;
    }

    // Checking if the current direction of the snake is left
    if (dir == LEFT) {
        features[board_state_size + 13] = 1.0;
    } else {
        features[board_state_size + 13] = 0.0;
    }

    // Add score as the last feature
    features[board_state_size + 14] = static_cast<double>(score);

    return features;
}

// Function to generate rewards based on the game state
double calculate_reward(bool gameOver, bool hasEatenFood) {
    // Define the rewards/penalties for various events
    const double eatingReward = 10.0;
    const double deathPenalty = -100.0;

    // If the game is over, return a large negative penalty
    if (gameOver) {
        return deathPenalty;
    }

    // If the snake eats food, return a positive reward
    if (hasEatenFood) {
        return eatingReward;
    }

    // If none of the above, return a small negative reward to encourage faster learning
    return -0.1;
}

// Function to update the Q-value for Deep Q-Learning
double update_q_value(double currentQValue, double reward, double maxNextQValue, double alpha, double gamma) {
    // Q-learning formula: Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
    return currentQValue + alpha * (reward + gamma * maxNextQValue - currentQValue);
}

// End Game Function
void EndGame() {
    endwin();// End ncurses mode
}

////////////////////////////////////////////////////////////////////////
// Machine Learning
// ReLU Function
double relu(double x){
  return std::max(0.0, x);
}

// Derivative of ReLU Function
double relu_derivative(double x){
  if (x > 0){
    return 1.0;
  }
  else{
    return 0.0;
  }
}

class Layer {
  public:
    // Initialize weights, biases, and other parameters
    std::vector<std::vector<double>> weights; // 2D vector to store the weights
    std::vector<double> biases; // 1D vector to store the biases
    std::vector<double> outputs; // 1D vector to store the outputs
    int inputSize, outputSize; // Integers to store the input and output sizes
    std::function<double(double)> activationFunction; // Activation function
    std::vector<std::vector<double>> dweights; // 2D vector to store the derivatives of
                                               // of the weights with respect to the loss
    std::vector<double> dbiases; // 1D vector to store the derivatives of the biases
                                 // with respect to the loss
    std::vector<double> dinputs; // 1D vector to store the derivatives of the inputs
                                 // with respect to the loss
    // Constructor
    Layer(
        int inputSize,
        int outputSize,
        std::function<double(double)> activationFunction
        ): inputSize(inputSize), 
           outputSize(outputSize),
           activationFunction(activationFunction) {

      // Resize the weights matrix to have 'outputSize' rows and 'inputSize' columns
      weights.resize(outputSize, std::vector<double>(inputSize));
      // Resize the biases vector to have 'outputSize' elements
      biases.resize(outputSize);
      // Ensure srand is called only once
      if (!isRandomSeeded) {
        srand(time(nullptr));
        isRandomSeeded = true;
      }
      // Initialize the weights and biases with random values
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          // Random number between -0.5 and 0.5
          weights[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5;
        }
        // Random number between -0.5 and 0.5
        biases[i] = static_cast<double>(rand()) / RAND_MAX - 0.5;
      }
    }

    // Forward Pass Function
    std::vector<double> forward(const std::vector<double>& input) {
      // Resize the outputs vector to be the same as outputSize
      outputs.resize(outputSize);
      // Each value in the outputs vector should be the sum of
      // the each input and its corresponding weight, plus the bias
      for (int i = 0; i < outputSize; ++i) {
        // Initialize the output to zero
        outputs[i] = 0;
        // Iterate throught the inputs
        for (int j = 0; j < inputSize; ++j) {
          // Sum the weighted input
          outputs[i] += input[j] * weights[i][j];
        }
        // Add the bias
        outputs[i] += biases[i];
        // Pass through the activation function
        outputs[i] = activationFunction(outputs[i]);
      }
      return outputs;
    }

    // Backward Pass Function
    std::vector<double> backward(
        const std::vector<double>& doutputs,
        const std::vector<double>& inputs) {

      // Resize dweights to match the dimensions of the
      // weights matrix in the current layer
      dweights.resize(outputSize, std::vector<double>(inputSize));
      // Reisze dbiases to match the dimensions of the
      // biases vector in the current layer
      dbiases.resize(outputSize);
      // Resize dinputs to match the dimensions of the
      // inputs vector in the current layer
      // Initialize all the values to 0.0
      dinputs.resize(inputSize, 0.0);

      // Iterate once for each output
      for (int i = 0; i <outputSize; i++){
        // The derivative of the loss with respect to the bias
        // is equal to the derivative of the output with respect
        dbiases[i] = doutputs[i];

        // Iterate once for each input
        for (int j = 0; j < inputSize; j++){
          // Use the chain rule to calculate the derivative of the loss
          // with respect to the weight
          dweights[i][j] = inputs[j] * doutputs[i];
          // Use the chain rule to calculate the derivative of the loss
          // with respect to the input
          dinputs[j] += weights[i][j] * doutputs[i];
        }
      }

      return dinputs;

    }

};

// Neural Network Class
class NeuralNetwork{
  public:
    // Vector to store the layers of the neural network.
    std::vector<Layer> layers;

    // Function to add a layer to the Neural Network
    void addLayer(
        int inputSize, 
        int outputSize,
        std::function<double(double)> activationFunction) {
        // Use emplace_back instead of push_back in order to constructs an object in-place at the end of the container, rather than adding a copy of an existing object to the end of a container.
        layers.emplace_back(inputSize, outputSize, activationFunction);
    }

    // Function to train the Neural Network
    void train(
        std::vector<std::vector<double>> inputs,
        std::vector<std::vector<double>> targets,
        double learningRate,
        int epochs){
          // Iterate once through each epoch
          for (int epoch = 0; epoch < epochs; epoch++) {
            // Double to store the total error for this epoch
            double totalError = 0;

            // Loop through all the training examples
            for (size_t i = 0; i < inputs.size(); i++){
             
              // Vector to store the current input
              std::vector<double> input = inputs[i];
              // Vector to store the original inputs for each layer
              // (for use in the backward pass)
              std::vector<std::vector<double>> layerInputs;

              // Loop throuch each layer in the Network
              // performing a forward pass in each iteration
              for (Layer& layer : layers){
                // Store the original input for use in the backward pass
                layerInputs.push_back(input);
                // Perform a forward pass
                input = layer.forward(input);
              }

              // Create a new vector to store the output
              // which is currently stored in input
              std::vector<double> output = input;
              // Create a new vector to store the derivative of the loss
              // with respect to the output of the last layer
              std::vector<double> dLoss_dOutput(output.size());
              // Double to store the loss
              double loss = 0.0;

              // Iterate through each element in the output
              for (size_t j = 0; j < output.size(); j++){
                // Calculate the error
                double error = output[j] - targets[i][j];
                // Add the error to the loss
                loss += error * error;
                // Calculate the derivative of the loss with respect to the output
                // This is the derivative of the Mean Squared Error
                // dLoss_dOutput[j] = 2 * output[j] - targets[i][j] / output.size();
                dLoss_dOutput[j] = 2 * error / output.size();
              }

              // Add the loss to the total error
              totalError += loss / output.size();

              // Vector to store the derivative of the output
              // with respect to the input of the last layer
              std::vector<double> dOutput = dLoss_dOutput;

              // Loop through each layer in the network in reverse
              for (int j = layers.size() - 1; j >= 0; --j){
                // Perform a backward pass
                dOutput = layers[j].backward(dOutput, layerInputs[j]);
              }

              // Loop through each layer
              for (Layer& layer : layers){
                // Loop through each weight in the layer
                for (size_t k = 0; k < layer.weights.size(); ++k){
                  for (size_t l = 0; l < layer.weights[k].size(); ++l){
                    // Update the weight
                    layer.weights[k][l] -= learningRate * layer.dweights[k][l];
                  }
                  // Update the bias
                  layer.biases[k] -= learningRate * layer.dbiases[k];
                }
              }

            }

            // Print the average error for this dataset
            std::cout << "Epoch " << epoch + 1 << " / " << epochs << ", Error: " << totalError / inputs.size() << std::endl;
          }
    }

    // Function to use the Neural Networks to make predictions
    std::vector<double> predict(const std::vector<double>& input) {
        // Initialize a vector to store the current output that starts
        // by being equal to the input
        std::vector<double> currentOutput = input;
        // Iterate through each layer in the neural network
        for (Layer& layer : layers) {
            // Update the current output by calling the forward function
            currentOutput = layer.forward(currentOutput);
        }
        return currentOutput;
    }
};

////////////////////////////////////////////////////////////////////////
// Main Function
int main(){

  // Declare variables
  int choice;
  int sleep_time;

  // Set sleep time
  sleep_time = 160000;

  // Print menu
  system("clear");
  std::cout << "Please select a mode: " << std::endl;
  std::cout << "1. Normal Mode" << std::endl;
  std::cout << "2. AI Mode" << std::endl;
  std::cout << "3. Train AI Model" << std::endl;
  std::cout << "4. Exit" << std::endl;
  std::cout << "Enter your choice: ";
  std::cin >> choice;

  // Normal Mode
  if (choice == 1){
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE); // Comment this out for debugging
    setup();
    set_walls();

    generate_food();

    while(!gameOver){
      update_snake();
      Input();
      hasEatenFood = Logic();

      // Get Features
      features = get_features();
      // Get Rewards
      // rewards = get_rewards(gameOver, hasEatenFood, score);
      reward = calculate_reward(gameOver, hasEatenFood);

      print_board();
      refresh();  // Refresh the screen

      // Sleep 
      usleep(sleep_time);

      
    }
   
    EndGame();
    system("clear");

    // Get Features
    features = get_features();
    // Get Rewards
    // rewards = get_rewards(gameOver, hasEatenFood, score);
    reward = calculate_reward(gameOver, hasEatenFood);

    print_board_ended();

    return 0;
  }

  // AI Mode
  else if (choice == 2) {
      // Model selection and initial setup
      system("clear");
      std::cout << "Please select a model: " << std::endl;
      std::cout << "1. placeholder_model.csv" << std::endl;
      std::cout << "2. placeholder_model.csv" << std::endl;
      std::cout << "3. placeholder_model.csv" << std::endl;
      std::cout << "4. Exit" << std::endl;
      std::cout << "Enter your choice: ";
      std::cin >> choice;

      // Initialize Neural Network
      srand(time(nullptr));
      NeuralNetwork nn;
      int inputSize = bd_size * bd_size + 15;  // Calculate the input size based on board size and additional features
      nn.addLayer(inputSize, 256, relu);       // First layer now takes the correct input size
      nn.addLayer(256, 128, relu);             // Subsequent layers remain unchanged
      nn.addLayer(128, 4, relu);   

      // Initialize game environment
      initscr();
      cbreak();
      keypad(stdscr, TRUE);
      noecho();
      curs_set(0);
      setup();
      set_walls();
      generate_food();

      // Define variables for the AI
      std::vector<double> input(inputSize), output, current_q_values, next_q_values;
      double reward, maxNextQValue;
      double alpha = 0.1, gamma = 0.9; // Hyperparameters

      // Main game loop
      while (!gameOver) {
          // Game logic
          update_snake();

          // Reinforcement Learning Steps
          // Step 1: Get Current Features
          features = get_features();

          // Step 2: Predict Current Q-Values
          current_q_values = nn.predict(features);

          // Step 3: Choose action and Move Snake
          actionIndex = AI_Input(current_q_values);

          hasEatenFood = Logic();

          // Step 4: Get Reward
          reward = calculate_reward(gameOver, hasEatenFood);

          // Step 5: Get Next Features
          next_features = get_features();

          // Step 6: Predict Next Q-Values
          next_q_values = nn.predict(next_features);

          // Step 7: Find Max Q-Value for Next State
          maxNextQValue = *std::max_element(next_q_values.begin(), next_q_values.end());

          // Step 8: Update Q-Value for the action taken
          current_q_values[actionIndex] = update_q_value(current_q_values[actionIndex], reward, maxNextQValue, alpha, gamma);

          // Display Information
          print_board();
          printw("\n");

          // Print current Q-values with corresponding directions
          printw("\nCurrent Q-Values: ");
          printw("UP: %.2f, RIGHT: %.2f, DOWN: %.2f, LEFT: %.2f", 
                 current_q_values[0], current_q_values[1], current_q_values[2], current_q_values[3]);

          // Print next Q-values with corresponding directions
          printw("\nNext Q-Values: ");
          printw("UP: %.2f, RIGHT: %.2f, DOWN: %.2f, LEFT: %.2f", 
                 next_q_values[0], next_q_values[1], next_q_values[2], next_q_values[3]);

          // Print current direction in one-hot encoded format
          printw("\nDirection: [%d %d %d %d]", 
                 dir == UP, dir == RIGHT, dir == DOWN, dir == LEFT);

          // Refresh the screen to update the output
          refresh();

          // Sleep for a set duration
          usleep(sleep_time);
      }

      // Ending the game
      EndGame();
      system("clear");

      // Print final game state
      print_board_ended();

      // Print current Q-values with corresponding directions
      std::cout << "\nCurrent Q-Values: ";
      std::cout << "UP: " << current_q_values[0] << ", "
                << "RIGHT: " << current_q_values[1] << ", "
                << "DOWN: " << current_q_values[2] << ", "
                << "LEFT: " << current_q_values[3] << std::endl;

      // Print next Q-values with corresponding directions
      std::cout << "Next Q-Values: ";
      std::cout << "UP: " << next_q_values[0] << ", "
                << "RIGHT: " << next_q_values[1] << ", "
                << "DOWN: " << next_q_values[2] << ", "
                << "LEFT: " << next_q_values[3] << std::endl;

      // Print current direction in one-hot encoded format
      std::cout << "Direction: [" 
                << (dir == UP) << " " 
                << (dir == RIGHT) << " " 
                << (dir == DOWN) << " " 
                << (dir == LEFT) << "]" << std::endl;

      return 0;
  }

  // Train AI Model
  else if (choice == 3){
      // Model selection and initial setup
      system("clear");
      std::cout << "Please select a model: " << std::endl;
      std::cout << "1. placeholder_model.csv" << std::endl;
      std::cout << "2. placeholder_model.csv" << std::endl;
      std::cout << "3. placeholder_model.csv" << std::endl;
      std::cout << "4. Exit" << std::endl;
      std::cout << "Enter your choice: ";
      std::cin >> choice;

      // Initialize Neural Network
      srand(time(nullptr));
      NeuralNetwork nn;
      int inputSize = bd_size * bd_size + 15;  // Calculate the input size based on board size and additional features
      nn.addLayer(inputSize, 256, relu);       // First layer now takes the correct input size
      nn.addLayer(256, 128, relu);             // Subsequent layers remain unchanged
      nn.addLayer(128, 4, relu);   

      // Initialize game environment
      initscr();
      cbreak();
      keypad(stdscr, TRUE);
      noecho();
      curs_set(0);
      setup();
      set_walls();
      generate_food();

      // Define variables for the AI
      std::vector<double> input(inputSize), output, current_q_values, next_q_values;
      double reward, maxNextQValue;
      double alpha = 0.1, gamma = 0.9; // Hyperparameters

      // Counter to keep track of the number of moves
      int moveCounter = 0;
      // Main game loop
      while (true) {
        // Main game loop
        if (!gameOver) {
            // Game logic
            update_snake();

            // Reinforcement Learning Steps
            // Step 1: Get Current Features
            features = get_features();

            // Step 2: Predict Current Q-Values
            current_q_values = nn.predict(features);

            // Step 3: Choose action and Move Snake
            actionIndex = AI_Input(current_q_values);

            hasEatenFood = Logic();

            // Step 4: Get Reward
            reward = calculate_reward(gameOver, hasEatenFood);

            // Step 5: Get Next Features
            next_features = get_features();

            // Step 6: Predict Next Q-Values
            next_q_values = nn.predict(next_features);

            // Step 7: Find Max Q-Value for Next State
            maxNextQValue = *std::max_element(next_q_values.begin(), next_q_values.end());

            // Step 8: Update Q-Value for the action taken
            current_q_values[actionIndex] = update_q_value(current_q_values[actionIndex], reward, maxNextQValue, alpha, gamma);

            // Display Information
            print_board();
            printw("\n");

            // Print current Q-values with corresponding directions
            printw("\nCurrent Q-Values: ");
            printw("UP: %.2f, RIGHT: %.2f, DOWN: %.2f, LEFT: %.2f", 
                   current_q_values[0], current_q_values[1], current_q_values[2], current_q_values[3]);

            // Print next Q-values with corresponding directions
            printw("\nNext Q-Values: ");
            printw("UP: %.2f, RIGHT: %.2f, DOWN: %.2f, LEFT: %.2f", 
                   next_q_values[0], next_q_values[1], next_q_values[2], next_q_values[3]);

            // Print current direction in one-hot encoded format
            printw("\nDirection: [%d %d %d %d]", 
                   dir == UP, dir == RIGHT, dir == DOWN, dir == LEFT);

            // Refresh the screen to update the output
            refresh();

            // Sleep for a set duration
            usleep(sleep_time);

            // Increment the move counter
            moveCounter++;

            // Check if it's time to pause and train
            if (moveCounter >= 10) {
                // Clear the screen and print "Training"
                clear();
                printw("Training\n");
                refresh();

                // Sleep for 10 seconds
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);
                usleep(sleep_time);

                // Reset the move counter
                moveCounter = 0;

                // TODO: Add your training code here

                // Clear the screen and resume the game
                clear();
                gameOver = false;
        }
        } else {
        // If the game is over, reset the game state to start again
        gameOver = false;
        setup(); // Assuming setup() reinitializes the game state
        set_walls();
        generate_food();
    }

    // Refresh the screen to update the output
    refresh();

    // Sleep for a set duration
    usleep(sleep_time);
      }

      // Ending the game
      EndGame();
      system("clear");

      // Print final game state
      print_board_ended();

      // Print current Q-values with corresponding directions
      std::cout << "\nCurrent Q-Values: ";
      std::cout << "UP: " << current_q_values[0] << ", "
                << "RIGHT: " << current_q_values[1] << ", "
                << "DOWN: " << current_q_values[2] << ", "
                << "LEFT: " << current_q_values[3] << std::endl;

      // Print next Q-values with corresponding directions
      std::cout << "Next Q-Values: ";
      std::cout << "UP: " << next_q_values[0] << ", "
                << "RIGHT: " << next_q_values[1] << ", "
                << "DOWN: " << next_q_values[2] << ", "
                << "LEFT: " << next_q_values[3] << std::endl;

      // Print current direction in one-hot encoded format
      std::cout << "Direction: [" 
                << (dir == UP) << " " 
                << (dir == RIGHT) << " " 
                << (dir == DOWN) << " " 
                << (dir == LEFT) << "]" << std::endl;

    return 0;
  }

  // Exit
  else if (choice == 4){
    return 0;
  }

  // Invalid choice
  else{
    std::cout << "Invalid choice" << std::endl;
    return 0;
  }
}
