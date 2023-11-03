#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <random>
#include <ctime>
#include <ncurses.h>
#include <vector>
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

// Create 1D-Array for Board
const int bd_area = bd_size * bd_size;
int board1D[bd_area] = {0};

// Create variable for number of actions before retraining
const int round_actions = 1000;

// // 1D-Array to store the Board States
// int board_states[round_actions] = {0};
//
// // 1D-Array to store the Actions Taken
// int actions[round_actions] = {0};
//
// // 1D-Aray to store the Rewards
// int rewards[round_actions] = {0};

// Deque to store the Snake
std::deque<std::pair<int, int>> snake;

// Direction of the Snake
Direction dir;

// Global variable to ensure srand is called only once
bool isRandomSeeded = false;

// Global variable for checking if the snake has eaten food
bool hasEatenFood;

// Global variables for features and rewards
std::vector<double> features;
std::vector<double> rewards;

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

    mvprintw(bd_size, 0, "Score: %d", score);
    mvprintw(bd_size + 1, 0, "Features: ");
    for (int i = 0; i < 4 && i < features.size(); i++) {
        mvprintw(bd_size + 1, 11 + i * 6, "%1.1f ", features[i]);
    }
    mvprintw(bd_size + 1, 35, "... ");

    // Print the rewards for each action
    mvprintw(bd_size + 2, 0, "Rewards: U:%1.1f R:%1.1f D:%1.1f L:%1.1f", 
             rewards[0], rewards[1], rewards[2], rewards[3]);

    refresh();  // Refresh the screen
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

  // Print the first four features
  std::cout << "Features: ";
  for (int i = 0; i < 4 && i < features.size(); i++) {
      std::cout << features[i] << " ";
  }
  std::cout << "... " << std::endl;

  // Print the rewards for each action
  std::cout << "Rewards: U:" << rewards[0] 
            << " R:" << rewards[1] 
            << " D:" << rewards[2] 
            << " L:" << rewards[3] << std::endl;
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

// AI Input Function
void AI_Input(const std::vector<double>& output) {
    // Find the index of the maximum value in output
    int maxIndex = 0;
    double maxValue = output[0];
    for (int i = 1; i < output.size(); ++i) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            maxIndex = i;
        }
    }

    // Map the index to a direction
    Direction newDir;
    if (maxIndex == 0) newDir = UP;
    else if (maxIndex == 1) newDir = DOWN;
    else if (maxIndex == 2) newDir = LEFT;
    else if (maxIndex == 3) newDir = RIGHT;

    // Update dir based on newDir, taking into account the current direction to prevent snake moving into itself
    if (newDir == UP && dir != DOWN) dir = UP;
    else if (newDir == DOWN && dir != UP) dir = DOWN;
    else if (newDir == LEFT && dir != RIGHT) dir = LEFT;
    else if (newDir == RIGHT && dir != LEFT) dir = RIGHT;

    refresh();
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
std::vector<double> get_features(){
    // The total number of features is 11 (initial features) + bd_size * bd_size (board state)
    std::vector<double> features(11 + bd_size * bd_size, 0.0);

    // Get the head of the snake
    int headX = snake.front().first;
    int headY = snake.front().second;

    // Check if there is a wall or snake body in front of the snake
    if (dir == UP && (board[headY - 1][headX] == 1 || board[headY - 1][headX] == 2)) {
        features[0] = 1.0;
    } else {
        features[0] = 0.0;
    }
    // Check if there is a wall or snake body behind the snake
    if (dir == DOWN && (board[headY + 1][headX] == 1 || board[headY + 1][headX] == 2)) {
        features[1] = 1.0;
    } else {
        features[1] = 0.0;
    }
    // Check if there is a wall or snake body to the left of the snake
    if (dir == LEFT && (board[headY][headX - 1] == 1 || board[headY][headX - 1] == 2)) {
        features[2] = 1.0;
    } else {
        features[2] = 0.0;
    }
    // Check if there is a wall or snake body to the right of the snake
    if (dir == RIGHT && (board[headY][headX + 1] == 1 || board[headY][headX + 1] == 2)) {
        features[3] = 1.0;
    } else {
        features[3] = 0.0;
    }

    // Check if the food is above the snake
    if (foodY < headY) {
        features[4] = 1.0;
    } else {
        features[4] = 0.0;
    }
    // Check if the food is below the snake
    if (foodY > headY) {
        features[5] = 1.0;
    } else {
        features[5] = 0.0;
    }
    // Check if the food is to the left of the snake
    if (foodX < headX) {
        features[6] = 1.0;
    } else {
        features[6] = 0.0;
    }
    // Check if the food is to the right of the snake
    if (foodX > headX) {
        features[7] = 1.0;
    } else {
        features[7] = 0.0;
    }

    // Check if the current direction is UP
    if (dir == UP) {
        features[8] = 1.0;
    } else {
        features[8] = 0.0;
    }
    // Check if the current direction is DOWN
    if (dir == DOWN) {
        features[9] = 1.0;
    } else {
        features[9] = 0.0;
    }
    // Check if the current direction is LEFT
    if (dir == LEFT) {
        features[10] = 1.0;
    } else {
        features[10] = 0.0;
    }
    // Check if the current direction is RIGHT
    if (dir == RIGHT) {
        features[11] = 1.0;
    } else {
        features[11] = 0.0;
    }

    // Flatten the board into a 1D vector and append it to features.
    // Start from index 11 since that's where the board state should begin.
    for (int i = 0; i < bd_size; i++) {
        for (int j = 0; j < bd_size; j++) {
            features[11 + i * bd_size + j] = static_cast<double>(board[i][j]);
        }
    }

    return features;
}

// Function to generate_outputs using a specified reward function
std::vector<double> get_rewards(bool gameOver, bool hasEatenFood, int currentScore){

  // Initialize a vector to store the rewards
  // Starting with a small negative reward for all actions
  std::vector<double> rewards(4, -0.1);

  return rewards;
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
      rewards = get_rewards(gameOver, hasEatenFood, score);

      print_board();

      // Sleep 
      usleep(sleep_time);

      
    }
   
    EndGame();
    system("clear");

    // Get Features
    features = get_features();
    // Get Rewards
    rewards = get_rewards(gameOver, hasEatenFood, score);

    print_board_ended();
    return 0;
  }

  // AI Moide
  else if (choice == 2){
    // Print menu for choosing a model
    system("clear");
    std::cout << "Please select a model: " << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "4. Exit" << std::endl;
    std::cout << "Enter your choice: ";
    std::cin >> choice;

    // Clear the screen for the game
    system("clear");

    //////////////////////////////////////////////
    // Initialize the neural network
    srand(time(nullptr));

    NeuralNetwork nn;
    nn.addLayer(1610, 800, relu);
    nn.addLayer(800, 400, relu);
    nn.addLayer(400, 4, relu);
    //////////////////////////////////////////////

    // Initialize ncurses mode
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();
    curs_set(0);

    // Setup the game
    setup();
    set_walls();
    generate_food();

    // Create input and output vectors
    std::vector<double> input(1610);
    std::vector<double> output;

    // Main game loop
    while(!gameOver){

//////////////////////////////////////////////////////////////
// Randomly initialize an input of length 1610
for (double& value : input) {
    value = static_cast<double>(rand()) / RAND_MAX - 0.5;
}

// Get the predictions
output = nn.predict(input);
//////////////////////////////////////////////////////////////

      // Print Board and sleep
      update_snake();
      
      // Get Features
      features = get_features();
      // Get Rewards
      rewards = get_rewards(gameOver, hasEatenFood, score);

      print_board();

      // Print input
      printw("\n");
      printw("Input: ");
      int count = 0;
      for (double value : input) {
          printw("%.2f ", value);
          // Only print first 4 elements
          if (count == 3) {
              break;
          }
          count++;
      }
      printw("...\n");

      // Print output
      printw("Predictions: ");
      for (double value : output) {
          printw("%.2f ", value);
      }
      printw("\n");

      // Move the snake based on the predictions
      AI_Input(output);
      hasEatenFood = Logic();

      // Generate and store features

      // Sleep
      usleep(sleep_time);
    }

    // End the game and ncurses mode
    EndGame();

    // Print the board (without n curses mode)
    system("clear");

    // Get Features
    features = get_features();
    // Get Rewards
    rewards = get_rewards(gameOver, hasEatenFood, score);

    print_board_ended();


    // Print input (without ncurses mode)
    std::cout << "Input: ";
    int count = 0;
    for (double value : input) {
        std::printf("%.2f ", value);
        if (count == 3) {  // This condition will be true after printing the fourth element
            break;
        }
        count++;
    }
    std::cout << "...\n";

    // Print output (without ncurses mode)
    std::cout << "Predictions: ";
    for (double value : output) {
        std::printf("%.2f ", value);
    }
    std::cout << std::endl;

    return 0;
  }

  // Train AI Model
  else if (choice == 3){
    set_walls();

    snake.push_back({2,3});
    snake.push_back({2,4});
    snake.push_back({2,5});
    snake.push_back({2,6});
    snake.push_back({2,7});

    update_snake();
    generate_food();

    print_board();
    std::cout << std::endl;
   
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
