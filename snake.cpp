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

// 1D-Array to store the Board States
int board_states[round_actions] = {0};

// 1D-Array to store the Actions Taken
int actions[round_actions] = {0};

// 1D-Aray to store the Rewards
int rewards[round_actions] = {0};

// Deque to store the Snake
std::deque<std::pair<int, int>> snake;

////////////////////////////////////////////////////////////////////////
// Preliminary Functions
// Setup Function
void setup(){
    gameOver = false;
    score = 0;
    x = rand() % (bd_size - 2) + 1;
    y = rand() % (bd_size - 2) + 1;
    snake.clear();
    snake.push_back({x, y});
    // GenerateFood();
}
// Print Board Function
void print_board(){
  for (int i=0; i < bd_size; i++){
    for (int j=0; j < bd_size; j++){
      if (board[i][j] == 0){
        std::cout << "  ";
      }
      else if (board[i][j] == 1){
        std::cout << "# ";
      }
      else if (board[i][j] == 2){
        std::cout << "O ";
      }
      else if (board[i][j] == 3){
        std::cout << "F ";
      }
      else{
        std::cout << "E ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << "Score: " << score << "\n";
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
  for(std::deque<std::pair<int, int>>::iterator it = snake.begin(); it != snake.end(); ++it) {
      int x = it->first;
      int y = it->second;
      board[y][x] = 2;
  }
}

// Generate Food Function
void generate_food() {
    bool foodPlaced = false;

    while(!foodPlaced) {
        // Generate random position for food
        foodX = rand() % (bd_size - 2) + 1;
        foodY = rand() % (bd_size - 2) + 1;

        // Assume this is a valid position for now
        bool isValidPosition = true;

        // Check if the generated position is on the snake's body
        for(std::deque<std::pair<int, int>>::iterator it = snake.begin(); it != snake.end(); ++it) {
            if(it->first == foodX && it->second == foodY) {
                isValidPosition = false;
                break;
            }
        }

        // If the position is valid and not on the walls, place the food there
        if(isValidPosition && board[foodY][foodX] == 0) {
            board[foodY][foodX] = 3;  // Set the food position on the board
            foodPlaced = true;
        }
    }
}

// Board to 1D-Array Function
void board_to_1D(){
  for(int i = 0; i < bd_size; i++){
    for(int j = 0; j < bd_size; j++){
      board1D[i*bd_size + j] = board[i][j];
    }
  }
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
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> outputs;
    int inputSize, outputSize;
    std::function<double(double)> activationFunction;

    Layer(int inputSize, int outputSize, std::function<double(double)> activationFunction)
        : inputSize(inputSize), outputSize(outputSize), activationFunction(activationFunction) {
      // Randomly initialize weights and biases
      weights.resize(outputSize, std::vector<double>(inputSize));
      biases.resize(outputSize);
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          weights[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5;
        }
        biases[i] = static_cast<double>(rand()) / RAND_MAX - 0.5;
      }
    }

    std::vector<double> forward(const std::vector<double>& input) {
      outputs.resize(outputSize);
      for (int i = 0; i < outputSize; ++i) {
        outputs[i] = 0;
        for (int j = 0; j < inputSize; ++j) {
          outputs[i] += input[j] * weights[i][j];
        }
        outputs[i] += biases[i];
        outputs[i] = activationFunction(outputs[i]);
      }
      return outputs;
    }
};

// Neural Network Class
class NeuralNetwork{
  public:
    std::vector<Layer> layers;
        void addLayer(int inputSize, int outputSize, std::function<double(double)> activationFunction) {
        layers.emplace_back(inputSize, outputSize, activationFunction);
    }

    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> currentOutput = input;
        for (Layer& layer : layers) {
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

  // Seed random number generator
  srand(time(nullptr));

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
   
    board_to_1D();

    return 0;
  }
  // AI Mode
  else if (choice == 2){
    system("clear");
    std::cout << "Please select a model: " << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "placeholder_model.csv" << std::endl;
    std::cout << "4. Exit" << std::endl;
    std::cout << "Enter your choice: ";
    std::cin >> choice;

    system("clear");

//////////////////////////////////////////////////////////////////////
  NeuralNetwork nn;
  nn.addLayer(1604, 800, relu);
  nn.addLayer(800, 400, relu);
  nn.addLayer(200, 4, relu);


  std::vector<double> input = {1.0, 0.5, -1.5};
  std::cout << "Input: ";
  int count = 0;
  for (double value : input) {
      std::cout << value << " ";
      if (count == 3) {  // This condition will be true after printing the fourth element
          break;
      }
      count++;
  }
  std::cout << "\n";
  std::vector<double> output = nn.predict(input);

  std::cout << "Predictions: ";
  for (double value : output) {
      std::cout << value << " ";
   }
  std::cout << "\n";

//////////////////////////////////////////////////////////////////////

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
   
    board_to_1D();
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
   
    board_to_1D();
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
