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

// 1D-Array to store the Board States
int board_states[round_actions] = {0};

// 1D-Array to store the Actions Taken
int actions[round_actions] = {0};

// 1D-Aray to store the Rewards
int rewards[round_actions] = {0};

// Deque to store the Snake
std::deque<std::pair<int, int>> snake;

// Direction of the Snake
Direction dir;

// Global variable to ensure srand is called only once
bool isRandomSeeded = false;

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
    refresh();  // Refresh the screen
}

// Print Board Function outside of ncurses mode once game
// has ended
void print_board_ended(){
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

// Logic Function
void Logic(){
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
      if (!isRandomSeeded) {
        srand(time(nullptr));
        isRandomSeeded = true;
      }
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
    nodelay(stdscr, TRUE);

    setup();
    set_walls();

    generate_food();

    while(!gameOver){
      update_snake();
      Input();
      Logic();
      // sleep for 5 seconds
      print_board();
      usleep(sleep_time);
    }
   
    board_to_1D();
    EndGame();
    system("clear");
    print_board_ended();
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
