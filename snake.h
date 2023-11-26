#ifndef SNAKE_H
#define SNAKE_H

// Variables
const int bd_height = 20;
const int bd_width = 20;

enum Direction { UP, DOWN, LEFT, RIGHT };
enum InputMode { USER_INPUT, AI_INPUT, RANDOM_INPUT };

// Menu
int printMenu();

// Setup Game
void setWalls();
void generateFood();
void generateDirection();
void generateSnake();
void setupGame();

// Reset Board
void resetBoard();

// Game Functions
Direction oppDir(Direction dir);
void getInput();
void gameLogic();
void printBoard();

// Machine Learning Functions
double* collectFeatures();
double calculateReward();

// Execute Game
void executeGame(bool resetOnDeath, InputMode mode, int lagTime);

#endif

