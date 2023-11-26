#include <ncurses.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "snake.h"

// Variables
int score = 0;

bool wallsSet = false;
bool ateFood = false;
bool gameOver = false;

int foodX;
int foodY;

std::vector<int> snakeY;
std::vector<int> snakeX;

Direction currentDir = UP;
std::string dirString;

int board[bd_height][bd_width] = {0};

// Menu
int printMenu() {
	printw("Please choose a mode:\n");
	printw("1. Normal Mode \n");
	printw("2. AI Mode \n");
	printw("3. Training Mode \n");

	int choice = getch() - '0';

	return choice;
}

// Setup Game
void setWalls() {
	if (!wallsSet) {
		for (int i = 0; i < bd_height; i++){
			for (int j = 0; j < bd_width; j++){
				if (i == 0 | i == bd_height - 1
					| j == 0 | j == (bd_width -1)) {
					board[i][j] = 1;
				}
			}
		}
		wallsSet = true;
	}
}

void generateFood() {
	int tempFoodY;
	int tempFoodX;

	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	do {
		tempFoodY = std::rand() % (bd_height - 2) + 1;
		tempFoodX = std::rand() % (bd_width -2) + 1;
	} while (board[tempFoodY][tempFoodX] != 0);
	
	board[tempFoodY][tempFoodX] = -1;

	foodY = tempFoodY;
	foodX = tempFoodX;
}

void generateDirection() {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	
	int randDir = std::rand() % 4 + 1;

	switch (randDir) {
		case 1:
			currentDir = UP;
			break;
		case 2:
			currentDir = DOWN;
			break;
		case 3:
			currentDir = LEFT;
			break;
		case 4:
			currentDir = RIGHT;
			break;
	}
}

void generateSnake() {
	// Initialize Variables to store location
	// of the snake's head and tail
	int startX;
	int startY;
	int tailX;
	int tailY;

	// Variable for checking if the snake's position
	// is valid
	bool validPos = false;

	// Clear the Snake
	snakeY.clear();
	snakeX.clear();
	
	// Set Random Seed
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	
	// Generate the position of the snake
	while (!validPos) {
		// Generate Random Head Position
		startY = std::rand() % (bd_height - 2) + 1;
		startX = std::rand() % (bd_width - 2) + 1;

		// Skip the rest of the loop if the position
		// is already occupied
		if (board[startY][startX] != 0) {
			continue;
		}

		// Determine the tail position based on the
		// current direction
		tailY = startY;
		tailX = startX;
		switch (currentDir) {
            case UP:
                tailY = startY + 1;
                break;
            case DOWN:
                tailY = startY - 1;
                break;
            case LEFT:
                tailX = startX + 1;
                break;
            case RIGHT:
                tailX = startX - 1;
                break;
        }

		// Check if the tail position is inside the
		// walls and not occupied
		if (tailY >= 0 &&
			tailY < bd_height &&
			tailX >= 0 &&
			tailX < bd_width &&
			board[tailY][tailX] == 0) {
			// Set validPos to True
			validPos = true;
		}
	}
		// Append the snake's head to the vector
		snakeY.push_back(startY);
		snakeX.push_back(startX);
		
		// Append the snake's tail to the vector
		snakeY.push_back(tailY);
		snakeX.push_back(tailX);

		// Update the Board
		board[startY][startX] = 2;
		board[tailY][tailX] = 2;
}

void setupGame() {
	setWalls();
	generateFood();
	generateDirection();
	generateSnake();
}

// Reset Board
void resetBoard() {
	// Set all values in the board to zero
	for (int i = 0; i < bd_height; ++i) {
		for (int j = 0; j < bd_width; ++j) {
			board[i][j] = 0;
		}
	}
	// Set WallsSet to Zero
	wallsSet = false;
}

// Game Functions
// Function to get the opposite direction
Direction oppDir(Direction dir) {
    switch (dir) {
        case UP: return DOWN;
        case DOWN: return UP;
        case LEFT: return RIGHT;
        case RIGHT: return LEFT;
    }
    return dir; // Fallback, should not happen
}

void getInput(InputMode mode) {
    int ch;
    switch (mode) {
        // USER_INPUT Mode
        case USER_INPUT:
            ch = getch();
            switch (ch) {
                case 'w':
                    if (currentDir != DOWN) currentDir = UP;
                    break;
                case 's':
                    if (currentDir != UP) currentDir = DOWN;
                    break;
                case 'a':
                    if (currentDir != RIGHT) currentDir = LEFT;
                    break;
                case 'd':
                    if (currentDir != LEFT) currentDir = RIGHT;
                    break;
            }   
            break;

        // AI_INPUT Mode
        case AI_INPUT:
            // AI logic here
            break;

        // RANDOM_INPUT Mode
        case RANDOM_INPUT:
            Direction newDir;
            do {
                newDir = static_cast<Direction>(std::rand() % 4);
            } while (newDir == oppDir(currentDir));
            currentDir = newDir;
            break;
    }
}

void gameLogic() {
	// Get the Location of the Snake's Head
	int headY = snakeY.front();
	int headX = snakeX.front();

	// Update the location of the snake's head
	switch(currentDir) {
		case UP: headY--; break;
		case DOWN: headY++; break;
        case LEFT: headX--; break;
        case RIGHT: headX++; break;
	}

	// Check if the game is over
	if (headY < 0 || headY >= bd_height || headX < 0 || headX >= bd_width || board[headY][headX] == 1 || board[headY][headX] == 2) {
        gameOver = true;
        return;
    }

	// Check for food
    if (board[headY][headX] == -1) {
        score++;
        generateFood();
		ateFood = true; // If Food was eaten, set ateFood to true
    } 
	// If no food was eaten, remove the tail 
	else {
        int tailY = snakeY.back();
        int tailX = snakeX.back();
        snakeY.pop_back();
        snakeX.pop_back();
        board[tailY][tailX] = 0;
    }

    // Add new head position
    snakeY.insert(snakeY.begin(), headY);
    snakeX.insert(snakeX.begin(), headX);
    board[headY][headX] = 2;
}

void printBoard() {
	for (int i = 0; i < bd_height; i++){
		for (int j = 0; j < bd_width; j++){
			if (board[i][j] == 0) {
				mvprintw(i, j * 2, "  ");
			}
			else if (board[i][j] == -1) {
				mvprintw(i, j * 2, "F ");
			}
			else if (board[i][j] == 1) {
				mvprintw(i, j * 2, "#  ");
			}
			else if (board[i][j] == 2) {
				mvprintw(i, j * 2, "O  ");
			}
		}
	}
	mvprintw(bd_height, 0, "Score: %d", score);
	switch (currentDir) {
		case UP: dirString = "UP"; break;
		case DOWN: dirString = "DOWN"; break;
		case LEFT: dirString = "LEFT"; break;
		case RIGHT: dirString = "RIGHT"; break;
	}	
	mvprintw(bd_height + 1, 0, "%s", dirString.c_str());
}

// Machine Learning Functions
double* collectFeatures() {
	// Create array to store one row of features
	static double features[12];

	// Get the position of the snake's head
	int headY = snakeY.front();
	int headX = snakeX.front();

	// Danger Features
	features[0] = (headY == 0 ||
					board[headY - 1][headX] == 1 ||
					board[headY - 1][headX] == 2) ? 1.0 : 0.0; // Danger Up
	features[1] = (headY == bd_height - 1 ||
					board[headY + 1][headX] == 1 ||
					board[headY + 1][headX] == 2) ? 1.0 : 0.0; // Danger Down
	features[2] = (headX == 0 ||
					board[headY][headX - 1] == 1 ||
					board[headY][headX - 1] == 2) ? 1.0 : 0.0; // Danger Left
	features[3] = (headX == bd_width - 1 ||
					board[headY][headX + 1] == 1 ||
					board[headY][headX + 1] == 2) ? 1.0 : 0.0; // Danger Right

	// Current Direction Features
	features[4] = (currentDir == UP) ? 1.0 : 0.0;    // Direction Up
    features[5] = (currentDir == DOWN) ? 1.0 : 0.0;  // Direction Down
    features[6] = (currentDir == LEFT) ? 1.0 : 0.0;  // Direction Left
    features[7] = (currentDir == RIGHT) ? 1.0 : 0.0; // Direction Right

	// Food Relative Position Features
	features[8] = (foodY < headY) ? 1.0 : 0.0;  // Food Up
    features[9] = (foodY > headY) ? 1.0 : 0.0;  // Food Down
    features[10] = (foodX < headX) ? 1.0 : 0.0; // Food Left
    features[11] = (foodX > headX) ? 1.0 : 0.0; // Food Right

	return features;
}

double calculateReward() {
	if (gameOver) {
		return -10.0;
	}
	if (ateFood) {
		return 10.0;
	}
	return 0.0;
}

// Execute Game
void executeGame(bool resetOnDeath, InputMode mode, int lagTime) {
	setupGame();
	nodelay(stdscr, true);
	
	while (!gameOver) {
		clear();
		getInput(mode);
		gameLogic();
		double* features = collectFeatures();
		double reward = calculateReward();
		printBoard();
		refresh();

		if (gameOver && resetOnDeath) {
			score = 0;
			resetBoard();
			setupGame();
			gameOver = false;
		}

		ateFood = false;

		std::this_thread::sleep_for(std::chrono::milliseconds(lagTime));

	}
	nodelay(stdscr, false);
}
