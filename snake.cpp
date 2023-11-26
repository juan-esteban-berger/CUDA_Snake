#include <ncurses.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include "snake.h"

// Initialize Variables
int score = 0;

std::vector<int> snakeY;
std::vector<int> snakeX;

Direction currentDir = UP;
std::string dirString = "Direction: ";

int board[bd_height][bd_width] = {0};

// Functions
void setWalls() {
	for (int i = 0; i < bd_height; i++){
		for (int j = 0; j < bd_width; j++){
			if (i == 0 | i == bd_height - 1
				| j == 0 | j == (bd_width -1)) {
				board[i][j] = 1;
			}
		}
	}
}

void generateFood() {
	int foodY;
	int foodX;

	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	do {
		foodY = std::rand() % (bd_height - 2) + 1;
		foodX = std::rand() % (bd_width -2) + 1;
	} while (board[foodY][foodX] != 0);
	
	board[foodY][foodX] = -1;
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

void updateDirection() {
	// TODO
}

void generateSnake() {
	int startX;
	int startY;

	snakeY.clear();
	snakeX.clear();

	do {
		startY = std::rand() % (bd_height - 2) + 1;
		startX = std::rand() % (bd_width - 2) + 1;
	} while (board[startY][startX] != 0);

	board[startY][startX] = 2;
}

void updateSnake() {
	// TODO
}

void gameLogic() {
	// TODO
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
		case UP: dirString += "UP"; break;
		case DOWN: dirString += "DOWN"; break;
		case LEFT: dirString += "LEFT"; break;
		case RIGHT: dirString += "RIGHT"; break;
	}	
	mvprintw(bd_height + 1, 0, "%s", dirString.c_str());
}
