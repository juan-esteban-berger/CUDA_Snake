#include <ncurses.h>
#include "snake.h"

int main() {
	// Initialize Curses Mode
	initscr();
	cbreak();
	noecho();

	// Set Walls
	setWalls();

	// Randomly Generate the Food, Direction, and Snake
	generateFood();
	generateDirection();
	generateSnake();

	// Print Board
	clear();
	printBoard();
	refresh();

	// Wait for User Input
	getch();

	// End Curses Mode
	endwin();

	return 0;
}
