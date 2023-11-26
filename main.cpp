#include <ncurses.h>
#include "snake.h"

bool resetOnDeath = false;

int main() {
	// Initialize Curses Mode
	initscr();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);

	int choice = printMenu();

	// Normal Mode
	if (choice == 1) {
		resetOnDeath = false;
		executeGame(resetOnDeath, USER_INPUT, 170);
		getch();
	}
	else if (choice == 2) {
		resetOnDeath = true;
		executeGame(resetOnDeath, RANDOM_INPUT, 170);
	}
	else if (choice == 3) {
		resetOnDeath = true;
		executeGame(resetOnDeath, RANDOM_INPUT, 170);
	}

	// End Curses Mode
	endwin();

	return 0;
}
