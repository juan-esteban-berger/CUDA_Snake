#include "snake.h"
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <ctime>
#include <ncurses.h>

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

    srand(time(NULL));

    Setup();
    while(!gameOver){
        clear();
        Draw();
        Input();
        Logic();
        usleep(sleep_time);
    }
    EndGame();
    system("clear");
    Draw_Ended();
    return 0;
  }
  // AI Mode
  else if (choice == 2){
    std::cout << "You have selected AI Mode" << std::endl;
    return 0;
  }
  // Train AI Model
  else if (choice == 3){
    std::cout << "You have selected Train AI Model" << std::endl;
    return 0;
  }
  // Exit
  else if (choice == 4){
    std::cout << "You have selected Exit" << std::endl;
    return 0;
  }
  // Invalid choice
  else{
    std::cout << "Invalid choice" << std::endl;
    return 0;
  }
}
