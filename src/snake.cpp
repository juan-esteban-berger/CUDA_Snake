#include "snake.h"
#include <deque>
#include <iostream>
#include <ctime>
#include <unistd.h>
#include <ncurses.h>

// Definitions (and initial values) for the globals
bool gameOver;
int x, y, foodX, foodY, score;
int width = 40;
int height = 20;
Direction dir;
std::deque<std::pair<int, int>> tail;  // Deque to store the snake's body

// Setup Function
void Setup(){
    gameOver = false;
    x = rand() % (width - 2) + 1;
    y = rand() % (height - 2) + 1;
    tail.clear();
    tail.push_back({x, y}); // Initialize the tail with the head's position
    GenerateFood();
}

// GenerateFood Function
void GenerateFood() {
    bool positionOccupied;
    do {
        foodX = (rand() % (width - 2)) + 1;
        foodY = rand()rewar % (height - 2) + 1;

        // Ensure foodX has the same parity as x
        while (x % 2 != foodX % 2) {
            foodX = (foodX + 1) % (width - 2);  // Cycle to the next x-coordinate, but keep it within the playable grid
            if (foodX == 0) foodX = 1;  // Avoid having 0 as an x-coordinate
        }

        positionOccupied = false;

        // Check if the food position overlaps with any part of the snake's tail
        for (auto const &segment : tail) {
            if (segment.first == foodX && segment.second == foodY) {
                positionOccupied = true;
                break;
            }
        }
    } while (positionOccupied || (foodX == x && foodY == y));
}

// Draw Function
void Draw(){
    clear();
    for (int i=0; i<width; i++){
        printw("#");
    }
    printw("\n");

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(j==0 || j==width-1){
                printw("#");
            }
            else if(i==y && j==x){
                printw("O");
            }
            else if(i==foodY && j==foodX){
                printw("F");
            }
            else {
                bool isTailPart = false;
                for (auto const &segment : tail) {
                    if (segment.first == j && segment.second == i) {
                        printw("O");
                        isTailPart = true;
                        break;
                    }
                }
                if (!isTailPart) {
                    printw(" ");
                }
            }
        }
        printw("\n");
    }

    for(int i=0; i<width; i++){
        printw("#");
    }
    printw("\n");
    printw("\n");
    printw("Score: %d\n", score);
    refresh();
}

// Input Function
void Input(){
    int ch;
    ch = getch();
    switch(ch){
        case 'w':
            if(dir != DOWN || tail.size() == 1)  // If moving UP, prevent moving DOWN if there's a tail
                dir = UP;
            break;
        case 'a':
            if(dir != RIGHT || tail.size() == 1)  // If moving LEFT, prevent moving RIGHT if there's a tail
                dir = LEFT;
            break;
        case 's':
            if(dir != UP || tail.size() == 1)  // If moving DOWN, prevent moving UP if there's a tail
                dir = DOWN;
            break;
        case 'd':
            if(dir != LEFT || tail.size() == 1)  // If moving RIGHT, prevent moving LEFT if there's a tail
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
            potentialX -= 2;
            break;
        case DOWN:
            potentialY++;
            break;
        case RIGHT:
            potentialX += 2;
            break;
        default:
            break;
    }

    // Check if the snake would go out of bounds
    if(potentialX > width-2 || potentialX < 0 || potentialY >= height || potentialY < 0){
        gameOver = true;
    }

    // Check if the snake would collide with its tail
    // We'll start from the second element to skip the head during the check
    for (auto it = tail.begin(); it != tail.end(); ++it) {
        if (it->first == potentialX && it->second == potentialY) {
            gameOver = true;
            break;
        }
    }

    // Only update the positions if the game isn't over
    if (!gameOver) {
        x = potentialX;
        y = potentialY;
        tail.push_front({x, y});   // Add new head position to the front of the tail
        tail.pop_back();           // Remove last segment of the tail

        // Check if the snake has eaten the food
        if ((x == foodX || prevX == foodX) && y == foodY) {
            score++;
            GenerateFood();
            tail.push_back({prevX, prevY});  // Add a new segment to the tail
        }
    }
}

// End Game Function
void EndGame() {
    endwin();// End ncurses mode
}

// Draw Function once ncurses mode has ended
void Draw_Ended(){
    for (int i=0; i<width; i++){
        std::cout << "#";
    }
    std::cout << "\n";

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(j==0 || j==width-1){
                std::cout << "#";
            }
            else if(i==y && j==x){
                std::cout << "O";
            }
            else if(i==foodY && j==foodX){
                std::cout << "F";
            }
            else {
                bool isTailPart = false;
                for (auto const &segment : tail) {
                    if (segment.first == j && segment.second == i) {
                        std::cout << "O";
                        isTailPart = true;
                        break;
                    }
                }
                if (!isTailPart) {
                    std::cout << " ";
                }
            }
        }
        std::cout << "\n";
    }

    for(int i=0; i<width; i++){
        std::cout << "#";
    }
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "Score: " << score << "\n";
}
