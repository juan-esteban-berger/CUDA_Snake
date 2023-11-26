#ifndef SNAKE_H
#define SNAKE_H

const int bd_height = 20;
const int bd_width = 20;

enum Direction { UP, DOWN, LEFT, RIGHT };

void setWalls();
void generateFood();
void generateDirection();
void updateDirection();
void generateSnake();
void updateSnake();
void gameLogic();
void printBoard();

#endif
