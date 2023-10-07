#pragma once

// Declarations for the functions
void Setup();
void GenerateFood(); // Add this line
void Draw();
void Input();
void Logic();
void EndGame();
void Draw_Ended();

extern bool gameOver;
extern int height;
extern int width;

extern int x, y, foodX, foodY, score;

enum Direction { UP, DOWN, LEFT, RIGHT };
extern Direction dir;
