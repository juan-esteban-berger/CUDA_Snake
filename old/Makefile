# Compiler and Flags
CC = g++
CFLAGS = -Wall -Iinclude
LDLIBS = -lncurses

# Source and Object Files
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

all: snake

snake: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDLIBS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f snake $(OBJ)

.PHONY: all clean
