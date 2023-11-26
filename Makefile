CXX = g++
CXXFLAGS = -std=c++11
LDFLAGS = -lncurses
TARGET = snake
SRCS = main.cpp snake.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -r $(TARGET)
