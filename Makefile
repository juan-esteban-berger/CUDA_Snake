CXX = g++
CXXFLAGS = -std=c++11
TARGET = snake

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -r $(TARGET)
