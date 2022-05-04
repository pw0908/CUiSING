CC=g++ -std=c++0x
DIR_INC=
CFLAGS=-Wall -g -O3 $(DIR_INC)
DIR_SRC= src
CPP_FILES=$(wildcard ${DIR_SRC}/*.cpp) $(wildcard ${DIR_SRC}/*/*.cpp) $(wildcard ${DIR_SRC}/*/*/*.cpp)
SRC=$(CPP_FILES)
OBJ=$(SRC:.cpp=.o)

TARGET= Ising

defaut: $(TARGET)
	-rm $(OBJ)

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean
clean:
	rm $(TARGET) $(OBJ)