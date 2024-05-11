TARGET  :=  -o Pri-Render.exe
TARGETDEBUG := -o Pri-Render-Debug.exe

CC      := g++

SOURCE_DIR := src
CUH_SOURCES := $(wildcard $(addsuffix /*.cuh, $(SOURCE_DIR)))
CU_SOURCES := $(wildcard $(addsuffix /*.cu, $(SOURCE_DIR)))
CPP_SOURCES := $(wildcard $(addsuffix /*.cpp, $(SOURCE_DIR)))
CPP_HEADERS := $(wildcard $(addsuffix /*.h, $(SOURCE_DIR)))

ALL_FILES := $(CPP_HEADERS) $(CPP_SOURCES) $(CUH_SOURCES) $(CU_SOURCES)

#CFLAGS  := -std=c++20 -pedantic -Wall -Wextra -Wshadow -Wwrite-strings -O3
LDFLAGS := -lsfml-graphics -lsfml-window -lsfml-system -lassimp -lGL
DEBUGFLAGS := -g -lassimp

STYLE_CHECKER := cpplint
STYLE_HEADERS := --headers=cuh,h,hpp
STYLE_EXTENSIONS := --extensions=c,c++,cc,cpp,cu,cuh

all: compile
# Compile
compile:
	@echo "Compiling"
	$(CC) $(TARGET) $(CPP_SOURCES) $(LDFLAGS) 

# Debug Build
debug:
	@echo "Compiling Debug Build"
	$(CC) $(TARGETDEBUG) $(CPP_SOURCES) $(DEBUGFLAGS)

#Style Check
style:
	@echo "Style Checking"
	$(STYLE_CHECKER) $(STYLE_HEADERS) $(STYLE_EXTENSIONS) $(ALL_FILES)

clean:
	rm -fr *.o *.out *.exe