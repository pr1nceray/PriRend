TARGET  :=  -o Pri-Render.exe
TARGETDEBUG := -o Debug-Build.exe

CC      := nvcc


SOURCE_DIR := src
CUH_SOURCES := $(shell find . -name "*.cuh") 
CU_SOURCES := $(shell find . -name "*.cu") 
CPP_SOURCES := $(shell find . -name "*.cpp") 
CPP_HEADERS := $(shell find . -name "*.h") 



ALL_FILES := $(CPP_HEADERS) $(CPP_SOURCES) $(CUH_SOURCES) $(CU_SOURCES)

CFLAGS  :=  -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wshadow -Xcompiler -Wwrite-strings  
CUDAFLAGS := -maxrregcount 64 -arch=compute_86 -rdc=true
LDFLAGS := -lassimp -lcudadevrt -lcurand 
SPEEDFLAGS := -std=c++20 -O3 
DEBUGFLAGS := -g -G

# error in glm::vec libraries that is very annoying.
WARNING := -diag-suppress 20012

STYLE_CHECKER := cpplint
STYLE_HEADERS := --headers=cuh,h,hpp
STYLE_EXTENSIONS := --extensions=c,c++,cc,cpp,cu,cuh

all: compile
# Compile
compile:
	@echo "Compiling"
	$(CC) $(CPP_SOURCES) $(CU_SOURCES) $(WARNING) $(LDFLAGS) $(SPEEDFLAGS) $(CUDAFLAGS) $(TARGET) 

# Debug Build
debug:
	@echo "Compiling Debug Build"
	$(CC) $(CPP_SOURCES) $(CU_SOURCES) $(WARNING) $(LDFLAGS) $(DEBUGFLAGS) $(CUDAFLAGS) $(TARGETDEBUG)

#Style Check
style:
	@echo "Style Checking"
	$(STYLE_CHECKER) $(STYLE_HEADERS) $(STYLE_EXTENSIONS) $(ALL_FILES)

clean:
	rm -fr *.o *.out *.exe