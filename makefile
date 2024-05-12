TARGET  :=  -o Pri-Render.exe
TARGETDEBUG := -o Pri-Render-Debug.exe

CC      := nvcc

SOURCE_DIR := src
CUH_SOURCES := $(wildcard $(addsuffix /*.cuh, $(SOURCE_DIR)))
CU_SOURCES := $(wildcard $(addsuffix /*.cu, $(SOURCE_DIR)))
CPP_SOURCES := $(wildcard $(addsuffix /*.cpp, $(SOURCE_DIR)))
CPP_HEADERS := $(wildcard $(addsuffix /*.h, $(SOURCE_DIR)))



ALL_FILES := $(CPP_HEADERS) $(CPP_SOURCES) $(CUH_SOURCES) $(CU_SOURCES)

CFLAGS  :=  -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wshadow -Xcompiler -Wwrite-strings 
LDFLAGS := -lassimp 
SPEEDFLAGS := -std=c++20 -O3 
DEBUGFLAGS := -g -lassimp
WARNING := -diag-suppress 20012

STYLE_CHECKER := cpplint
STYLE_HEADERS := --headers=cuh,h,hpp
STYLE_EXTENSIONS := --extensions=c,c++,cc,cpp,cu,cuh

all: compile
# Compile
compile:
	@echo "Compiling"
	$(CC) $(TARGET) $(CPP_SOURCES) $(CU_SOURCES) $(WARNING) $(LDFLAGS) $(SPEEDFLAGS)

# Debug Build
debug:
	@echo "Compiling Debug Build"
	$(CC) $(TARGETDEBUG) $(CPP_SOURCES) $(CU_SOURCES) $(DEBUGFLAGS)

#Style Check
style:
	@echo "Style Checking"
	$(STYLE_CHECKER) $(STYLE_HEADERS) $(STYLE_EXTENSIONS) $(ALL_FILES)

clean:
	rm -fr *.o *.out *.exe