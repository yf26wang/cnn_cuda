# Compiler
NVCC = nvcc

BUILD_DIR = build
ARCH = sm_86 # modify if needed

# Target binary
TARGET = ${BUILD_DIR}/cnn_cuda

# Source files
SRC = cnn_cuda.cu

# Compile flags
NVCC_FLAGS = -O3 -arch=${ARCH}

# Include / library paths (adjust if needed)
INCLUDES =
LIBS =

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)