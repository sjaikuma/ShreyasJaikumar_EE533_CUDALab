# NVCC Compiler
NVCC = nvcc

# Defining the executables
TARGETS_CUDA = p1 p2

# Combine all targets
TARGETS = $(TARGETS_CUDA)

# Build Process
all: $(TARGETS)

# Build CUDA targets
p1: p1.cu
	$(NVCC) -o p1 p1.cu

p2: p2.cu
	$(NVCC) -o p2 p2.cu

# Run targets and redirect output
run_p1: p1
	./p1

run_p2: p2
	./p2

job: job.sl
	cp job.sl ./job.sl
	
# Clean up the executables and output files
clean:
	rm -f $(TARGETS) job.sl