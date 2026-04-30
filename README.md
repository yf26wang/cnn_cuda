# cnn_cuda
Author: Yi Fei Wang

# Overview
Running a simplified cnn algorithm and its 3 layers (convolution, ReLU, output) using the gpu instead of the cpu. The input 100x100 input matrix is first put through the convolutions layer consisting of 10 neurons each with a 5x5 filter. The input matrix is divided into 5x5 sections and dotted with filter in each neuron. This results in 20x20 x10 output. The next layer is the ReLU layer, where the negative values from the previous layer are set to 0. The last layer is the output layer consisting of 10 neurons each with a 4000x1 weight matrix. The 20x20x10 output is flattened into a 4000x1 matrix and dotted against the weight matrix in each neuron. The resulting output is a 10x1 output array of prediction results.

The cuda device code contains two \_\_global\_\_ functions that can be invoked from the host. First, the compute_cnn_part1() function contains the convolution layer, ReLU layer, and the flattening part of the output layer. The function runs with 10 blocks each representing a neuron, and 400 threads used per block representing a 20x20 matrix result. Each thread is responsible for the following:

1. Calculating the dot product between a 5x5 area in the input matrix and its convolution filter
2. Perform ReLU on the dot product by setting negative values to 0
3. Write the result to the 4000 length output array

Next, the computer_cnn_part2() function contains the rest of output layer. The function runs with 10 blocks each representing a neuron, and 1024 threads per block. The first 1000 threads of the 1024 threads will calculate the element wise product of all elements with their weights, and sum the results that they themselves calculated together. (e.g. 0th thread will sum together the element wise product of the 0th, 1000th, 2000th, and 3000th element) This will result in an array of 1000 length to be summed together for the final dot product. The function will then run a divide and conquer reduction algorithm as described in the nvivdia docs to sum together the 1000 elements and place the result in the 0 entry of the shared array. The final step is to copy that result to the output array on the appropriate neuron/block index.

# Generate Inputs
```
python3 generate.py {num_samples}
```
Generates a cnn.csv file with random values for cnn filters and output neurons, and a input.csv file with {num_samples} number of random 100x100 input matrices

# Build & Run
```
make
./build/cnn_cuda ./input/cnn.csv ./input/in.csv
```

# Results
```
Running comparision on 200 inputs...
CPU time elapsed: 18366us
CUDA time elapsed: 18502us
```

The performance of the gpu implementation is similar to the cpu implementation, with the cpu implementation a little faster. To further increase performance in the future, more shared memories can be used in the kernel functions to boost performance by saving global memory bandwidth, more parallelism can be achieved can be achieved by batching many sets of inputs together, etc.