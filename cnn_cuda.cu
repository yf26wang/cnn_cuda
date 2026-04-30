#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>

#include "helper.h"

// Includes convolution, ReLU, and flattening part of the output layer
__global__ void compute_cnn_part1(const double input[100][100], const double conv_filters[10][5][5], double out_conv[4000]) {
    
    // 10 blocks with 400 threads each
    if (threadIdx.x >= 400) {
        return;
    }

    int row = (threadIdx.x / 20) * 5;
    int col = (threadIdx.x % 20) * 5;
    double conv_res = 0;
    for(int i = 0; i < 5; i++) {
        int row_idx = row + i;
        for (int j = 0; j < 5; j++) {
            int col_idx = col + j;
            conv_res += input[row_idx][col_idx] * conv_filters[blockIdx.x][i][j];
        }
    }
    
    // ReLU
    if (conv_res < 0) {
        conv_res = 0; 
    }

    // output layer flatten
    out_conv[blockIdx.x * 400 + threadIdx.x] = conv_res;
}

// Includes the dot product part of the output layer
#define THREADS_PART2 1024
__global__ void compute_cnn_part2(const double out_conv[4000], double weights[10][4000], double output[10]) {
    if (threadIdx.x >= THREADS_PART2) {
        return;
    }

    // dot product, and sum 4000 entries to 1000 entries
    __shared__ double dot_product[1024];
    dot_product[threadIdx.x] = 0;
    if(threadIdx.x < 1000) {
        for(int i = 0; i < 4; i++) {
            dot_product[threadIdx.x] += weights[blockIdx.x][i*1000 + threadIdx.x] * out_conv[i*1000 + threadIdx.x];
        }
    }
    __syncthreads();
    
    // reduction add 1000 entries (round up to 1024)
    for(int i = 512;i > 0;i /= 2) {
        if(threadIdx.x < i) {
            dot_product[threadIdx.x] += dot_product[threadIdx.x + i];
        }
        __syncthreads();
    }
    output[blockIdx.x] = dot_product[0];
}

void fill_row(std::ifstream & str, double * row, int row_length) {
    std::string line;
    std::getline(str,line);
    std::stringstream lineStream(line);
    std::string cell;
    for(int i = 0; i < row_length; i++) {
        if(std::getline(lineStream,cell,',')) {
            // std::cout << cell << std::endl;
            row[i] = std::stod(cell);
        }
        else {
            std::cout << "row_length mismatch" << std::endl;
            return;
        }
    }
}

 void read_next_input(std::ifstream & str, std::unique_ptr<InputMatrix> & input) {
    for(int i = 0; i < INPUT_DIM; i++) {
        // std::cout << "row " << i << std::endl;
        fill_row(str,input->data() + i * INPUT_DIM,INPUT_DIM);
    }
    std::string line;
    std::getline(str,line);
}

std::unique_ptr<Cnn> read_cnn(std::string & file_path) {
    auto cnn = std::make_unique<Cnn>();
    cnn->conv_layer = std::make_unique<ConvLayer>();
    cnn->output_layer = std::make_unique<OutputLayer>();
    std::ifstream data(file_path);

    for(int i = 0; i < CONV_LAYER_SIZE; i++) {
        // std::cout << "row " << i << std::endl;
        fill_row(data,cnn->conv_layer->data() + i*FILTER_DIM*FILTER_DIM,FILTER_DIM*FILTER_DIM);
    }

    std::string line;
    std::getline(data,line);
    for(int i = 0; i < OUT_LAYER_SIZE; i++) {
        fill_row(data,cnn->output_layer->data() + i*OUT_NEURON_DIM,OUT_NEURON_DIM);
    }

    return cnn;
}

std::unique_ptr<OutputArray> cpu_compute(std::unique_ptr<InputMatrix> & input, std::unique_ptr<Cnn> & cnn) {
    // input -> 100x100
    // filter -> 10x 5x5
    // conv_out -> 20x20 x10
    auto conv_out = std::make_unique<ConvOutput>();
    for(int i = 0; i < CONV_LAYER_SIZE; i++) {
        for(int x = 0; x < INPUT_DIM; x+=FILTER_DIM) {
            for(int y = 0; y < INPUT_DIM; y+=FILTER_DIM) {
                // dot product of filter to 5x5 input region
                double res = 0;
                for(int d1 = 0; d1 < FILTER_DIM; d1++) {
                    for(int d2 = 0; d2 < FILTER_DIM; d2++) {
                        res += input->at((x+d1)*INPUT_DIM + (y+d2)) * cnn->conv_layer->at(i*FILTER_DIM*FILTER_DIM + d1*FILTER_DIM + d2);
                    }
                }
                (*conv_out)[i*CONV_OUT_DIM*CONV_OUT_DIM +  (x/FILTER_DIM)*CONV_OUT_DIM + (y/FILTER_DIM)] = res;
            }
        }
    }

    // ReLU
    for(auto &e : *conv_out) {
        if(e < 0) {
            e = 0;
        }
    }
    
    // dot conv_out with output layer of each neuron in the cnn
    // conv_out: 20x20x10 dot cnn_out_neuron: 4000x (10 neurons/10 times)
    auto out = std::make_unique<OutputArray>();
    for(int i = 0; i < OUT_LAYER_SIZE; i++) {
        double res = 0;
        for(int j = 0; j < OUT_NEURON_DIM; j++) {
            res += conv_out->at(j) * cnn->output_layer->at(i*OUT_NEURON_DIM + j);
        }
        (*out)[i] = res;
    }
    return out;
}

DeviceContext cuda_init() {
    DeviceContext ctx;
    cudaMalloc(&ctx.dinput, ctx.dinput_bytes);
    cudaMalloc(&ctx.dconv_layer, ctx.dconv_layer_bytes);
    cudaMalloc(&ctx.dout_conv, ctx.dout_conv_bytes);
    cudaMalloc(&ctx.dout_layer, ctx.dout_layer_bytes);
    cudaMalloc(&ctx.dout_array, ctx.dout_array_bytes);
    return ctx;
}

std::unique_ptr<OutputArray> cuda_compute(std::unique_ptr<InputMatrix> & input, std::unique_ptr<Cnn> & cnn, DeviceContext ctx) {    
    cudaMemcpy(ctx.dinput, input->data(), ctx.dinput_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.dconv_layer, cnn->conv_layer->data(), ctx.dconv_layer_bytes, cudaMemcpyHostToDevice);
    compute_cnn_part1<<<10, 512, 0, 0>>>((double (*)[100])ctx.dinput, (double (*)[5][5]) ctx.dconv_layer, ctx.dout_conv);
    cudaDeviceSynchronize();

    auto out = std::make_unique<OutputArray>();
    cudaMemcpy(ctx.dout_layer, cnn->output_layer->data(), ctx.dout_layer_bytes, cudaMemcpyHostToDevice);
    compute_cnn_part2<<<10, 1024, 0, 0>>>(ctx.dout_conv, (double (*)[4000])ctx.dout_layer, ctx.dout_array);
    cudaMemcpy(out->data(), ctx.dout_array, ctx.dout_array_bytes, cudaMemcpyDeviceToHost);
    return out;
}

void cuda_cleanup(DeviceContext ctx) {
    cudaFree(ctx.dinput);
    cudaFree(ctx.dconv_layer);
    cudaFree(ctx.dout_conv);
    cudaFree(ctx.dout_layer);
    cudaFree(ctx.dout_array);
}

bool verify_results(std::unique_ptr<OutputArray> & cpu, std::unique_ptr<OutputArray> & cuda) {
    double epsilon = 1e-9;
    for(int i = 0; i < OUT_LAYER_SIZE;i++) {
        if(!(std::abs((*cpu)[i] - (*cuda)[i]) < epsilon)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cout << "Usage: cnn_file input_file" << std::endl;
        return 1;
    }
    std::string cnn_file = argv[1];
    std::string input_file = argv[2];
    std::unique_ptr<Cnn> cnn = read_cnn(cnn_file);
    
    std::ifstream input_stream(input_file);
    std::string line;
    std::getline(input_stream,line);
    size_t input_size = std::stol(line);
    std::cout << "Running comparision on " << input_size << " inputs..." << std::endl;
    unsigned long cpu_elapsed = 0;
    unsigned long cuda_elapsed = 0;


    DeviceContext ctx = cuda_init();
    for(int i = 0; i < input_size;i++) {
        std::unique_ptr<InputMatrix> input = std::make_unique<InputMatrix>();
        read_next_input(input_stream, input);

        auto cpu_start = std::chrono::steady_clock::now();
        std::unique_ptr<OutputArray> out_cpu = cpu_compute(input, cnn);
        auto cpu_end = std::chrono::steady_clock::now();
        cpu_elapsed += std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();

        auto cuda_start = std::chrono::steady_clock::now();
        std::unique_ptr<OutputArray> out_cuda = cuda_compute(input ,cnn, ctx);
        auto cuda_end = std::chrono::steady_clock::now();
        cuda_elapsed += std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_start).count();
        if(!verify_results(out_cpu, out_cuda)) {
            std::cout << "CPU and GPU implementation results differ." << std::endl;
            return 1;
        }
    }
    cuda_cleanup(ctx);
    
    std::cout << "CPU time elapsed: " << cpu_elapsed << "us" << std::endl;
    std::cout << "CUDA time elapsed: " << cuda_elapsed << "us" << std::endl;
    return 0;
}