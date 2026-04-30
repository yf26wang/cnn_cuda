#include <array>

// The CNN consists of 100x100 input matrix, a convolution layer of 10 5x5 filter matrices, a RELU
// layer, and an output layer of 10 4000x1 weight vectors. CNN output is a 10x1 vector.
#define INPUT_DIM 100
#define FILTER_DIM 5 // should be factor of INPUT_DIM
#define CONV_OUT_DIM INPUT_DIM / FILTER_DIM
#define CONV_LAYER_SIZE 10
#define OUT_NEURON_DIM CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE
#define OUT_LAYER_SIZE 10

typedef std::array<double, INPUT_DIM * INPUT_DIM> InputMatrix;
typedef std::array<double, FILTER_DIM * FILTER_DIM * CONV_LAYER_SIZE> ConvLayer;
typedef std::array<double, CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE> ConvOutput;
typedef std::array<double, OUT_NEURON_DIM * OUT_LAYER_SIZE> OutputLayer;
typedef std::array<double, OUT_LAYER_SIZE> OutputArray;

struct Cnn {
    std::unique_ptr<ConvLayer>conv_layer;
    std::unique_ptr<OutputLayer> output_layer;
};

struct DeviceContext {
    double * dinput;
    size_t dinput_bytes = sizeof(double) * INPUT_DIM*INPUT_DIM;
    double * dconv_layer;
    size_t dconv_layer_bytes = sizeof(double) * FILTER_DIM * FILTER_DIM * CONV_LAYER_SIZE;
    double * dout_conv;
    size_t dout_conv_bytes = sizeof(double) * OUT_NEURON_DIM;
    double * dout_layer;
    size_t dout_layer_bytes = sizeof(double) * OUT_NEURON_DIM * OUT_LAYER_SIZE;
    double * dout_array;
    size_t dout_array_bytes = sizeof(double) * OUT_LAYER_SIZE;
};