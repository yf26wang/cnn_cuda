from random import uniform
from sys import argv

MAX = 10
samples = 200

INPUT_DIM = 100
FILTER_DIM = 5 # should be factor of INPUT_DIM
CONV_OUT_DIM = INPUT_DIM // FILTER_DIM
CONV_LAYER_SIZE = 10
OUT_NEURON_DIM = CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE
OUT_LAYER_SIZE = 10

if len(argv) > 1:
    samples = int(argv[1])
    
inputname = "in.csv"
cnnname = "cnn.csv"

with open("input/"+cnnname, 'w') as f:
    for i in range(10):
        row = [str(uniform(-MAX, MAX)) for n in range(FILTER_DIM*FILTER_DIM)]
        row = ','.join(row)+'\n'
        f.write(row)

    f.write('\n')
    for i in range(10):
        row = [str(uniform(-MAX, MAX)) for n in range(OUT_NEURON_DIM)]
        row = ','.join(row)+'\n'
        f.write(row)

with open("input/"+inputname, 'w') as f:
    f.write(str(samples)+'\n')
    for x in range(samples):
        for i in range(INPUT_DIM):
            row = [str(uniform(-MAX, MAX)) for n in range(INPUT_DIM)]
            row = ','.join(row)+'\n'
            f.write(row)
        f.write('\n')
