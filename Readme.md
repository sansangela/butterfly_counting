# 18645 Final Project: Speed Up Butterfly Counting Algorithms for Bipartite Graphs

## Project Overview

This project introduces an innovative approach to accelerate butterfly counting algorithms in bipartite graphs. By optimizing matrix-vector multiplication in Compressed Sparse Row (CSR) format and employing various parallelism techniques, our method significantly improves computational efficiency.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- You should have access to the datasets located in the `/data/` directory.

### Dataset Conversion

We convert `out.dbpedia-<dataset>` into CSR format using `graph_reader_csr.c`. There is a MAX_EDGES variable need to be set before adding new dataset. The default value is 251000 and should be valid for exisiting four datasets.

### Running the Sequential Kernel

To run our kernel sequentially, follow these instructions:

1. Uncomment the block corresponding to the target dataset in `kernel.c`.
2. Update the path string to your dataset.
3. Run the following commands:

```bash
make clean && make kernel && make run
```

### Running the OpenMP Parallelized Kernel

To run our parallelized kernel, follow these instructions:

1. Uncomment the block corresponding to the target dataset in `kernel_parallel.c`.
2. Update the path string to your dataset.
3. Run the following commands:

```bash
make clean && make kernel_parallel && make run_parallel
