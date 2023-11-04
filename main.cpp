#include <iostream>
#include <vector>
#include "graph_reader_csr.cpp"

#define KERNEL_C 
#include "kernel.c"

int main() {
    std::vector<uint64_t> row_indices;
    std::vector<uint64_t> col_indices;
    std::vector<uint64_t> IA;
    std::vector<uint64_t> JA;

    std::string inputFilePath = "input.txt";

    uint64_t maxId = read_edge_list_CSR(inputFilePath, row_indices, col_indices, IA, JA);

    // Print the results or perform any other testing as needed
    std::cout << "Maximum ID: " << maxId << std::endl;

    std::cout << "Row Indices:" << std::endl;
    for (const auto& row : row_indices) {
        std::cout << row << " ";
    }
    std::cout << std::endl;

    std::cout << "Column Indices:" << std::endl;
    for (const auto& col : col_indices) {
        std::cout << col << " ";
    }
    std::cout << std::endl;

    std::cout << "IA Vector:" << std::endl;
    for (const auto& ia : IA) {
        std::cout << ia << " ";
    }
    std::cout << std::endl;

    std::cout << "JA Vector:" << std::endl;
    for (const auto& ja : JA) {
        std::cout << ja << " ";
    }
    std::cout << std::endl;

    // mm_kernel();

    return 0;
}
