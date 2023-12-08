#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define MAX_EDGES 60000 // Adjust this based on expected input size

typedef struct {
    uint64_t first;
    uint64_t second;
} Pair;

int pairCompare(const void *a, const void *b) {
    Pair *pair1 = (Pair *)a;
    Pair *pair2 = (Pair *)b;

    if (pair1->first < pair2->first) return -1;
    if (pair1->first > pair2->first) return 1;
    if (pair1->first == pair2->first) {
        if (pair1->second < pair2->second) return -1;
        if (pair1->second > pair2->second) return 1;
    }
    return 0;
}

uint64_t read_edge_list_CSR(const char *pathname, uint64_t *IA, uint64_t *JA) {
    FILE *infile = fopen(pathname, "r");
    if (infile == NULL) {
        perror("Error opening file");
        return 0;
    }

    uint64_t max_id = 0;
    uint64_t src, dst;
    Pair edges[MAX_EDGES];
    uint64_t edge_count = 0;

    while (fscanf(infile, "%llu %llu", &src, &dst) == 2) {
        max_id = (src > max_id) ? src : max_id;
        max_id = (dst > max_id) ? dst : max_id;

        edges[edge_count].first = src;
        edges[edge_count].second = dst;
        edge_count++;
    }

    fclose(infile);

    // Sort the edges
    qsort(edges, edge_count, sizeof(Pair), pairCompare);

    // Construct CSR
    uint64_t node_count = max_id + 1;
    memset(IA, 0, sizeof(uint64_t) * (node_count + 1));
    uint64_t ja_index = 0;

    for (uint64_t i = 0; i < edge_count; ++i) {
        while (edges[i].first >= ja_index) {
            IA[ja_index] = i;
            ja_index++;
        }
        JA[i] = edges[i].second;
    }
    IA[ja_index] = edge_count;

    return node_count;
}