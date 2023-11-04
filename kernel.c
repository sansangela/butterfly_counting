#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <immintrin.h>

#define MAX_FREQ 3.2
#define BASE_FREQ 1.2
#define NUM_INST 1

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}



#ifdef KERNEL_C
void mm_kernel() {
    printf("Calling kernel\n");
}
#endif


void matrix_multiply_simd(const int A_0_values[], const int A_0_columns[],
                          const int A_0_row_ptr[], const int a_1_values[],
                          const int a_1_columns[], const int a_1_row_ptr[],
                          int num_rows_A_0, int num_cols_A_0, int num_cols_a_1) {

    __m256i a1_broadcast, A0, mask;
    int *counter = (int *)malloc(sizeof(int) * num_rows_A_0);
    for (int i = 0; i < num_rows_A_0; i++) {
        counter[i] = 0;
    }

    /* Debug Variables */
    // int out1[] = {0, 0, 0, 0, 0, 0, 0, 0};
    // int out2[] = {0, 0, 0, 0, 0, 0, 0, 0};
    // int out3[] = {0, 0, 0, 0, 0, 0, 0, 0};
    

    // Load a1
    for (int col_idx = 0; col_idx < num_cols_a_1; col_idx++) {
    // for (int col_idx = 1; col_idx < num_cols_a_1; col_idx++) {  <- if 1-indexed
        a1_broadcast = _mm256_set1_epi32(a_1_columns[col_idx]);


        // Load A0[i,:]
        for (int i = 0; i < num_rows_A_0; i++){
            int j = A_0_row_ptr[i];     // A0_start
            int A0_end = A_0_row_ptr[i+1];

            for (j; j < A0_end; j+=8) {
                if (j < A0_end - 8) {
                    A0 = _mm256_loadu_si256(&A_0_columns[j]);
                    // TO DO
                    printf("TO BE TESTED\n");
                } else {
                    // Load rest of A0[i:,]
                    int j_offset = j;
                    int zeros[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                    for (j; j < A0_end; j++) {
                        zeros[j-j_offset] = A_0_columns[j];
                    }
                    A0 = _mm256_loadu_si256(&zeros[0]);

                    mask = _mm256_cmpeq_epi32(A0, a1_broadcast);
                    
                    /* Debug Variables */
                    // _mm256_storeu_si256(&out1[0],A0);
                    // _mm256_storeu_si256(&out2[0],a1_broadcast);
                    // _mm256_storeu_si256(&out3[0],mask);
                }

                int has_minus_one = !_mm256_testz_si256(mask, mask);
                if (has_minus_one != 0) {
                    counter[i]++;
                    break;
                }
                
            }

            /* Debug Prints */
            // printf("\nA0:\n");
            // for (int print_i = 0; print_i < 8; print_i++) {
            //     printf("%d ", out1[print_i]);
            // }
            // printf("\na1:\n");
            // for (int print_i = 0; print_i < 8; print_i++) {
            //     printf("%d ", out2[print_i]);
            // }
            // printf("\nmask:\n");
            // for (int print_i = 0; print_i < 8; print_i++) {
            //     printf("%d ", out3[print_i]);
            // }
            
        }

    }

    printf("\nCounter:\n");
    for (int print_i = 0; print_i < num_rows_A_0; print_i++) {
        printf("%d ", counter[print_i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    printf("Testing kernel\n");

    // int runs = atoi(argv[1]);
    int runs = 1;
    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;
    
    // Example data in CSR format
    int A_0_values[] = {1, 1, 1, 1, 1, 1};
    int A_0_columns[] = {1, 0, 2, 1, 3, 2, 3};
    int A_0_row_ptr[] = {0, 1, 3, 5, 7}; // CSR row_ptr
    int num_rows_A_0 = 4;
    int num_cols_A_0 = 4;
    int num_cols_a_1 = 2;

    int a_1_values[] = {1, 1};
    int a_1_columns[] = {0, 2};
    int a_1_row_ptr[] = {0, 2}; // CSR row_ptr

    for (int run_id = 0; run_id < runs; run_id++) {
        st = rdtsc();
        matrix_multiply_simd(A_0_values, A_0_columns, A_0_row_ptr, a_1_values, a_1_columns, a_1_row_ptr, num_rows_A_0, num_cols_A_0, num_cols_a_1);
        et = rdtsc();
        sum += (et-st);
    }
    
    printf("RDTSC Base Cycles Taken: %llu\n\r",sum);
    // printf("Latency: %lf\n\r", ((MAX_FREQ/BASE_FREQ) * sum) / (NUM_INST * runs));

    return 0;
}