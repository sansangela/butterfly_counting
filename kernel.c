#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <x86intrin.h>
#include "graph_reader_csr.c"

#define MAX_FREQ 3.2
#define BASE_FREQ 1.2
#define NUM_INST 1

unsigned long long st;
unsigned long long et;
unsigned long long sum = 0;
unsigned long long num_ops = 0;

int *A_0_values = NULL;
int *A_0_columns = NULL;
int *A_0_row_ptr = NULL;

long long butterfly_count = 0;
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}


#define kernel1(A_0_columns, a1_broadcast, r0, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  flag = !_mm256_testz_si256(r0, r0);

#define kernel2(A_0_columns, a1_broadcast, r0, r1, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast); \
  flag = (!_mm256_testz_si256(r0, r0)) | \
        ((!_mm256_testz_si256(r1, r1)) << 1);

#define kernel4(A_0_columns, a1_broadcast, r0, r1, r2, r3, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r2 = _mm256_loadu_si256(&A_0_columns[j+16]); \
  r3 = _mm256_loadu_si256(&A_0_columns[j+24]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast); \
  r2 = _mm256_cmpeq_epi32(r2, a1_broadcast); \
  r3 = _mm256_cmpeq_epi32(r3, a1_broadcast); \
  flag = (!_mm256_testz_si256(r0, r0)) | \
        ((!_mm256_testz_si256(r1, r1)) << 1) | \
        ((!_mm256_testz_si256(r2, r2)) << 2) | \
        ((!_mm256_testz_si256(r3, r3)) << 3);

#define kernel4(A_0_columns, a1_broadcast, r0, r1, r2, r3, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r2 = _mm256_loadu_si256(&A_0_columns[j+16]); \
  r3 = _mm256_loadu_si256(&A_0_columns[j+24]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast); \
  r2 = _mm256_cmpeq_epi32(r2, a1_broadcast); \
  r3 = _mm256_cmpeq_epi32(r3, a1_broadcast); \
  flag = (!_mm256_testz_si256(r0, r0)) | \
        ((!_mm256_testz_si256(r1, r1)) << 1) | \
        ((!_mm256_testz_si256(r2, r2)) << 2) | \
        ((!_mm256_testz_si256(r3, r3)) << 3);

#define kernel8(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r2 = _mm256_loadu_si256(&A_0_columns[j+16]); \
  r3 = _mm256_loadu_si256(&A_0_columns[j+24]); \
  r4 = _mm256_loadu_si256(&A_0_columns[j+32]); \
  r5 = _mm256_loadu_si256(&A_0_columns[j+40]); \
  r6 = _mm256_loadu_si256(&A_0_columns[j+48]); \
  r7 = _mm256_loadu_si256(&A_0_columns[j+56]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast); \
  r2 = _mm256_cmpeq_epi32(r2, a1_broadcast); \
  r3 = _mm256_cmpeq_epi32(r3, a1_broadcast); \
  r4 = _mm256_cmpeq_epi32(r4, a1_broadcast); \
  r5 = _mm256_cmpeq_epi32(r5, a1_broadcast); \
  r6 = _mm256_cmpeq_epi32(r6, a1_broadcast); \
  r7 = _mm256_cmpeq_epi32(r7, a1_broadcast); \
  flag = (!_mm256_testz_si256(r0, r0)) | \
          ((!_mm256_testz_si256(r1, r1)) << 1) | \
          ((!_mm256_testz_si256(r2, r2)) << 2) | \
          ((!_mm256_testz_si256(r3, r3)) << 3) | \
          ((!_mm256_testz_si256(r4, r4)) << 4) | \
          ((!_mm256_testz_si256(r5, r5)) << 5) | \
          ((!_mm256_testz_si256(r6, r6)) << 6) | \
          ((!_mm256_testz_si256(r7, r7)) << 7);

#define kernel15(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, \
                   r8, r9, r10, r11, r12, r13, r14, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r2 = _mm256_loadu_si256(&A_0_columns[j+16]); \
  r3 = _mm256_loadu_si256(&A_0_columns[j+24]); \
  r4 = _mm256_loadu_si256(&A_0_columns[j+32]); \
  r5 = _mm256_loadu_si256(&A_0_columns[j+40]); \
  r6 = _mm256_loadu_si256(&A_0_columns[j+48]); \
  r7 = _mm256_loadu_si256(&A_0_columns[j+56]); \
  r8 = _mm256_loadu_si256(&A_0_columns[j+64]); \
  r9 = _mm256_loadu_si256(&A_0_columns[j+72]); \
  r10 = _mm256_loadu_si256(&A_0_columns[j+80]); \
  r11 = _mm256_loadu_si256(&A_0_columns[j+88]); \
  r12 = _mm256_loadu_si256(&A_0_columns[j+96]); \
  r13 = _mm256_loadu_si256(&A_0_columns[j+104]); \
  r14 = _mm256_loadu_si256(&A_0_columns[j+112]); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast); \
  r2 = _mm256_cmpeq_epi32(r2, a1_broadcast); \
  r3 = _mm256_cmpeq_epi32(r3, a1_broadcast); \
  r4 = _mm256_cmpeq_epi32(r4, a1_broadcast); \
  r5 = _mm256_cmpeq_epi32(r5, a1_broadcast); \
  r6 = _mm256_cmpeq_epi32(r6, a1_broadcast); \
  r7 = _mm256_cmpeq_epi32(r7, a1_broadcast); \
  r8 = _mm256_cmpeq_epi32(r8, a1_broadcast); \
  r9 = _mm256_cmpeq_epi32(r9, a1_broadcast); \
  r10 = _mm256_cmpeq_epi32(r10, a1_broadcast); \
  r11 = _mm256_cmpeq_epi32(r11, a1_broadcast); \
  r12 = _mm256_cmpeq_epi32(r12, a1_broadcast); \
  r13 = _mm256_cmpeq_epi32(r13, a1_broadcast); \
  r14 = _mm256_cmpeq_epi32(r14, a1_broadcast); \
  flag = (!_mm256_testz_si256(r0, r0)) | \
          ((!_mm256_testz_si256(r1, r1)) << 1) | \
          ((!_mm256_testz_si256(r2, r2)) << 2) | \
          ((!_mm256_testz_si256(r3, r3)) << 3) | \
          ((!_mm256_testz_si256(r4, r4)) << 4) | \
          ((!_mm256_testz_si256(r5, r5)) << 5) | \
          ((!_mm256_testz_si256(r6, r6)) << 6) | \
          ((!_mm256_testz_si256(r7, r7)) << 7) | \
          ((!_mm256_testz_si256(r8, r8)) << 8) | \
          ((!_mm256_testz_si256(r9, r9)) << 9) | \
          ((!_mm256_testz_si256(r10, r10)) << 10) | \
          ((!_mm256_testz_si256(r11, r11)) << 11) | \
          ((!_mm256_testz_si256(r12, r12)) << 12) | \
          ((!_mm256_testz_si256(r13, r13)) << 13) | \
          ((!_mm256_testz_si256(r14, r14)) << 14)
  

#define kernel4_dual(A_0_columns, a1_broadcast0, a1_broadcast1, r0, r1, r2, r3, r0_a0, r1_a0, r2_a0, r3_a0, j, flag) \
  r0 = _mm256_loadu_si256(&A_0_columns[j]); \
  r1 = _mm256_loadu_si256(&A_0_columns[j+8]); \
  r2 = _mm256_loadu_si256(&A_0_columns[j+16]); \
  r3 = _mm256_loadu_si256(&A_0_columns[j+24]); \
  r0_a0 = _mm256_cmpeq_epi32(r0, a1_broadcast0); \
  r1_a0 = _mm256_cmpeq_epi32(r1, a1_broadcast0); \
  r2_a0 = _mm256_cmpeq_epi32(r2, a1_broadcast0); \
  r3_a0 = _mm256_cmpeq_epi32(r3, a1_broadcast0); \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast1); \
  r1 = _mm256_cmpeq_epi32(r1, a1_broadcast1); \
  r2 = _mm256_cmpeq_epi32(r2, a1_broadcast1); \
  r3 = _mm256_cmpeq_epi32(r3, a1_broadcast1); \
  flag = (!_mm256_testz_si256(r0_a0, r0_a0)) | \
          ((!_mm256_testz_si256(r1_a0, r1_a0)) << 1) | \
          ((!_mm256_testz_si256(r2_a0, r2_a0)) << 2) | \
          ((!_mm256_testz_si256(r3_a0, r3_a0)) << 3) | \
          ((!_mm256_testz_si256(r0, r0)) << 4) | \
          ((!_mm256_testz_si256(r1, r1)) << 5) | \
          ((!_mm256_testz_si256(r2, r2)) << 6) | \
          ((!_mm256_testz_si256(r3, r3)) << 7);


void matrix_multiply_simd(const int A_0_columns[], const int A_0_row_ptr[],
                          int a_1_columns_start, int num_rows_A_0,
                          int num_cols_A_0, int num_cols_a_1) {

  __m256i a1_broadcast;
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;
  // append 8 zeros in case of our of bound access when summing up.
  // int *counter = (int *)calloc((num_cols_a_1 + 8), sizeof(int));
  int counter = 0;

  // Load A0[i,:]
  for (int i = 0; i < num_rows_A_0; i++) {
    counter = 0;

    // Load a1
    for (int col_idx = a_1_columns_start;
         col_idx < a_1_columns_start + num_cols_a_1; col_idx++) {

      a1_broadcast = _mm256_set1_epi32(A_0_columns[col_idx]);

      int j = A_0_row_ptr[i]; // A0_start
      int A0_end = A_0_row_ptr[i + 1];
      int flag = 0;

      while (j < A0_end - 8) {
        if (j + 120 < A0_end) {
          kernel15(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7,
                   r8, r9, r10, r11, r12, r13, r14, j, flag);
          j += 120;
        } else if (j + 64 < A0_end) {
          kernel8(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j, flag);
          j += 64;
        } else if (j + 32 < A0_end) {
          kernel4_dual(A_0_columns, a1_broadcast, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j, flag);
          j += 32;
        } else if (j + 16 < A0_end) {
          kernel2(A_0_columns, a1_broadcast, r0, r1, j, flag);
          j += 16;
        } else {
          kernel1(A_0_columns, a1_broadcast, r0, j, flag);
          j += 8;
        }
        if (flag) {
          counter++;
          break;
        }
      }
    }
    
    butterfly_count += counter * counter - counter;
  }
}



/* Data Preprocessing */
// omit values as not relevant to the kenel
void pad_csr(uint64_t *columns, uint64_t *row_ptr, int num_rows) {
  // Calculate new length for values and columns
  int new_length = row_ptr[num_rows] + 8 * num_rows;

  // Allocate memory for columns arrays
  A_0_columns = (int *)malloc(new_length * sizeof(int));
  A_0_row_ptr = (int *)malloc((row_ptr)[num_rows] * sizeof(int));

  // For each row
  for (int i = 0; i < num_rows; i++) {
    // Copy existing columns
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      A_0_columns[j + 8 * i] = columns[j];
    }
    // Add eight -1 columns
    for (int j = 0; j < 8; j++) {
      A_0_columns[row_ptr[i + 1] + 8 * i + j] = -1;
    }
  }

  A_0_row_ptr[0] = row_ptr[0];
  // Update row_ptr
  for (int i = 1; i <= num_rows; i++) {
    A_0_row_ptr[i] = row_ptr[i] + 8 * i;
  }
}


void matrix_multiply_scalar(const int A_0_columns[], const int A_0_row_ptr[],
                          int a_1_columns_start, int num_rows_A_0,
                          int num_cols_A_0, int num_cols_a_1) {

  int counter = 0;
  
  // Load A0[i,:]
  for (int i = 0; i < num_rows_A_0; i++) {
    counter = 0;
    for (int col_idx = a_1_columns_start;
         col_idx < a_1_columns_start + num_cols_a_1; col_idx++) {

      // Scalar Baseline: Linear Search
      for (int j = A_0_row_ptr[i]; j < A_0_row_ptr[i + 1]; j++) {
        num_ops++;   // comparison and increment 2 + for loop 2
        if (A_0_columns[j] == A_0_columns[col_idx]) {
          num_ops++;
          counter++;
          break;
        }
      }

      // num_ops+=2; // for loop 2
    }
    butterfly_count += counter * counter - counter;
    num_ops += 3; // multiplication, subtraction, addition & assignment 3 + for loop 2
  }
}

void matrix_multiply_scalar_two_pointer(const int A_0_columns[], const int A_0_row_ptr[],
                          int a_1_columns_start, int num_rows_A_0,
                          int num_cols_A_0, int num_cols_a_1) {

  int counter = 0;
  
  // Load A0[i,:]
  for (int i = 0; i < num_rows_A_0; i++) {
    counter = 0;
    int A_0_start_idx = A_0_row_ptr[i];
    int a_1_start_idx = a_1_columns_start;
    
    while ((A_0_start_idx < A_0_row_ptr[i + 1]) && (a_1_start_idx < a_1_columns_start + num_cols_a_1)) {
      if (A_0_columns[A_0_start_idx] == A_0_columns[a_1_start_idx]) {
        counter++;
        A_0_start_idx++;
        a_1_start_idx++;
        num_ops += 6; // 4 + while 2
      } else if (A_0_columns[A_0_start_idx] < A_0_columns[a_1_start_idx]) {
        A_0_start_idx++;
        num_ops += 4; // 2 + while 2
      } else {
        a_1_start_idx++;
        num_ops += 4; // 2 + while 2
      }
    }

    butterfly_count += (counter * counter - counter);
    num_ops += 5; // multiplication, subtraction, addition & assignment 3 + for loop 2
  }
}


int main(int argc, char **argv) {
  printf("Testing kernel\n");

  // Read data from txt file
  // IA: row_ptr, JA: col_idx
  uint64_t A_0_columns_origin[MAX_EDGES], A_0_row_ptr_origin[MAX_EDGES];
  uint64_t node_count = read_edge_list_CSR("/afs/andrew.cmu.edu/usr10/xinyuc2/private/18645/project/butterfly/data/opsahl-collaboration/out.opsahl-collaboration", A_0_row_ptr_origin, A_0_columns_origin);
  // uint64_t node_count = read_edge_list_CSR("/afs/andrew.cmu.edu/usr10/xinyuc2/private/18645/project/butterfly/data/test/input.txt", A_0_row_ptr_origin, A_0_columns_origin);
  int num_rows_A_0 = 16726;
  int num_cols_A_0 = node_count;
  printf("Node count = %d\n", node_count);
  printf("\n");

  // // Example data in CSR format
  // int A_0_columns_origin[] = {0, 1, 3, 4, 5, 7, 8, 9, 1, 2, 6, 7, 8, 0,
  //                             1, 2, 4, 8, 9, 1, 3, 4, 7, 1, 2, 4, 9, 3,
  //                             4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 0, 1,
  //                             2, 3, 5, 8, 9, 0, 1, 3, 7, 1, 6, 7, 8, 9};
  // int A_0_row_ptr_origin[] = {0,  8,  13, 19, 23, 27,
  //                             34, 40, 47, 51, 56}; // CSR row_ptr
  // int num_rows_A_0 = 10;
  // int num_cols_A_0 = 10;
  // int node_count = 10;


  // int runs = atoi(argv[1]);
  int runs = 1;

  num_ops = 0;
  pad_csr(A_0_columns_origin, A_0_row_ptr_origin, num_rows_A_0);

  for (int run_id = 0; run_id < runs; run_id++) {
    butterfly_count = 0;
    for (int a_1_row = 1; a_1_row < num_rows_A_0; ++a_1_row) {
      int num_cols_a_1 = A_0_row_ptr[a_1_row + 1] - A_0_row_ptr[a_1_row] - 8;
      int a_1_columns_start = A_0_row_ptr[a_1_row];
      printf("row: %d\n", a_1_row);
      
      // SIMD test
      st = rdtsc();
      matrix_multiply_simd(A_0_columns, A_0_row_ptr, a_1_columns_start, a_1_row,
                           num_cols_A_0, num_cols_a_1);
      et = rdtsc();
      sum += (et - st);
      
      // // Scalar test
      // st = rdtsc();
      // matrix_multiply_scalar(A_0_columns, A_0_row_ptr, a_1_columns_start, a_1_row,
      //                      num_cols_A_0, num_cols_a_1);
      // // matrix_multiply_scalar_two_pointer(A_0_columns, A_0_row_ptr, a_1_columns_start, a_1_row,
      // //                      num_cols_A_0, num_cols_a_1);
      // et = rdtsc();
      // sum += (et - st);
    }
    butterfly_count /= 2;
    printf("butterfly_count: %lld\n", butterfly_count);
  }

  // num_ops = 2162;  // needed for SIMD
  num_ops=4673066757;
  // num_ops /= runs;   // needed for scalar
  printf("num_ops=%llu\n", num_ops);
  printf("RDTSC Base Cycles Taken: %llu\n\r", sum);
  printf("Latency: %lf\n\r", ((MAX_FREQ/BASE_FREQ) * sum) / (num_ops * runs));
  printf("Throughput: %lf\n", (num_ops*runs)/((double)sum*MAX_FREQ/BASE_FREQ));

  return 0;
}