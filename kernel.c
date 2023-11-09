#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

#define MAX_FREQ 3.2
#define BASE_FREQ 1.2
#define NUM_INST 1

int *A_0_values = NULL;
int *A_0_columns = NULL;
int *A_0_row_ptr = NULL;

long long butterfly_count = 0;
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

#ifdef KERNEL_C
void mm_kernel() { printf("Calling kernel\n"); }
#endif

#define kernel_add2(r0, r1) r0 = _mm256_add_epi32(r0, r1);

#define kernel_add4(r0, r1, r2, r3)                                            \
  kernel_add2(r0, r1) kernel_add2(r2, r3) r0 = _mm256_add_epi32(r0, r2);

#define kernel_add8(r0, r1, r2, r3, r4, r5, r6, r7)                            \
  kernel_add4(r0, r1, r2, r3) kernel_add4(r4, r5, r6, r7) r0 =                 \
      _mm256_add_epi32(r0, r4);

#define kernel_add16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12,    \
                     r13, r14, r15)                                            \
  kernel_add8(r0, r1, r2, r3, r4, r5, r6, r7)                                  \
      kernel_add8(r8, r9, r10, r11, r12, r13, r14, r15) r0 =                   \
          _mm256_add_epi32(r0, r8);

#define kernel1(A_0_columns, a1_broadcast, r0, j, flag)                        \
  r0 = _mm256_cmpeq_epi32(r0, a1_broadcast);                                   \
  flag |= !_mm256_testz_si256(r0, r0);

#define kernel2(A_0_columns, a1_broadcast, r0, r1, j, flag)                    \
  kernel1(A_0_columns, a1_broadcast, r0, j, flag)                              \
      kernel1(A_0_columns, a1_broadcast, r1, j, flag)

#define kernel4(A_0_columns, a1_broadcast, r0, r1, r2, r3, j, flag)            \
  kernel2(A_0_columns, a1_broadcast, r0, r1, j, flag)                          \
      kernel2(A_0_columns, a1_broadcast, r2, r3, j, flag)

#define kernel8(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j,  \
                flag)                                                          \
  kernel4(A_0_columns, a1_broadcast, r0, r1, r2, r3, j, flag)                  \
      kernel4(A_0_columns, a1_broadcast, r4, r5, r6, r7, j, flag)

#define kernel15(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7,    \
                 r8, r9, r10, r11, r12, r13, r14, j, flag)                     \
  kernel8(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j, flag)  \
      kernel4(A_0_columns, a1_broadcast, r8, r9, r10, r11, j, flag)            \
          kernel2(A_0_columns, a1_broadcast, r12, r13, j, flag)                \
              kernel1(A_0_columns, a1_broadcast, r14, j, flag)

int sum_up(int counter[], int size) {
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  int result[8];
  int total_sum = 0;

  int i = 0;
  while (i < size) {
    if (i + 128 <= size) {
      r0 = _mm256_loadu_si256(&counter[i]);
      r1 = _mm256_loadu_si256(&counter[i + 8]);
      r2 = _mm256_loadu_si256(&counter[i + 16]);
      r3 = _mm256_loadu_si256(&counter[i + 24]);
      r4 = _mm256_loadu_si256(&counter[i + 32]);
      r5 = _mm256_loadu_si256(&counter[i + 40]);
      r6 = _mm256_loadu_si256(&counter[i + 48]);
      r7 = _mm256_loadu_si256(&counter[i + 56]);
      r8 = _mm256_loadu_si256(&counter[i + 64]);
      r9 = _mm256_loadu_si256(&counter[i + 72]);
      r10 = _mm256_loadu_si256(&counter[i + 80]);
      r11 = _mm256_loadu_si256(&counter[i + 88]);
      r12 = _mm256_loadu_si256(&counter[i + 96]);
      r13 = _mm256_loadu_si256(&counter[i + 104]);
      r14 = _mm256_loadu_si256(&counter[i + 112]);
      r15 = _mm256_loadu_si256(&counter[i + 120]);
      kernel_add16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
                   r14, r15);
      i += 128;
    } else if (i + 64 <= size) {
      r0 = _mm256_loadu_si256(&counter[i]);
      r1 = _mm256_loadu_si256(&counter[i + 8]);
      r2 = _mm256_loadu_si256(&counter[i + 16]);
      r3 = _mm256_loadu_si256(&counter[i + 24]);
      r4 = _mm256_loadu_si256(&counter[i + 32]);
      r5 = _mm256_loadu_si256(&counter[i + 40]);
      r6 = _mm256_loadu_si256(&counter[i + 48]);
      r7 = _mm256_loadu_si256(&counter[i + 56]);
      kernel_add8(r0, r1, r2, r3, r4, r5, r6, r7);
      i += 64;
    } else if (i + 32 <= size) {
      r0 = _mm256_loadu_si256(&counter[i]);
      r1 = _mm256_loadu_si256(&counter[i + 8]);
      r2 = _mm256_loadu_si256(&counter[i + 16]);
      r3 = _mm256_loadu_si256(&counter[i + 24]);
      kernel_add4(r0, r1, r2, r3);
      i += 32;
    } else if (i + 16 <= size) {
      r0 = _mm256_loadu_si256(&counter[i]);
      r1 = _mm256_loadu_si256(&counter[i + 8]);
      kernel_add2(r0, r1);
      i += 16;
    } else {
      r0 = _mm256_loadu_si256(&counter[i]);
      i += 8;
    }
    _mm256_storeu_si256(result, r0);
    for (int i = 0; i < 8; i++) {
      total_sum += result[i];
    }
  }
  return total_sum;
}

void matrix_multiply_simd(const int A_0_columns[], const int A_0_row_ptr[],
                          int a_1_columns_start, int num_rows_A_0,
                          int num_cols_A_0, int num_cols_a_1) {

  __m256i a1_broadcast;
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;
  // append 8 zeros in case of our of bound access when summing up.
  int *counter = (int *)calloc((num_cols_a_1 + 8), sizeof(int));

  // Load A0[i,:]
  for (int i = 0; i < num_rows_A_0; i++) {
    counter = (int *)calloc((num_cols_a_1 + 8), sizeof(int));
    // Load a1
    for (int col_idx = a_1_columns_start;
         col_idx < a_1_columns_start + num_cols_a_1; col_idx++) {
      a1_broadcast = _mm256_set1_epi32(A_0_columns[col_idx]);

      printf("a_1_element: %d\n", A_0_columns[col_idx]);

      int j = A_0_row_ptr[i]; // A0_start
      int A0_end = A_0_row_ptr[i + 1];
      int flag = 0;
      for (int tmp = 0; tmp < 8; tmp++) {
        printf("%d ", A_0_columns[j + tmp]);
      }
      printf("\n");
      while (j < A0_end - 8) {
        if (j + 120 < A0_end) {
          r0 = _mm256_loadu_si256(&A_0_columns[j]);
          r1 = _mm256_loadu_si256(&A_0_columns[j + 8]);
          r2 = _mm256_loadu_si256(&A_0_columns[j + 16]);
          r3 = _mm256_loadu_si256(&A_0_columns[j + 24]);
          r4 = _mm256_loadu_si256(&A_0_columns[j + 32]);
          r5 = _mm256_loadu_si256(&A_0_columns[j + 40]);
          r6 = _mm256_loadu_si256(&A_0_columns[j + 48]);
          r7 = _mm256_loadu_si256(&A_0_columns[j + 56]);
          r8 = _mm256_loadu_si256(&A_0_columns[j + 64]);
          r9 = _mm256_loadu_si256(&A_0_columns[j + 72]);
          r10 = _mm256_loadu_si256(&A_0_columns[j + 80]);
          r11 = _mm256_loadu_si256(&A_0_columns[j + 88]);
          r12 = _mm256_loadu_si256(&A_0_columns[j + 96]);
          r13 = _mm256_loadu_si256(&A_0_columns[j + 104]);
          r14 = _mm256_loadu_si256(&A_0_columns[j + 112]);
          kernel15(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7,
                   r8, r9, r10, r11, r12, r13, r14, j, flag);
          j += 120;
        } else if (j + 64 < A0_end) {
          r0 = _mm256_loadu_si256(&A_0_columns[j]);
          r1 = _mm256_loadu_si256(&A_0_columns[j + 8]);
          r2 = _mm256_loadu_si256(&A_0_columns[j + 16]);
          r3 = _mm256_loadu_si256(&A_0_columns[j + 24]);
          r4 = _mm256_loadu_si256(&A_0_columns[j + 32]);
          r5 = _mm256_loadu_si256(&A_0_columns[j + 40]);
          r6 = _mm256_loadu_si256(&A_0_columns[j + 48]);
          r7 = _mm256_loadu_si256(&A_0_columns[j + 56]);
          kernel8(A_0_columns, a1_broadcast, r0, r1, r2, r3, r4, r5, r6, r7, j,
                  flag);
          j += 64;
        } else if (j + 32 < A0_end) {
          r0 = _mm256_loadu_si256(&A_0_columns[j]);
          r1 = _mm256_loadu_si256(&A_0_columns[j + 8]);
          r2 = _mm256_loadu_si256(&A_0_columns[j + 16]);
          r3 = _mm256_loadu_si256(&A_0_columns[j + 24]);
          kernel4(A_0_columns, a1_broadcast, r0, r1, r2, r3, j, flag);
          j += 32;
        } else if (j + 16 < A0_end) {
          r0 = _mm256_loadu_si256(&A_0_columns[j]);
          r1 = _mm256_loadu_si256(&A_0_columns[j + 8]);
          kernel2(A_0_columns, a1_broadcast, r0, r1, j, flag);
          j += 16;
        } else {
          r0 = _mm256_loadu_si256(&A_0_columns[j]);
          kernel1(A_0_columns, a1_broadcast, r0, j, flag);
          j += 8;
        }
        if (flag != 0) {
          counter[col_idx - a_1_columns_start]++;
          break;
        }
      }
    }
    int curr_sum = sum_up(counter, num_cols_a_1);
    butterfly_count += curr_sum * curr_sum - curr_sum;
  }
}

// omit values as not relevant to the kenel
void pad_csr(int *columns, int *row_ptr, int num_rows) {
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

  // Update row_ptr
  for (int i = 1; i <= num_rows; i++) {
    A_0_row_ptr[i] += row_ptr[i] + 8 * i;
  }
}

int main(int argc, char **argv) {
  printf("Testing kernel\n");

  // int runs = atoi(argv[1]);
  int runs = 1;
  unsigned long long st;
  unsigned long long et;
  unsigned long long sum = 0;

  // Example data in CSR format
  int A_0_columns_origin[] = {0, 1, 3, 4, 5, 7, 8, 9, 1, 2, 6, 7, 8, 0,
                              1, 2, 4, 8, 9, 1, 3, 4, 7, 1, 2, 4, 9, 3,
                              4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 0, 1,
                              2, 3, 5, 8, 9, 0, 1, 3, 7, 1, 6, 7, 8, 9};
  int A_0_row_ptr_origin[] = {0,  8,  13, 19, 23, 27,
                              34, 40, 47, 51, 56}; // CSR row_ptr
  int num_rows_A_0 = 10;
  int num_cols_A_0 = 10;

  pad_csr(A_0_columns_origin, A_0_row_ptr_origin, num_rows_A_0);

  for (int run_id = 0; run_id < runs; run_id++) {
    butterfly_count = 0;
    for (int a_1_row = 1; a_1_row < num_rows_A_0; ++a_1_row) {
      int num_cols_a_1 = A_0_row_ptr[a_1_row + 1] - A_0_row_ptr[a_1_row] - 8;
      int a_1_columns_start = A_0_row_ptr[a_1_row];
      st = rdtsc();
      matrix_multiply_simd(A_0_columns, A_0_row_ptr, a_1_columns_start, a_1_row,
                           num_cols_A_0, num_cols_a_1);
      et = rdtsc();
      sum += (et - st);
    }
    butterfly_count /= 2;
    printf("butterfly_count: %lld\n", butterfly_count);
  }

  printf("RDTSC Base Cycles Taken: %llu\n\r", sum);
  // printf("Latency: %lf\n\r", ((MAX_FREQ/BASE_FREQ) * sum) / (NUM_INST *
  // runs));

  return 0;
}