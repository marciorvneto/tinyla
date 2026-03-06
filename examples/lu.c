#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main() {
  tla_Arena a = tla_arena_create(1024 * 1024); // 1MB

  tla_Matrix *A = tla_matrix_eye(&a, 5);
  tla_Vector *b = tla_vector_of_value(&a, 5, 7);
  tla_vector_set_value(b, 1, -2);
  tla_swap_rows(A, 2, 4);
  tla_matrix_set_value(A, 2, 0, 5);
  tla_matrix_set_value(A, 0, 3, -2);
  tla_matrix_set_value(A, 2, 2, 3);

  printf("A:\n");
  tla_print_matrix(A);

  tla_Vector *x = tla_vector_of_value(&a, A->rows, 0);
  tla_Matrix *Ab = tla_matrix_append_column(&a, A, b);
  gauss_solve(x, Ab);
  tla_print_vector(x);

  tla_Vector *check = tla_vector_of_value(&a, A->rows, 0);
  tla_matrix_tla_vector_mul(check, A, x);
  tla_print_vector(check);
  tla_vector_sub(check, check, b);
  tla_print_vector(check);

  PLUFactorization factor = plu_factor(&a, A);
  printf("L:\n");
  tla_print_matrix(factor.L);
  printf("U:\n");
  tla_print_matrix(factor.U);

  printf("PA:\n");
  tla_print_matrix(tla_matrix_apply_permutation_new(&a, factor.p, A));
  printf("LU:\n");
  tla_print_matrix(tla_matrix_tla_matrix_mul_new(&a, factor.L, factor.U));

  printf("LU solution:\n");
  tla_Vector *sol = lu_solve(&a, factor, b);
  tla_print_vector(sol);

  tla_arena_destroy(&a);
  return 0;
}
