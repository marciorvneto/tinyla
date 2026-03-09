#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

static void print_section(const char *title) {
  printf("\n=== %s ===\n", title);
}

static void build_example_system(tla_Arena *arena, tla_Matrix **A_out,
                                 tla_Vector **b_out) {
  tla_Matrix *A = tla_matrix_eye(arena, 5);
  tla_Vector *b = tla_vector_of_value(arena, 5, 7.0);

  tla_vector_set_value(b, 1, -2.0);

  tla_swap_rows(A, 2, 4);
  tla_matrix_set_value(A, 2, 0, 5.0);
  tla_matrix_set_value(A, 0, 3, -2.0);
  tla_matrix_set_value(A, 2, 2, 3.0);

  *A_out = A;
  *b_out = b;
}

int main(void) {
  tla_Arena arena = tla_arena_create(1024 * 1024);

  tla_Matrix *A;
  tla_Vector *b;
  build_example_system(&arena, &A, &b);

  print_section("System matrix A");
  tla_print_matrix(A);

  print_section("Right-hand side b");
  tla_print_vector(b);

  tla_Matrix *Ab = tla_matrix_append_column(&arena, A, b);
  print_section("Augmented matrix [A | b]");
  tla_print_matrix(Ab);

  tla_Vector *x = tla_vector_of_value(&arena, A->rows, 0.0);
  tla_gauss_solve(x, Ab);

  print_section("Solution x from Gaussian elimination");
  tla_print_vector(x);

  tla_Vector *Ax = tla_vector_of_shape(&arena, x, 0.0);
  tla_matrix_vector_mul(Ax, A, x);

  print_section("Check: A*x");
  tla_print_vector(Ax);

  tla_vector_sub(Ax, Ax, b);
  print_section("Residual: A*x - b");
  tla_print_vector(Ax);

  tla_arena_destroy(&arena);
  return 0;
}
