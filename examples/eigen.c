#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

static void print_section(const char *title) {
  printf("\n=== %s ===\n", title);
}

int main() {
  tla_Arena a = tla_arena_create(1024 * 1024); // 1MB

  print_section("3x3 Matrix (Real Eigenvalues)");
  tla_Matrix *m1 = tla_matrix_create(&a, 3, 3);
  tla_matrix_set_value(m1, 0, 0, 4);
  tla_matrix_set_value(m1, 0, 1, 1);
  tla_matrix_set_value(m1, 0, 2, -2);
  tla_matrix_set_value(m1, 1, 0, 1);
  tla_matrix_set_value(m1, 1, 1, 3);
  tla_matrix_set_value(m1, 1, 2, 0);
  tla_matrix_set_value(m1, 2, 0, -2);
  tla_matrix_set_value(m1, 2, 1, 0);
  tla_matrix_set_value(m1, 2, 2, 5);

  tla_print_matrix(m1);
  printf("\nEigenvalues (Real parts):\n");
  tla_print_vector(tla_eigenvalues(&a, m1));

  print_section("4x4 Matrix (Complex Eigenvalues)");
  tla_Matrix *m2 = tla_matrix_create(&a, 4, 4);
  tla_matrix_set_value(m2, 0, 0, 1);
  tla_matrix_set_value(m2, 0, 1, 3);
  tla_matrix_set_value(m2, 0, 2, 2);
  tla_matrix_set_value(m2, 0, 3, 1);
  tla_matrix_set_value(m2, 1, 0, 2);
  tla_matrix_set_value(m2, 1, 1, 1);
  tla_matrix_set_value(m2, 1, 2, 4);
  tla_matrix_set_value(m2, 1, 3, 3);
  tla_matrix_set_value(m2, 2, 0, 4);
  tla_matrix_set_value(m2, 2, 1, 2);
  tla_matrix_set_value(m2, 2, 2, 1);
  tla_matrix_set_value(m2, 2, 3, 5);
  tla_matrix_set_value(m2, 3, 0, 3);
  tla_matrix_set_value(m2, 3, 1, 4);
  tla_matrix_set_value(m2, 3, 2, 2);
  tla_matrix_set_value(m2, 3, 3, 1);

  tla_print_matrix(m2);
  printf("\nEigenvalues (Real parts):\n");
  tla_print_vector(tla_eigenvalues(&a, m2));

  print_section("2x2 Symmetric Matrix");
  tla_Matrix *m3 = tla_matrix_create(&a, 2, 2);
  tla_matrix_set_value(m3, 0, 0, 2.0);
  tla_matrix_set_value(m3, 0, 1, 1.0);
  tla_matrix_set_value(m3, 1, 0, 1.0);
  tla_matrix_set_value(m3, 1, 1, 2.0);

  tla_print_matrix(m3);
  printf("\nEigenvalues (Real parts):\n");
  tla_print_vector(tla_eigenvalues(&a, m3));

  tla_arena_destroy(&a);
  return 0;
}
