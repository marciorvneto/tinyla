#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

static void print_section(const char *title) {
  printf("\n=== %s ===\n", title);
}

int main(void) {
  tla_Arena a = tla_arena_create(1024 * 1024);

  tla_Vector *v = tla_vector_create(&a, 3);
  tla_vector_set_value(v, 0, 1.0);
  tla_vector_set_value(v, 1, 2.0);
  tla_vector_set_value(v, 2, 3.0);

  print_section("Original Vector (v)");
  tla_print_vector(v);

  tla_Matrix *m_col = tla_matrix_from_vector(&a, v);
  print_section("Column Matrix (M)");
  tla_print_matrix(m_col);

  tla_Matrix *m_row = tla_matrix_transpose_new(&a, m_col);
  print_section("Row Matrix (M^T)");
  tla_print_matrix(m_row);

  tla_Matrix *outer = tla_matrix_matrix_mul_new(&a, m_col, m_row);
  print_section("Outer Product (M * M^T) -> 3x3 Matrix");
  tla_print_matrix(outer);

  tla_Matrix *inner = tla_matrix_matrix_mul_new(&a, m_row, m_col);
  print_section("Inner Product (M^T * M) -> 1x1 Matrix");
  tla_print_matrix(inner);

  tla_Matrix *eye = tla_matrix_eye(&a, 3);
  tla_Matrix *eye_check = tla_matrix_matrix_mul_new(&a, eye, outer);
  print_section("Check: Identity * Outer Product");
  tla_print_matrix(eye_check);

  tla_arena_destroy(&a);
  return 0;
}
