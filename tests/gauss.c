#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(gauss, identity) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_eye(&arena, 3);
  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(b, (const double[]){3.0, 2.0, 1.0});

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));
  expect_vector_close(b, x, TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, solve_2x2) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){2.0, 1.0, 1.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){1.0, 1.0});

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));
  EXPECT_CLOSE(0.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 1), TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, solve_3x3) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0});
  tla_Vector *x_true = tla_vector_create(&arena, 3);
  vector_fill(x_true, (const double[]){1.0, 2.0, 3.0});
  tla_Vector *b = tla_matrix_vector_mul_new(&arena, A, x_true);

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));
  expect_vector_close(x_true, x, TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, needs_pivot) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  /* Zero on diagonal without pivoting. */
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){0.0, 1.0, 1.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){2.0, 3.0}); /* x=1, y=2 */

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(2.0, tla_vector_get_value(x, 1), TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, singular) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 2.0, 4.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){1.0, 2.0});

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(-1, tla_gauss_solve(x, aug));

  tla_arena_destroy(&arena);
}

TEST(gauss, solve_new_preserves_input) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){2.0, 1.0, 1.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){1.0, 1.0});
  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);

  /* Snapshot of augmented matrix before solve_new. */
  tla_Matrix *aug_before = tla_matrix_clone(&arena, aug);

  int code = 99;
  tla_Vector *x = tla_gauss_solve_new(&arena, aug, &code);
  EXPECT_EQ(0, code);
  EXPECT_CLOSE(0.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 1), TOL);

  /* Original augmented matrix must be unchanged. */
  expect_matrix_close(aug_before, aug, TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, example_system) {
  tla_Arena arena = tla_arena_create(256 * 1024);
  tla_Matrix *A = tla_matrix_eye(&arena, 5);
  tla_Vector *b = tla_vector_of_value(&arena, 5, 7.0);
  tla_vector_set_value(b, 1, -2.0);
  tla_swap_rows(A, 2, 4);
  tla_matrix_set_value(A, 2, 0, 5.0);
  tla_matrix_set_value(A, 0, 3, -2.0);
  tla_matrix_set_value(A, 2, 2, 3.0);

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_value(&arena, A->rows, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));

  expect_vector_values(x, (const double[]){21.0, -2.0, 7.0, 7.0, -119.0}, TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(gauss, larger_randomish) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  /* Well-conditioned diagonally dominant system. */
  tla_Matrix *A = tla_matrix_create(&arena, 4, 4);
  matrix_fill(A, (const double[]){10.0, 1.0, 0.0, 2.0, 1.0, 12.0, -1.0, 0.0,
                                  0.0, -1.0, 11.0, 3.0, 2.0, 0.0, 3.0, 13.0});
  tla_Vector *x_true = tla_vector_create(&arena, 4);
  vector_fill(x_true, (const double[]){0.5, -1.0, 2.0, 1.5});
  tla_Vector *b = tla_matrix_vector_mul_new(&arena, A, x_true);

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  EXPECT_EQ(0, tla_gauss_solve(x, aug));
  expect_vector_close(x_true, x, TOL);

  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(gauss, identity);
  RUN_TEST(gauss, solve_2x2);
  RUN_TEST(gauss, solve_3x3);
  RUN_TEST(gauss, needs_pivot);
  RUN_TEST(gauss, singular);
  RUN_TEST(gauss, solve_new_preserves_input);
  RUN_TEST(gauss, example_system);
  RUN_TEST(gauss, larger_randomish);
  TINYTEST_REPORT();
}
