#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(lu, identity) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_eye(&arena, 3);
  tla_Vector *b = tla_vector_of_value(&arena, 3, 0.0);
  vector_fill(b, (const double[]){1.0, 2.0, 3.0});

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);

  expect_unit_lower(plu.L, TOL);
  expect_upper(plu.U, TOL);
  expect_matrix_close(A, plu.U, TOL);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(i, plu.p[i]);
  }

  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);
  expect_vector_close(b, x, TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, solve_2x2) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){2.0, 1.0, 1.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){1.0, 1.0});

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);
  expect_unit_lower(plu.L, TOL);
  expect_upper(plu.U, TOL);

  tla_Matrix *PA = tla_matrix_apply_permutation_new(&arena, plu.p, A);
  tla_Matrix *LU = tla_matrix_matrix_mul_new(&arena, plu.L, plu.U);
  expect_matrix_close(PA, LU, TOL);

  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);
  EXPECT_CLOSE(0.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 1), TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, solve_3x3) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0});
  tla_Vector *x_true = tla_vector_create(&arena, 3);
  vector_fill(x_true, (const double[]){1.0, 2.0, 3.0});
  tla_Vector *b = tla_matrix_vector_mul_new(&arena, A, x_true);

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);
  expect_unit_lower(plu.L, TOL);
  expect_upper(plu.U, TOL);

  tla_Matrix *PA = tla_matrix_apply_permutation_new(&arena, plu.p, A);
  tla_Matrix *LU = tla_matrix_matrix_mul_new(&arena, plu.L, plu.U);
  expect_matrix_close(PA, LU, TOL);

  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);
  expect_vector_close(x_true, x, TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, needs_pivot) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){0.0, 1.0, 1.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){2.0, 3.0}); /* solution: x=1, y=2 */

  size_t p[2];
  tla_Matrix *L = tla_matrix_of_value(&arena, 2, 2, 0.0);
  tla_Matrix *U = tla_matrix_of_value(&arena, 2, 2, 0.0);
  int code = tla_plu(p, L, U, A);
  EXPECT_EQ(0, code);

  EXPECT_EQ((size_t)1, p[0]);
  EXPECT_EQ((size_t)0, p[1]);

  expect_unit_lower(L, TOL);
  expect_upper(U, TOL);

  tla_Matrix *PA = tla_matrix_apply_permutation_new(&arena, p, A);
  tla_Matrix *LU = tla_matrix_matrix_mul_new(&arena, L, U);
  expect_matrix_close(PA, LU, TOL);

  tla_PLUFactorization plu = {.p = p, .L = L, .U = U};
  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(2.0, tla_vector_get_value(x, 1), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, singular) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 2.0, 4.0}); /* rank 1 */

  size_t p[2];
  tla_Matrix *L = tla_matrix_of_value(&arena, 2, 2, 0.0);
  tla_Matrix *U = tla_matrix_of_value(&arena, 2, 2, 0.0);
  int code = tla_plu(p, L, U, A);
  EXPECT_EQ(-1, code);

  tla_arena_destroy(&arena);
}

TEST(lu, forward_sub) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *L = tla_matrix_create(&arena, 3, 3);
  matrix_fill(L, (const double[]){1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 4.0, 1.0});
  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(b, (const double[]){1.0, 5.0, 18.0});
  tla_Vector *y = tla_vector_of_shape(&arena, b, 0.0);

  tla_lu_forward(L, b, y);
  EXPECT_CLOSE(1.0, tla_vector_get_value(y, 0), TOL);
  EXPECT_CLOSE(3.0, tla_vector_get_value(y, 1), TOL);
  EXPECT_CLOSE(3.0, tla_vector_get_value(y, 2), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, backward_sub) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *U = tla_matrix_create(&arena, 3, 3);
  matrix_fill(U, (const double[]){2.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0});
  tla_Vector *y = tla_vector_create(&arena, 3);
  vector_fill(y, (const double[]){7.0, 9.0, 12.0});
  tla_Vector *x = tla_vector_of_shape(&arena, y, 0.0);

  tla_lu_backward(U, y, x);
  EXPECT_CLOSE(1.0, tla_vector_get_value(x, 0), TOL);
  EXPECT_CLOSE(2.0, tla_vector_get_value(x, 1), TOL);
  EXPECT_CLOSE(3.0, tla_vector_get_value(x, 2), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, example_system) {
  tla_Arena arena = tla_arena_create(256 * 1024);
  tla_Matrix *A = tla_matrix_eye(&arena, 5);
  tla_Vector *b = tla_vector_of_value(&arena, 5, 7.0);
  tla_vector_set_value(b, 1, -2.0);
  tla_swap_rows(A, 2, 4);
  tla_matrix_set_value(A, 2, 0, 5.0);
  tla_matrix_set_value(A, 0, 3, -2.0);
  tla_matrix_set_value(A, 2, 2, 3.0);

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);
  expect_unit_lower(plu.L, TOL);
  expect_upper(plu.U, TOL);

  tla_Matrix *PA = tla_matrix_apply_permutation_new(&arena, plu.p, A);
  tla_Matrix *LU = tla_matrix_matrix_mul_new(&arena, plu.L, plu.U);
  expect_matrix_close(PA, LU, TOL);

  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);

  expect_vector_values(x, (const double[]){21.0, -2.0, 7.0, 7.0, -119.0}, TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, agrees_with_gauss) {
  tla_Arena arena = tla_arena_create(256 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 4, 4);
  matrix_fill(A, (const double[]){4.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0,
                                  0.0, 5.0, 2.0, 0.0, 1.0, 2.0, 4.0});
  tla_Vector *b = tla_vector_create(&arena, 4);
  vector_fill(b, (const double[]){1.0, 2.0, 3.0, 4.0});

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);
  tla_Vector *x_lu = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x_lu, plu, b);

  tla_Matrix *aug = tla_matrix_append_column(&arena, A, b);
  tla_Vector *x_gauss = tla_vector_of_shape(&arena, b, 0.0);
  int code = tla_gauss_solve(x_gauss, aug);
  EXPECT_EQ(0, code);

  expect_vector_close(x_lu, x_gauss, TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x_lu, b), TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, multiple_rhs) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0,
                                  2.0});

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);

  tla_Vector *b1 = tla_vector_create(&arena, 3);
  vector_fill(b1, (const double[]){1.0, 0.0, 0.0});
  tla_Vector *x1 = tla_vector_of_shape(&arena, b1, 0.0);
  tla_lu_solve(&arena, x1, plu, b1);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x1, b1), TOL);

  tla_Vector *b2 = tla_vector_create(&arena, 3);
  vector_fill(b2, (const double[]){0.0, 1.0, 0.0});
  tla_Vector *x2 = tla_vector_of_shape(&arena, b2, 0.0);
  tla_lu_solve(&arena, x2, plu, b2);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x2, b2), TOL);

  expect_unit_lower(plu.L, TOL);
  expect_upper(plu.U, TOL);

  tla_arena_destroy(&arena);
}

TEST(lu, one_by_one) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 1, 1);
  tla_matrix_set_value(A, 0, 0, 7.0);
  tla_Vector *b = tla_vector_of_value(&arena, 1, 21.0);

  tla_PLUFactorization plu = tla_plu_factor(&arena, A);
  EXPECT_EQ((size_t)0, plu.p[0]);
  EXPECT_CLOSE(1.0, tla_matrix_get_value(plu.L, 0, 0), TOL);
  EXPECT_CLOSE(7.0, tla_matrix_get_value(plu.U, 0, 0), TOL);

  tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);
  tla_lu_solve(&arena, x, plu, b);
  EXPECT_CLOSE(3.0, tla_vector_get_value(x, 0), TOL);

  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(lu, identity);
  RUN_TEST(lu, solve_2x2);
  RUN_TEST(lu, solve_3x3);
  RUN_TEST(lu, needs_pivot);
  RUN_TEST(lu, singular);
  RUN_TEST(lu, forward_sub);
  RUN_TEST(lu, backward_sub);
  RUN_TEST(lu, example_system);
  RUN_TEST(lu, agrees_with_gauss);
  RUN_TEST(lu, multiple_rhs);
  RUN_TEST(lu, one_by_one);
  TINYTEST_REPORT();
}
