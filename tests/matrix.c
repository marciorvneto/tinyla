#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(matrix, eye_and_set) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *I = tla_matrix_eye(&arena, 3);
  expect_matrix_values(I,
                       (const double[]){1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                        1.0},
                       TOL);

  tla_Matrix *z = tla_matrix_of_value(&arena, 2, 3, -1.0);
  EXPECT_EQ(2, z->rows);
  EXPECT_EQ(3, z->cols);
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_CLOSE(-1.0, tla_matrix_get_value(z, i, j), TOL);
    }
  }

  tla_arena_destroy(&arena);
}

TEST(matrix, clone_and_copy) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0});

  tla_Matrix *C = tla_matrix_clone(&arena, A);
  expect_matrix_close(A, C, TOL);
  tla_matrix_set_value(C, 0, 0, 99.0);
  EXPECT_CLOSE(1.0, tla_matrix_get_value(A, 0, 0), TOL);

  tla_Matrix *D = tla_matrix_of_shape(&arena, A, 0.0);
  tla_matrix_copy_into(D, A);
  expect_matrix_close(A, D, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, add_sub_scale) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  tla_Matrix *B = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0});
  matrix_fill(B, (const double[]){4.0, 3.0, 2.0, 1.0});

  tla_Matrix *S = tla_matrix_matrix_add_new(&arena, A, B);
  expect_matrix_values(S, (const double[]){5.0, 5.0, 5.0, 5.0}, TOL);

  tla_Matrix *D = tla_matrix_matrix_sub_new(&arena, A, B);
  expect_matrix_values(D, (const double[]){-3.0, -1.0, 1.0, 3.0}, TOL);

  tla_Matrix *K = tla_matrix_scalar_mul_new(&arena, A, 0.5);
  expect_matrix_values(K, (const double[]){0.5, 1.0, 1.5, 2.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, mul_and_identity) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 3);
  tla_Matrix *B = tla_matrix_create(&arena, 3, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  matrix_fill(B, (const double[]){7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  /* A*B = [[58, 64], [139, 154]] */
  tla_Matrix *C = tla_matrix_matrix_mul_new(&arena, A, B);
  EXPECT_EQ(2, C->rows);
  EXPECT_EQ(2, C->cols);
  expect_matrix_values(C, (const double[]){58.0, 64.0, 139.0, 154.0}, TOL);

  /* Right-multiply by identity preserves A. */
  tla_Matrix *I3 = tla_matrix_eye(&arena, 3);
  tla_Matrix *AI3 = tla_matrix_matrix_mul_new(&arena, A, I3);
  expect_matrix_close(A, AI3, TOL);

  /* Left-multiply by identity preserves A. */
  tla_Matrix *I2 = tla_matrix_eye(&arena, 2);
  tla_Matrix *I2A = tla_matrix_matrix_mul_new(&arena, I2, A);
  expect_matrix_close(A, I2A, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, matvec) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  tla_Vector *x = tla_vector_create(&arena, 3);
  vector_fill(x, (const double[]){1.0, 1.0, 1.0});

  tla_Vector *y = tla_matrix_vector_mul_new(&arena, A, x);
  expect_vector_values(y, (const double[]){6.0, 15.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, transpose) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  tla_Matrix *T = tla_matrix_transpose_new(&arena, A);
  EXPECT_EQ(3, T->rows);
  EXPECT_EQ(2, T->cols);
  expect_matrix_values(T, (const double[]){1.0, 4.0, 2.0, 5.0, 3.0, 6.0}, TOL);

  /* (A^T)^T = A */
  tla_Matrix *TT = tla_matrix_transpose_new(&arena, T);
  expect_matrix_close(A, TT, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, outer_and_inner_product) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 3);
  vector_fill(v, (const double[]){1.0, 2.0, 3.0});

  tla_Matrix *col = tla_matrix_from_vector(&arena, v);
  tla_Matrix *row = tla_matrix_transpose_new(&arena, col);

  tla_Matrix *outer = tla_matrix_matrix_mul_new(&arena, col, row);
  expect_matrix_values(outer,
                       (const double[]){1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0,
                                        9.0},
                       TOL);

  tla_Matrix *inner = tla_matrix_matrix_mul_new(&arena, row, col);
  EXPECT_EQ(1, inner->rows);
  EXPECT_EQ(1, inner->cols);
  EXPECT_CLOSE(14.0, tla_matrix_get_value(inner, 0, 0), TOL);
  EXPECT_CLOSE(tla_vector_dot(v, v), tla_matrix_get_value(inner, 0, 0), TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, swap_rows_and_combine) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  tla_swap_rows(A, 0, 2);
  expect_matrix_values(A,
                       (const double[]){7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 1.0, 2.0,
                                        3.0},
                       TOL);

  /* row1 = row1 - 2 * row0  =>  [4,5,6] - 2*[7,8,9] = [-10,-11,-12] */
  tla_matrix_combine_rows(A, 1, 2.0, 0);
  EXPECT_CLOSE(-10.0, tla_matrix_get_value(A, 1, 0), TOL);
  EXPECT_CLOSE(-11.0, tla_matrix_get_value(A, 1, 1), TOL);
  EXPECT_CLOSE(-12.0, tla_matrix_get_value(A, 1, 2), TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, append_column) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_eye(&arena, 2);
  tla_Vector *b = tla_vector_create(&arena, 2);
  vector_fill(b, (const double[]){7.0, 8.0});

  tla_Matrix *Ab = tla_matrix_append_column(&arena, A, b);
  EXPECT_EQ(2, Ab->rows);
  EXPECT_EQ(3, Ab->cols);
  expect_matrix_values(Ab, (const double[]){1.0, 0.0, 7.0, 0.0, 1.0, 8.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, apply_permutation) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 2);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  size_t p[] = {2, 0, 1}; /* new row i comes from old row p[i] */

  tla_Matrix *PA = tla_matrix_apply_permutation_new(&arena, p, A);
  expect_matrix_values(PA, (const double[]){5.0, 6.0, 1.0, 2.0, 3.0, 4.0}, TOL);

  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(b, (const double[]){10.0, 20.0, 30.0});
  tla_Vector *Pb = tla_vector_apply_permutation_new(&arena, p, b);
  expect_vector_values(Pb, (const double[]){30.0, 10.0, 20.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(matrix, arena_scratch) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *keep = tla_matrix_eye(&arena, 2);
  size_t mark = tla_arena_save(&arena);

  tla_Matrix *tmp = tla_matrix_of_value(&arena, 10, 10, 1.0);
  EXPECT_TRUE(tmp != NULL);
  EXPECT_TRUE(arena.offset > mark);

  tla_arena_restore(&arena, mark);
  EXPECT_EQ(mark, arena.offset);

  /* Earlier allocation still valid. */
  EXPECT_CLOSE(1.0, tla_matrix_get_value(keep, 0, 0), TOL);

  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(matrix, eye_and_set);
  RUN_TEST(matrix, clone_and_copy);
  RUN_TEST(matrix, add_sub_scale);
  RUN_TEST(matrix, mul_and_identity);
  RUN_TEST(matrix, matvec);
  RUN_TEST(matrix, transpose);
  RUN_TEST(matrix, outer_and_inner_product);
  RUN_TEST(matrix, swap_rows_and_combine);
  RUN_TEST(matrix, append_column);
  RUN_TEST(matrix, apply_permutation);
  RUN_TEST(matrix, arena_scratch);
  TINYTEST_REPORT();
}
