#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(eigen, householder_reflector) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  /* Unit vector along e1 -> H = diag(-1, 1, 1) */
  tla_Vector *u = tla_vector_create(&arena, 3);
  vector_fill(u, (const double[]){1.0, 0.0, 0.0});
  tla_Matrix *H = tla_householder(&arena, u);

  expect_matrix_values(H,
                       (const double[]){-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                        1.0},
                       TOL);

  /* H is symmetric and involutory: H^T = H, H^2 = I */
  tla_Matrix *HT = tla_matrix_transpose_new(&arena, H);
  expect_matrix_close(H, HT, TOL);

  tla_Matrix *H2 = tla_matrix_matrix_mul_new(&arena, H, H);
  tla_Matrix *I = tla_matrix_eye(&arena, 3);
  expect_matrix_close(I, H2, TOL);

  /* Reflects u to -u */
  tla_Vector *Hu = tla_matrix_vector_mul_new(&arena, H, u);
  expect_vector_values(Hu, (const double[]){-1.0, 0.0, 0.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(eigen, householder_diagonalizes_direction) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  /* Build H that reflects v onto a multiple of e0. */
  tla_Vector *v = tla_vector_create(&arena, 3);
  vector_fill(v, (const double[]){3.0, 4.0, 0.0});
  double norm = tla_vector_norm(v);
  /* u = (v - ||v|| e0) / ||...||  (Householder that maps v -> ||v|| e0-ish) */
  tla_Vector *u = tla_vector_clone(&arena, v);
  tla_vector_set_value(u, 0, tla_vector_get_value(u, 0) - norm);
  tla_vector_normalize(u, u);

  tla_Matrix *H = tla_householder(&arena, u);
  tla_Vector *Hv = tla_matrix_vector_mul_new(&arena, H, v);

  /* Should land on ±||v|| e0 */
  EXPECT_CLOSE(norm, fabs(tla_vector_get_value(Hv, 0)), TOL);
  EXPECT_CLOSE(0.0, tla_vector_get_value(Hv, 1), TOL);
  EXPECT_CLOSE(0.0, tla_vector_get_value(Hv, 2), TOL);

  tla_arena_destroy(&arena);
}

TEST(eigen, apply_householder_left_matches_explicit) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Vector *u = tla_vector_create(&arena, 3);
  vector_fill(u, (const double[]){0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)});
  tla_Matrix *H = tla_householder(&arena, u);

  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  tla_Matrix *A_exp = tla_matrix_matrix_mul_new(&arena, H, A);

  tla_Matrix *A_imp = tla_matrix_clone(&arena, A);
  tla_apply_householder_left(A_imp, u, 0, 0);
  expect_matrix_close(A_exp, A_imp, TOL);

  tla_arena_destroy(&arena);
}

TEST(eigen, givens_zeros_second_component) {
  double a = 3.0, b = 4.0, s = 0.0, c = 0.0;
  double r_expected = 5.0; /* hypot(3,4) */

  tla_givens_rotation2(&a, &b, &s, &c);
  EXPECT_CLOSE(0.0, b, TOL);
  EXPECT_CLOSE(r_expected, fabs(a), TOL);
  EXPECT_CLOSE(1.0, c * c + s * s, TOL);

  /* Already zero b: identity rotation. */
  a = 2.0;
  b = 0.0;
  tla_givens_rotation2(&a, &b, &s, &c);
  EXPECT_CLOSE(1.0, c, TOL);
  EXPECT_CLOSE(0.0, s, TOL);
  EXPECT_CLOSE(2.0, a, TOL);
}

TEST(eigen, upper_hessenberg_structure) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 4, 4);
  matrix_fill(A, (const double[]){1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0,
                                  2.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0});

  /* Similarity via householder reductions is not tracked here; we only check
     that the in-place reduction produces upper-Hessenberg form. */
  tla_upper_hessenberg(&arena, A);

  for (size_t j = 0; j < A->cols; j++) {
    for (size_t i = j + 2; i < A->rows; i++) {
      EXPECT_CLOSE(0.0, tla_matrix_get_value(A, i, j), TOL);
    }
  }

  tla_arena_destroy(&arena);
}

TEST(eigen, diagonal_matrix) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0});

  tla_Vector *eigs = tla_eigenvalues(&arena, A);
  expect_eigenvalues_unordered(eigs, (const double[]){3.0, 1.0, 4.0}, 3,
                               EIG_TOL);

  tla_arena_destroy(&arena);
}

TEST(eigen, symmetric_2x2) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  /* [[2,1],[1,2]] has eigenvalues 3 and 1. */
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  matrix_fill(A, (const double[]){2.0, 1.0, 1.0, 2.0});

  tla_Vector *eigs = tla_eigenvalues(&arena, A);
  expect_eigenvalues_unordered(eigs, (const double[]){3.0, 1.0}, 2, EIG_TOL);

  tla_arena_destroy(&arena);
}

TEST(eigen, symmetric_3x3) {
  tla_Arena arena = tla_arena_create(64 * 1024);
  /* Same matrix as examples/eigen.c — eigenvalues known from numpy. */
  tla_Matrix *A = tla_matrix_create(&arena, 3, 3);
  matrix_fill(A, (const double[]){4.0, 1.0, -2.0, 1.0, 3.0, 0.0, -2.0, 0.0, 5.0});

  tla_Vector *eigs = tla_eigenvalues(&arena, A);
  expect_eigenvalues_unordered(
      eigs, (const double[]){6.66907909, 3.4760236, 1.85489731}, 3, 1e-4);

  tla_arena_destroy(&arena);
}

TEST(eigen, identity) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *I = tla_matrix_eye(&arena, 4);
  tla_Vector *eigs = tla_eigenvalues(&arena, I);
  expect_eigenvalues_unordered(eigs, (const double[]){1.0, 1.0, 1.0, 1.0}, 4,
                               EIG_TOL);
  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(eigen, householder_reflector);
  RUN_TEST(eigen, householder_diagonalizes_direction);
  RUN_TEST(eigen, apply_householder_left_matches_explicit);
  RUN_TEST(eigen, givens_zeros_second_component);
  RUN_TEST(eigen, upper_hessenberg_structure);
  RUN_TEST(eigen, diagonal_matrix);
  RUN_TEST(eigen, symmetric_2x2);
  RUN_TEST(eigen, symmetric_3x3);
  RUN_TEST(eigen, identity);
  TINYTEST_REPORT();
}
