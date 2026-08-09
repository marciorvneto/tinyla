#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(convenience, element_access) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Matrix *A = tla_matrix_create(&arena, 2, 2);
  tla_Vector *v = tla_vector_create(&arena, 2);

  TLA_M(A, 0, 0) = 1.0;
  TLA_M(A, 0, 1) = 2.0;
  TLA_M(A, 1, 0) = 3.0;
  TLA_M(A, 1, 1) = 4.0;
  TLA_V(v, 0) = 10.0;
  TLA_V(v, 1) = 20.0;

  EXPECT_CLOSE(1.0, TLA_M(A, 0, 0), TOL);
  EXPECT_CLOSE(4.0, TLA_M(A, 1, 1), TOL);
  EXPECT_CLOSE(10.0, TLA_V(v, 0), TOL);
  EXPECT_CLOSE(20.0, TLA_V(v, 1), TOL);

  /* Short aliases match the function helpers. */
  MSET(A, 0, 0, 9.0);
  EXPECT_CLOSE(9.0, MGET(A, 0, 0), TOL);
  VSET(v, 0, 7.0);
  EXPECT_CLOSE(7.0, VGET(v, 0), TOL);

  tla_arena_destroy(&arena);
}

TEST(convenience, literal_constructors) {
  tla_Arena arena = tla_arena_create(16 * 1024);

  tla_Vector *b = TLA_VECTOR(&arena, 1.0, 2.0, 3.0);
  EXPECT_EQ(3, b->size);
  expect_vector_values(b, (const double[]){1.0, 2.0, 3.0}, TOL);

  tla_Matrix *A = TLA_MATRIX(&arena, 2, 2, 2.0, 1.0, 1.0, 1.0);
  EXPECT_EQ(2, A->rows);
  EXPECT_EQ(2, A->cols);
  expect_matrix_values(A, (const double[]){2.0, 1.0, 1.0, 1.0}, TOL);

  /* Stack-backed literals (by value) for temporary use. */
  tla_Vector lit = TLA_VECTOR_LIT(4.0, 5.0, 6.0);
  EXPECT_EQ(3, lit.size);
  EXPECT_CLOSE(4.0, TLA_V(&lit, 0), TOL);
  EXPECT_CLOSE(6.0, TLA_V(&lit, 2), TOL);

  tla_Vector e1 = tla_vec3(1.0, 0.0, 0.0);
  EXPECT_CLOSE(1.0, TLA_V(&e1, 0), TOL);
  EXPECT_CLOSE(0.0, TLA_V(&e1, 1), TOL);

  tla_arena_destroy(&arena);
}

TEST(convenience, scratch_restores) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *keep = tla_matrix_eye(&arena, 2);
  size_t before = tla_arena_save(&arena);

  TLA_SCRATCH(&arena) {
    tla_Matrix *tmp = tla_matrix_of_value(&arena, 8, 8, 1.0);
    tla_Vector *junk = TLA_VECTOR(&arena, 1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(tmp != NULL);
    EXPECT_TRUE(junk != NULL);
    EXPECT_TRUE(arena.offset > before);
  }

  EXPECT_EQ(before, arena.offset);
  EXPECT_CLOSE(1.0, TLA_M(keep, 0, 0), TOL);
  EXPECT_CLOSE(0.0, TLA_M(keep, 0, 1), TOL);

  tla_arena_destroy(&arena);
}

TEST(convenience, solve) {
  tla_Arena arena = tla_arena_create(32 * 1024);
  tla_Matrix *A = TLA_MATRIX(&arena, 2, 2, 2.0, 1.0, 1.0, 1.0);
  tla_Vector *b = TLA_VECTOR(&arena, 1.0, 1.0);

  tla_Vector *x;
  TLA_SOLVE(&arena, A, b, x);

  EXPECT_CLOSE(0.0, TLA_V(x, 0), TOL);
  EXPECT_CLOSE(1.0, TLA_V(x, 1), TOL);
  EXPECT_CLOSE(0.0, residual_norm(&arena, A, x, b), TOL);

  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(convenience, element_access);
  RUN_TEST(convenience, literal_constructors);
  RUN_TEST(convenience, scratch_restores);
  RUN_TEST(convenience, solve);
  TINYTEST_REPORT();
}
