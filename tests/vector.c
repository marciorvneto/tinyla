#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include "test_helpers.h"
#include "tinytest.h"

TEST(vector, create_and_set) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 3);
  EXPECT_EQ(3, v->size);

  vector_fill(v, (const double[]){1.0, 2.0, 3.0});
  expect_vector_values(v, (const double[]){1.0, 2.0, 3.0}, TOL);

  tla_Vector *ones = tla_vector_of_value(&arena, 4, 1.5);
  EXPECT_EQ(4, ones->size);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_CLOSE(1.5, tla_vector_get_value(ones, i), TOL);
  }

  tla_arena_destroy(&arena);
}

TEST(vector, clone_and_copy) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 3);
  vector_fill(v, (const double[]){4.0, 5.0, 6.0});

  tla_Vector *c = tla_vector_clone(&arena, v);
  expect_vector_close(v, c, TOL);

  /* Mutating the clone must not touch the original. */
  tla_vector_set_value(c, 0, -1.0);
  EXPECT_CLOSE(4.0, tla_vector_get_value(v, 0), TOL);
  EXPECT_CLOSE(-1.0, tla_vector_get_value(c, 0), TOL);

  tla_Vector *dst = tla_vector_of_value(&arena, 3, 0.0);
  tla_vector_copy_into(dst, v);
  expect_vector_close(v, dst, TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, add_sub) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *a = tla_vector_create(&arena, 3);
  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(a, (const double[]){1.0, 2.0, 3.0});
  vector_fill(b, (const double[]){4.0, 5.0, 6.0});

  tla_Vector *sum = tla_vector_add_new(&arena, a, b);
  expect_vector_values(sum, (const double[]){5.0, 7.0, 9.0}, TOL);

  tla_Vector *diff = tla_vector_sub_new(&arena, b, a);
  expect_vector_values(diff, (const double[]){3.0, 3.0, 3.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, dot_norm) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *a = tla_vector_create(&arena, 3);
  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(a, (const double[]){1.0, 2.0, 3.0});
  vector_fill(b, (const double[]){-1.0, 0.0, 6.0});

  EXPECT_CLOSE(17.0, tla_vector_dot(a, b), TOL); /* -1 + 0 + 18 */
  EXPECT_CLOSE(14.0, tla_vector_norm2(a), TOL);  /* 1+4+9 */
  EXPECT_CLOSE(sqrt(14.0), tla_vector_norm(a), TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, cross_product) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *i = tla_vector_create(&arena, 3);
  tla_Vector *j = tla_vector_create(&arena, 3);
  vector_fill(i, (const double[]){1.0, 0.0, 0.0});
  vector_fill(j, (const double[]){0.0, 1.0, 0.0});

  tla_Vector *k = tla_vector_vec_new(&arena, i, j);
  expect_vector_values(k, (const double[]){0.0, 0.0, 1.0}, TOL);

  /* a x b = - (b x a) */
  tla_Vector *neg = tla_vector_vec_new(&arena, j, i);
  expect_vector_values(neg, (const double[]){0.0, 0.0, -1.0}, TOL);

  /* Example from examples/vector.c: (1,2,3) x (-1,0,6) = (12, -9, 2) */
  tla_Vector *a = tla_vector_create(&arena, 3);
  tla_Vector *b = tla_vector_create(&arena, 3);
  vector_fill(a, (const double[]){1.0, 2.0, 3.0});
  vector_fill(b, (const double[]){-1.0, 0.0, 6.0});
  tla_Vector *c = tla_vector_vec_new(&arena, a, b);
  expect_vector_values(c, (const double[]){12.0, -9.0, 2.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, normalize_and_scale) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 3);
  vector_fill(v, (const double[]){3.0, 0.0, 4.0});

  tla_Vector *n = tla_vector_normalize_new(&arena, v);
  EXPECT_CLOSE(1.0, tla_vector_norm(n), TOL);
  expect_vector_values(n, (const double[]){0.6, 0.0, 0.8}, TOL);

  /* Original unchanged. */
  expect_vector_values(v, (const double[]){3.0, 0.0, 4.0}, TOL);

  tla_Vector *s = tla_vector_scalar_mul_new(&arena, v, 2.0);
  expect_vector_values(s, (const double[]){6.0, 0.0, 8.0}, TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, slice) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 5);
  vector_fill(v, (const double[]){10.0, 20.0, 30.0, 40.0, 50.0});

  /* slice is a view: [start, end) */
  tla_Vector s = tla_vector_slice(v, 1, 4);
  EXPECT_EQ(3, s.size);
  EXPECT_CLOSE(20.0, tla_vector_get_value(&s, 0), TOL);
  EXPECT_CLOSE(30.0, tla_vector_get_value(&s, 1), TOL);
  EXPECT_CLOSE(40.0, tla_vector_get_value(&s, 2), TOL);

  tla_arena_destroy(&arena);
}

TEST(vector, rotation_z_90) {
  tla_Arena arena = tla_arena_create(16 * 1024);
  tla_Vector *v = tla_vector_create(&arena, 3);
  vector_fill(v, (const double[]){1.0, 0.0, 0.0});

  /* Active +90° about z: (1,0,0) -> (0,1,0) */
  tla_Vector *r = tla_apply_rot_z_new(&arena, v, M_PI / 2.0);
  expect_vector_values(r, (const double[]){0.0, 1.0, 0.0}, 1e-12);

  /* Passive +90° is active -90°: (1,0,0) -> (0,-1,0) */
  tla_Vector *p = tla_apply_rot_z_passive_new(&arena, v, M_PI / 2.0);
  expect_vector_values(p, (const double[]){0.0, -1.0, 0.0}, 1e-12);

  tla_arena_destroy(&arena);
}

INIT_TINYTEST()

int main(void) {
  RUN_TEST(vector, create_and_set);
  RUN_TEST(vector, clone_and_copy);
  RUN_TEST(vector, add_sub);
  RUN_TEST(vector, dot_norm);
  RUN_TEST(vector, cross_product);
  RUN_TEST(vector, normalize_and_scale);
  RUN_TEST(vector, slice);
  RUN_TEST(vector, rotation_z_90);
  TINYTEST_REPORT();
}
