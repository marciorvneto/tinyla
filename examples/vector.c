#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

static void print_section(const char *title) {
  printf("\n=== %s ===\n", title);
}

int main() {
  tla_Arena a = tla_arena_create(1024); // 1kB

  tla_Vector *v1 = tla_vector_create(&a, 3);
  tla_vector_set_value(v1, 0, 1);
  tla_vector_set_value(v1, 1, 2);
  tla_vector_set_value(v1, 2, 3);

  print_section("Vector v1");
  tla_print_vector(v1);

  tla_Vector *v2 = tla_vector_create(&a, 3);
  tla_vector_set_value(v2, 0, -1);
  tla_vector_set_value(v2, 1, 0);
  tla_vector_set_value(v2, 2, 6);

  print_section("Vector v2");
  tla_print_vector(v2);

  print_section("Vector operations");

  double norm2 = tla_vector_norm2(v1);
  printf("||v1||² = %f\n\n", norm2);

  double dot_product = tla_vector_dot(v1, v2);
  printf("dot(v1,v2) = %f\n\n", dot_product);

  tla_Vector *vec_product = tla_vector_of_shape(&a, v1, 0);
  tla_vector_vec(vec_product, v1, v2);
  printf("v1 x v2 =\n\n");
  tla_print_vector(vec_product);

  tla_arena_destroy(&a);
}
