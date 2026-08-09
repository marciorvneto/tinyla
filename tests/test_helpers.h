#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "tinytest.h"
#include "tinyla.h"

#include <stddef.h>
#include <stdlib.h>

/* Strict tolerance for direct methods; looser for iterative eigensolvers. */
#define TOL 1e-10
#define EIG_TOL 1e-6

static inline void matrix_fill(tla_Matrix *m, const double *data) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      tla_matrix_set_value(m, i, j, data[i * m->cols + j]);
    }
  }
}

static inline void vector_fill(tla_Vector *v, const double *data) {
  for (size_t i = 0; i < v->size; i++) {
    tla_vector_set_value(v, i, data[i]);
  }
}

static inline void expect_vector_close(tla_Vector *expected, tla_Vector *actual,
                                       double tol) {
  EXPECT_EQ(expected->size, actual->size);
  for (size_t i = 0; i < expected->size; i++) {
    EXPECT_CLOSE(tla_vector_get_value(expected, i),
                 tla_vector_get_value(actual, i), tol);
  }
}

static inline void expect_vector_values(tla_Vector *v, const double *expected,
                                        double tol) {
  for (size_t i = 0; i < v->size; i++) {
    EXPECT_CLOSE(expected[i], tla_vector_get_value(v, i), tol);
  }
}

static inline void expect_matrix_close(tla_Matrix *expected, tla_Matrix *actual,
                                       double tol) {
  EXPECT_EQ(expected->rows, actual->rows);
  EXPECT_EQ(expected->cols, actual->cols);
  for (size_t i = 0; i < expected->rows; i++) {
    for (size_t j = 0; j < expected->cols; j++) {
      EXPECT_CLOSE(tla_matrix_get_value(expected, i, j),
                   tla_matrix_get_value(actual, i, j), tol);
    }
  }
}

static inline void expect_matrix_values(tla_Matrix *m, const double *expected,
                                        double tol) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      EXPECT_CLOSE(expected[i * m->cols + j], tla_matrix_get_value(m, i, j),
                   tol);
    }
  }
}

static inline void expect_unit_lower(tla_Matrix *L, double tol) {
  for (size_t i = 0; i < L->rows; i++) {
    for (size_t j = 0; j < L->cols; j++) {
      double v = tla_matrix_get_value(L, i, j);
      if (i == j) {
        EXPECT_CLOSE(1.0, v, tol);
      } else if (j > i) {
        EXPECT_CLOSE(0.0, v, tol);
      }
    }
  }
}

static inline void expect_upper(tla_Matrix *U, double tol) {
  for (size_t i = 0; i < U->rows; i++) {
    for (size_t j = 0; j < i && j < U->cols; j++) {
      EXPECT_CLOSE(0.0, tla_matrix_get_value(U, i, j), tol);
    }
  }
}

/* Residual ||Ax - b||_2 */
static inline double residual_norm(tla_Arena *a, tla_Matrix *A, tla_Vector *x,
                                   tla_Vector *b) {
  size_t scratch = tla_arena_save(a);
  tla_Vector *Ax = tla_vector_of_shape(a, b, 0.0);
  tla_matrix_vector_mul(Ax, A, x);
  tla_vector_sub(Ax, Ax, b);
  double n = tla_vector_norm(Ax);
  tla_arena_restore(a, scratch);
  return n;
}

/* Match eigenvalues regardless of order (multiset compare). */
static inline void expect_eigenvalues_unordered(tla_Vector *got,
                                                const double *expected,
                                                size_t n, double tol) {
  EXPECT_EQ(n, got->size);

  int *used = calloc(n, sizeof(int));
  EXPECT_TRUE(used != NULL);
  if (!used)
    return;

  for (size_t i = 0; i < n; i++) {
    double e = expected[i];
    int found = 0;
    for (size_t j = 0; j < n; j++) {
      if (used[j])
        continue;
      if (fabs(tla_vector_get_value(got, j) - e) <= tol) {
        used[j] = 1;
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("  %s:%d: Failure\n", __FILE__, __LINE__);
      printf("    Missing eigenvalue near %g in result: [", e);
      for (size_t j = 0; j < n; j++) {
        printf("%g%s", tla_vector_get_value(got, j), j + 1 < n ? ", " : "");
      }
      printf("]\n");
      tinytest_current_failed = 1;
    }
  }
  free(used);
}

#endif /* TEST_HELPERS_H */
