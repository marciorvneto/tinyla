#ifndef TINY_LA_H
#define TINY_LA_H

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

//=============================
//
//   Memory alignment
//
//=============================

#ifndef TLA_ALIGNMENT
#define TLA_ALIGNMENT 16
#endif

// Align must be a power of 2!
#define TLA_ALIGN_FORWARD(offset, align)                                       \
  (((offset) + ((align) - 1)) & ~((align) - 1))

//=============================
//
//   tla_Arena
//
//=============================

typedef struct {
  char *base;
  size_t offset;
  size_t capacity;
} tla_Arena;

tla_Arena tla_arena_create(size_t capacity);
void *tla_arena_alloc(tla_Arena *a, size_t size);
size_t tla_arena_save(tla_Arena *a);
void tla_arena_restore(tla_Arena *a, size_t saved_offset);
void tla_arena_destroy(tla_Arena *a);

//=============================
//
//   tla_Vectors and Matrices
//
//=============================

typedef struct {
  size_t rows;
  size_t cols;
  double *values;
} tla_Matrix;
typedef struct {
  size_t size;
  double *values;
} tla_Vector;

tla_Matrix *tla_matrix_create(tla_Arena *a, size_t rows, size_t cols);
tla_Vector *tla_vector_create(tla_Arena *a, size_t size);

//=============================
//
//   Helpers
//
//=============================

// ------------ Matrices ------------------

tla_Matrix *tla_matrix_clone(tla_Arena *a, tla_Matrix *m);
tla_Matrix *tla_matrix_of_value(tla_Arena *a, size_t rows, size_t cols,
                                double value);
tla_Matrix *tla_matrix_of_shape(tla_Arena *a, tla_Matrix *base, double value);
tla_Matrix *tla_matrix_eye(tla_Arena *a, size_t size);

// ------------ tla_Vectors ------------------

tla_Vector *tla_vector_clone(tla_Arena *a, tla_Vector *v);
tla_Vector *tla_vector_of_value(tla_Arena *a, size_t size, double value);
tla_Vector *tla_vector_of_shape(tla_Arena *a, tla_Vector *base, double value);

#define TLA_PRINT_SCI_WIDTH 11
#define TLA_PRINT_WIDTH 7
#define TLA_PRINT_PREC 3

void tla_print_vector(tla_Vector *v);
void tla_print_matrix(tla_Matrix *m);

//=============================
//
//   Operations
//
//=============================

tla_Matrix *tla_matrix_append_column(tla_Arena *a, tla_Matrix *m,
                                     tla_Vector *column);
void tla_swap_rows(tla_Matrix *m, size_t row1, size_t row2);

tla_Matrix *tla_matrix_apply_permutation_new(tla_Arena *a, size_t *p,
                                             tla_Matrix *A);
tla_Vector *tla_vector_apply_permutation_new(tla_Arena *a, size_t *p,
                                             tla_Vector *b);
void tla_matrix_combine_rows(tla_Matrix *m, size_t row2, double k, size_t row1);

// ------

void tla_vector_add(tla_Vector *out, tla_Vector *v1, tla_Vector *v2);
tla_Vector *tla_vector_add_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2);

// ------

void tla_vector_sub(tla_Vector *out, tla_Vector *v1, tla_Vector *v2);
tla_Vector *tla_vector_sub_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2);

// ------

void tla_matrix_tla_vector_mul(tla_Vector *out, tla_Matrix *m, tla_Vector *v);
tla_Vector *tla_matrix_tla_vector_mul_new(tla_Arena *a, tla_Matrix *m,
                                          tla_Vector *v);

// ------

void tla_matrix_tla_matrix_mul(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2);
tla_Matrix *tla_matrix_tla_matrix_mul_new(tla_Arena *a, tla_Matrix *m1,
                                          tla_Matrix *m2);

//=============================
//
//   Linear Algebra
//
//=============================

// -------- Gauss -------------

int gauss_solve(tla_Vector *x, tla_Matrix *aug);
tla_Vector *gauss_solve_new(tla_Arena *a, tla_Matrix *aug, int *code);

// -------- LU -------------

void swap_indices(size_t *indices, size_t i, size_t j);
int plu(size_t *p, tla_Matrix *L, tla_Matrix *U, tla_Matrix *A);
void lu_forward(tla_Matrix *L, tla_Vector *Pb, tla_Vector *y);
void lu_backward(tla_Matrix *U, tla_Vector *y, tla_Vector *x);

typedef struct {
  size_t *p;
  tla_Matrix *L;
  tla_Matrix *U;
} PLUFactorization;

PLUFactorization plu_factor(tla_Arena *a, tla_Matrix *A);
tla_Vector *lu_solve(tla_Arena *a, tla_Vector *x, PLUFactorization factor,
                     tla_Vector *b);

//=============================
//
//   Static inline helpers
//
//=============================

static inline void tla_matrix_set_value(tla_Matrix *m, size_t row, size_t col,
                                        double value) {
  m->values[row * m->cols + col] = value;
}
static inline double tla_matrix_get_value(tla_Matrix *m, size_t row,
                                          size_t col) {
  return m->values[row * m->cols + col];
}

static inline void tla_vector_set_value(tla_Vector *v, size_t idx,
                                        double value) {
  v->values[idx] = value;
}
static inline double tla_vector_get_value(tla_Vector *v, size_t idx) {
  return v->values[idx];
}

#ifdef TINY_LA_IMPLEMENTATION

//=============================
//
//   tla_Arena
//
//=============================

tla_Arena tla_arena_create(size_t capacity) {
  tla_Arena a = {0};
  a.capacity = capacity;
  a.base = aligned_alloc(TLA_ALIGNMENT, capacity);
  return a;
}

void *tla_arena_alloc(tla_Arena *a, size_t size) {
  size_t aligned_offset = TLA_ALIGN_FORWARD(a->offset, TLA_ALIGNMENT);
  if (aligned_offset + size > a->capacity) {
    return NULL;
  }
  char *addr = a->base + aligned_offset;
  a->offset = aligned_offset + size;
  return addr;
}

size_t tla_arena_save(tla_Arena *a) { return a->offset; }

void tla_arena_restore(tla_Arena *a, size_t saved_offset) {
  assert(saved_offset <= a->offset);
  a->offset = saved_offset;
}
void tla_arena_destroy(tla_Arena *a) { free(a->base); }

//=============================
//
//   tla_Vectors and Matrices
//
//=============================

tla_Matrix *tla_matrix_create(tla_Arena *a, size_t rows, size_t cols) {
  tla_Matrix *m = tla_arena_alloc(a, sizeof(tla_Matrix));
  m->rows = rows;
  m->cols = cols;
  m->values = tla_arena_alloc(a, rows * cols * sizeof(double));
  return m;
}

tla_Vector *tla_vector_create(tla_Arena *a, size_t size) {
  tla_Vector *v = tla_arena_alloc(a, sizeof(tla_Vector));
  v->size = size;
  v->values = tla_arena_alloc(a, size * sizeof(double));
  return v;
}

//=============================
//
//   Helpers
//
//=============================

// ------------ Matrices ------------------

tla_Matrix *tla_matrix_clone(tla_Arena *a, tla_Matrix *m) {
  tla_Matrix *new = tla_matrix_create(a, m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      tla_matrix_set_value(new, i, j, tla_matrix_get_value(m, i, j));
    }
  }
  return new;
}

tla_Matrix *tla_matrix_of_value(tla_Arena *a, size_t rows, size_t cols,
                                double value) {
  tla_Matrix *m = tla_matrix_create(a, rows, cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      m->values[m->cols * i + j] = value;
    }
  }
  return m;
}

tla_Matrix *tla_matrix_of_shape(tla_Arena *a, tla_Matrix *base, double value) {
  tla_Matrix *m = tla_matrix_create(a, base->rows, base->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      m->values[m->cols * i + j] = value;
    }
  }
  return m;
}

tla_Matrix *tla_matrix_eye(tla_Arena *a, size_t size) {
  tla_Matrix *m = tla_matrix_create(a, size, size);
  for (size_t i = 0; i < m->rows; i++) {
    size_t j = i;
    m->values[m->cols * i + j] = 1.0;
  }
  return m;
}

// ------------ tla_Vectors ------------------

tla_Vector *tla_vector_clone(tla_Arena *a, tla_Vector *v) {
  tla_Vector *new = tla_vector_create(a, v->size);
  for (size_t i = 0; i < v->size; i++) {
    tla_vector_set_value(v, i, tla_vector_get_value(v, i));
  }
  return new;
}

tla_Vector *tla_vector_of_value(tla_Arena *a, size_t size, double value) {
  tla_Vector *v = tla_vector_create(a, size);
  for (size_t i = 0; i < v->size; i++) {
    v->values[i] = value;
  }
  return v;
}

tla_Vector *tla_vector_of_shape(tla_Arena *a, tla_Vector *base, double value) {
  tla_Vector *v = tla_vector_create(a, base->size);
  for (size_t i = 0; i < v->size; i++) {
    v->values[i] = value;
  }
  return v;
}

static int should_use_scientific(double value) {
  double abs_value = fabs(value);
  if (abs_value == 0)
    return 0;
  double exp = log10(abs_value);
  if (exp < -3 || exp > 4) {
    return 1;
  }
  return 0;
}

void tla_print_vector(tla_Vector *v) {
  int use_scientific = 0;
  for (size_t i = 0; i < v->size; i++) {
    double value = v->values[i];
    use_scientific = should_use_scientific(value);
    if (use_scientific) {
      use_scientific = 1;
      break;
    }
  }
  for (size_t i = 0; i < v->size; i++) {
    double value = v->values[i];
    if (use_scientific) {
      printf("%*.*e\n", TLA_PRINT_SCI_WIDTH, TLA_PRINT_PREC, value);
    } else {
      printf("%*.*f\n", TLA_PRINT_WIDTH, TLA_PRINT_PREC, value);
    }
  }
  printf("\n");
}

void tla_print_matrix(tla_Matrix *m) {
  int use_scientific = 0;
  for (size_t i = 0; i < m->rows && !use_scientific; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      double value = m->values[i * m->cols + j];
      use_scientific = should_use_scientific(value);
      if (use_scientific) {
        use_scientific = 1;
        break;
      }
    }
  }
  printf("\n");
  for (size_t i = 0; i < m->rows; i++) {
    printf("|");
    for (size_t j = 0; j < m->cols; j++) {
      // If a number is too small or too large, use scientific notation
      double value = m->values[i * m->cols + j];
      if (use_scientific) {
        printf("%*.*e", TLA_PRINT_SCI_WIDTH, TLA_PRINT_PREC, value);
      } else {
        printf("%*.*f", TLA_PRINT_WIDTH, TLA_PRINT_PREC, value);
      }
    }
    printf(" |\n");
  }
  printf("\n");
}

//=============================
//
//   Operations
//
//=============================

tla_Matrix *tla_matrix_append_column(tla_Arena *a, tla_Matrix *m,
                                     tla_Vector *column) {
  assert(m->rows == column->size);
  tla_Matrix *new_m = tla_matrix_create(a, m->rows, m->cols + 1);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      new_m->values[new_m->cols * i + j] = m->values[m->cols * i + j];
    }
  }
  for (size_t i = 0; i < m->rows; i++) {
    size_t j = m->cols;
    new_m->values[new_m->cols * i + j] = column->values[i];
  }
  return new_m;
}
void tla_swap_rows(tla_Matrix *m, size_t row1, size_t row2) {
  for (size_t col = 0; col < m->cols; col++) {
    double tmp = tla_matrix_get_value(m, row1, col);
    tla_matrix_set_value(m, row1, col, tla_matrix_get_value(m, row2, col));
    tla_matrix_set_value(m, row2, col, tmp);
  }
}

tla_Matrix *tla_matrix_apply_permutation_new(tla_Arena *a, size_t *p,
                                             tla_Matrix *A) {
  tla_Matrix *new = tla_matrix_create(a, A->rows, A->cols);
  for (size_t row = 0; row < A->rows; row++) {
    size_t src_row = p[row];
    for (size_t col = 0; col < A->cols; col++) {
      tla_matrix_set_value(new, row, col,
                           tla_matrix_get_value(A, src_row, col));
    }
  }
  return new;
}

tla_Vector *tla_vector_apply_permutation_new(tla_Arena *a, size_t *p,
                                             tla_Vector *b) {
  tla_Vector *new = tla_vector_create(a, b->size);
  for (size_t idx = 0; idx < b->size; idx++) {
    size_t src_idx = p[idx];
    tla_vector_set_value(new, idx, tla_vector_get_value(b, src_idx));
  }
  return new;
}

void tla_matrix_combine_rows(tla_Matrix *m, size_t row2, double k,
                             size_t row1) {
  // row2 <- row2 - k * row1
  for (size_t col = 0; col < m->cols; col++) {
    double r1 = tla_matrix_get_value(m, row1, col);
    double r2 = tla_matrix_get_value(m, row2, col);
    tla_matrix_set_value(m, row2, col, r2 - k * r1);
  }
}

// ------

void tla_vector_add(tla_Vector *out, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  assert(out->size == v2->size);
  for (size_t i = 0; i < v1->size; i++) {
    double value = tla_vector_get_value(v1, i) + tla_vector_get_value(v2, i);
    tla_vector_set_value(out, i, value);
  }
}
tla_Vector *tla_vector_add_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  tla_Vector *res = tla_vector_create(a, v1->size);
  tla_vector_add(res, v1, v2);
  return res;
}

// ------

void tla_vector_sub(tla_Vector *out, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  assert(out->size == v2->size);
  for (size_t i = 0; i < v1->size; i++) {
    double value = tla_vector_get_value(v1, i) - tla_vector_get_value(v2, i);
    tla_vector_set_value(out, i, value);
  }
}
tla_Vector *tla_vector_sub_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  tla_Vector *res = tla_vector_create(a, v1->size);
  tla_vector_sub(res, v1, v2);
  return res;
}

// ------

void tla_matrix_tla_vector_mul(tla_Vector *out, tla_Matrix *m, tla_Vector *v) {
  assert(m->cols == v->size);
  assert(out->size == m->rows);
  for (size_t i = 0; i < m->rows; i++) {
    double acum = 0;
    for (size_t j = 0; j < m->cols; j++) {
      acum += tla_matrix_get_value(m, i, j) * tla_vector_get_value(v, j);
    }
    tla_vector_set_value(out, i, acum);
  }
}
tla_Vector *tla_matrix_tla_vector_mul_new(tla_Arena *a, tla_Matrix *m,
                                          tla_Vector *v) {
  assert(m->cols == v->size);
  tla_Vector *res = tla_vector_create(a, m->rows);
  tla_matrix_tla_vector_mul(res, m, v);
  return res;
}

// ------

void tla_matrix_tla_matrix_mul(tla_Matrix *out, tla_Matrix *m1,
                               tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  assert(out->cols == m2->cols);
  assert(out->rows == m1->rows);
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      double acum = 0;
      for (size_t k = 0; k < m1->cols; k++) {
        acum += tla_matrix_get_value(m1, i, k) * tla_matrix_get_value(m2, k, j);
      }
      tla_matrix_set_value(out, i, j, acum);
    }
  }
}
tla_Matrix *tla_matrix_tla_matrix_mul_new(tla_Arena *a, tla_Matrix *m1,
                                          tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  tla_Matrix *res = tla_matrix_create(a, m1->rows, m2->cols);
  tla_matrix_tla_matrix_mul(res, m1, m2);
  return res;
}

//=============================
//
//   Linear Algebra
//
//=============================

// -------- Gauss -------------

int gauss_solve(tla_Vector *x, tla_Matrix *aug) {

  // First pass
  for (size_t col = 0; col < aug->cols - 1; col++) {
    size_t start_row = col;

    // Find pivot
    double max_pivot_value = tla_matrix_get_value(aug, start_row, col);
    size_t pivot_idx = start_row;
    for (size_t row = start_row; row < aug->rows; row++) {
      double value = tla_matrix_get_value(aug, row, col);
      if (fabs(value) > fabs(max_pivot_value)) {
        max_pivot_value = value;
        pivot_idx = row;
      }
    }
    if (fabs(max_pivot_value) < 1e-16)
      return -1;
    tla_swap_rows(aug, pivot_idx, start_row);

    // Substract rows
    for (size_t row = start_row + 1; row < aug->rows; row++) {
      // The pivot has been swapped to start_row
      double pivot = tla_matrix_get_value(aug, start_row, col);
      double k = tla_matrix_get_value(aug, row, col) / pivot;
      tla_matrix_combine_rows(aug, row, k, start_row);
    }
  }

  // Solve for variables

  size_t num_vars = aug->cols - 1;
  for (size_t row = num_vars; row-- > 0;) {
    // ax + by +cz = rhs_b => x = 1/a * (b_rhs -by -cz)
    double a = tla_matrix_get_value(aug, row, row);
    double b = tla_matrix_get_value(aug, row, aug->cols - 1);
    double acum = 0;

    // We start accumulating from row + 1
    // [1  2  3  4 | 8]
    // [0  2  3  4 | 1]
    // [0  0  3  4 | 2]
    // [0  0  0  4 | 3]
    //     ^ row
    for (size_t j = row + 1; j < aug->cols - 1; j++) {
      acum += tla_vector_get_value(x, j) * tla_matrix_get_value(aug, row, j);
    }
    tla_vector_set_value(x, row, 1.0 / a * (b - acum));
  }
  return 0;
}
tla_Vector *gauss_solve_new(tla_Arena *a, tla_Matrix *aug, int *code) {
  tla_Vector *x = tla_vector_create(a, aug->rows);
  if (!x)
    return NULL;
  size_t scratch = tla_arena_save(a);
  tla_Matrix *aug_clone = tla_matrix_clone(a, aug);
  *code = gauss_solve(x, aug_clone);
  tla_arena_restore(a, scratch);
  return x;
}

// -------- LU -------------

void swap_indices(size_t *indices, size_t i, size_t j) {
  size_t tmp = indices[i];
  indices[i] = indices[j];
  indices[j] = tmp;
}
int plu(size_t *p, tla_Matrix *L, tla_Matrix *U, tla_Matrix *A) {
  for (size_t i = 0; i < A->rows; i++) {
    p[i] = i;
    for (size_t j = 0; j < A->cols; j++) {
      // L
      if (i == j) {
        tla_matrix_set_value(L, i, j, 1);
      } else {
        tla_matrix_set_value(L, i, j, 0);
      }
      // U
      tla_matrix_set_value(U, i, j, tla_matrix_get_value(A, i, j));
    }
  }

  for (size_t col = 0; col < U->cols; col++) {
    size_t start_row = col;

    // Find pivot
    double max_pivot_value = tla_matrix_get_value(U, start_row, col);
    size_t pivot_idx = start_row;
    for (size_t row = start_row; row < U->rows; row++) {
      double value = tla_matrix_get_value(U, row, col);
      if (fabs(value) > fabs(max_pivot_value)) {
        max_pivot_value = value;
        pivot_idx = row;
      }
    }
    if (fabs(max_pivot_value) < 1e-16)
      return -1;

    if (pivot_idx != start_row) {
      tla_swap_rows(U, pivot_idx, start_row);
      swap_indices(p, pivot_idx, start_row);

      // Swap everything that came before.
      // The rightmose structure needs to remain
      // identity-like
      for (size_t j = 0; j < col; j++) {
        double tmp = tla_matrix_get_value(L, pivot_idx, j);
        tla_matrix_set_value(L, pivot_idx, j,
                             tla_matrix_get_value(L, start_row, j));
        tla_matrix_set_value(L, start_row, j, tmp);
      }
    }

    // Substract rows
    for (size_t row = start_row + 1; row < U->rows; row++) {
      // The pivot has been swapped to start_row
      double pivot = tla_matrix_get_value(U, start_row, col);
      double k = tla_matrix_get_value(U, row, col) / pivot;
      tla_matrix_combine_rows(U, row, k, start_row);
      tla_matrix_set_value(L, row, col, k);
    }
  }
  return 0;
}
void lu_forward(tla_Matrix *L, tla_Vector *Pb, tla_Vector *y) {
  // a11 x1                      = b1
  // a21 x1 + a22 x2             = b2
  // a31 x1 + a32 x2 + a32 x3    = b3
  for (size_t i = 0; i < L->cols; i++) {
    double acum = 0;
    for (size_t j = 0; j < i; j++) {
      acum += tla_matrix_get_value(L, i, j) * tla_vector_get_value(y, j);
    }
    double b = tla_vector_get_value(Pb, i);
    // a is 1 by construction
    tla_vector_set_value(y, i, b - acum);
  }
}
void lu_backward(tla_Matrix *U, tla_Vector *y, tla_Vector *x) {
  // a11 x1 + a12 x2 + a12 x3    = b1
  //          a22 x2 + a23 x3    = b2
  //                   a33 x3    = b3
  for (size_t i = U->cols; i-- > 0;) {
    double acum = 0;
    for (size_t j = i + 1; j < U->cols; j++) {
      acum += tla_matrix_get_value(U, i, j) * tla_vector_get_value(x, j);
    }
    double b = tla_vector_get_value(y, i);
    double a = tla_matrix_get_value(U, i, i);
    tla_vector_set_value(x, i, (b - acum) / a);
  }
}

PLUFactorization plu_factor(tla_Arena *a, tla_Matrix *A) {
  PLUFactorization factor;
  factor.p = tla_arena_alloc(a, A->rows * sizeof(size_t));
  factor.L = tla_matrix_of_value(a, A->rows, A->cols, 0);
  factor.U = tla_matrix_of_value(a, A->rows, A->cols, 0);
  plu(factor.p, factor.L, factor.U, A);
  return factor;
}

tla_Vector *lu_solve(tla_Arena *a, tla_Vector *x, PLUFactorization factor,
                     tla_Vector *b) {
  size_t scratch = tla_arena_save(a);
  tla_Vector *y = tla_vector_create(a, b->size);
  tla_Vector *Pb = tla_vector_apply_permutation_new(a, factor.p, b);
  lu_forward(factor.L, Pb, y);
  lu_backward(factor.U, y, x);
  tla_arena_restore(a, scratch);
  return x;
}
#endif // TINY_LA_IMPLEMENTATION

#endif // TINY_LA_H
