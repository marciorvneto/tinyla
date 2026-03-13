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

// ------------ Vectors ------------------

tla_Vector tla_vector_slice(tla_Vector *v, size_t start_index, size_t length);
tla_Vector *tla_vector_clone(tla_Arena *a, tla_Vector *v);
tla_Vector *tla_vector_of_value(tla_Arena *a, size_t size, double value);
tla_Vector *tla_vector_of_shape(tla_Arena *a, tla_Vector *base, double value);

// ------------ Conversions ------------------

tla_Matrix *tla_matrix_from_vector(tla_Arena *a, tla_Vector *base);

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

tla_Matrix *tla_matrix_transpose_new(tla_Arena *a, tla_Matrix *m);

// ------

void tla_vector_add(tla_Vector *out, tla_Vector *v1, tla_Vector *v2);
tla_Vector *tla_vector_add_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2);

// ------

void tla_vector_sub(tla_Vector *out, tla_Vector *v1, tla_Vector *v2);
tla_Vector *tla_vector_sub_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2);

// ------

double tla_vector_dot(tla_Vector *v1, tla_Vector *v2);

// ------

void tla_vector_vec(tla_Vector *vout, tla_Vector *v1, tla_Vector *v2);
tla_Vector *tla_vector_vec_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2);

// ------

double tla_vector_norm2(tla_Vector *v1);
double tla_vector_norm(tla_Vector *v1);

// ------

void tla_vector_normalize(tla_Vector *out, tla_Vector *v);
tla_Vector *tla_vector_normalize_new(tla_Arena *a, tla_Vector *v);

// ------

void tla_matrix_vector_mul(tla_Vector *out, tla_Matrix *m, tla_Vector *v);
tla_Vector *tla_matrix_vector_mul_new(tla_Arena *a, tla_Matrix *m,
                                      tla_Vector *v);

// ------

void tla_matrix_matrix_mul(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2);
tla_Matrix *tla_matrix_matrix_mul_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2);

// ------

void tla_matrix_matrix_add(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2);
tla_Matrix *tla_matrix_matrix_add_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2);
// ------

void tla_matrix_matrix_sub(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2);
tla_Matrix *tla_matrix_matrix_sub_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2);

// ------

void tla_matrix_scalar_mul(tla_Matrix *out, tla_Matrix *m, double scalar);
tla_Matrix *tla_matrix_scalar_mul_new(tla_Arena *a, tla_Matrix *m,
                                      double scalar);
//=============================
//
//   Linear Algebra
//
//=============================

// -------- Gauss -------------

int tla_gauss_solve(tla_Vector *x, tla_Matrix *aug);
tla_Vector *tla_gauss_solve_new(tla_Arena *a, tla_Matrix *aug, int *code);

// -------- LU -------------

void tla_swap_indices(size_t *indices, size_t i, size_t j);
int tla_plu(size_t *p, tla_Matrix *L, tla_Matrix *U, tla_Matrix *A);
void tla_lu_forward(tla_Matrix *L, tla_Vector *Pb, tla_Vector *y);
void tla_lu_backward(tla_Matrix *U, tla_Vector *y, tla_Vector *x);

// -------- Spectral -------------
tla_Vector *tla_eigenvalues(tla_Arena *a, tla_Matrix *m);

typedef struct {
  size_t *p;
  tla_Matrix *L;
  tla_Matrix *U;
} tla_PLUFactorization;

tla_PLUFactorization tla_plu_factor(tla_Arena *a, tla_Matrix *A);
tla_Vector *tla_lu_solve(tla_Arena *a, tla_Vector *x,
                         tla_PLUFactorization factor, tla_Vector *b);

// ----------- Eigenvalues -----------

// u must be a unit vector
tla_Matrix *tla_householder(tla_Arena *a, tla_Vector *u);
void tla_apply_householder_left(tla_Matrix *A, tla_Vector *u, size_t row_start,
                                size_t col_start);
void tla_apply_householder_right(tla_Matrix *A, tla_Vector *u, size_t row_start,
                                 size_t col_start);
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

tla_Vector tla_vector_slice(tla_Vector *v, size_t start_index, size_t length) {
  int n = (int)length - (int)start_index;
  assert(n >= 0);
  assert(n <= v->size);
  tla_Vector slice;
  slice.size = n;
  slice.values = v->values + start_index;
  return slice;
}

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

// ------------ Conversions ------------------

// Returns a column matrix containing the vector entries
// Example: if base is N-dimensional this funciton will return
// an Nx1 matrix
tla_Matrix *tla_matrix_from_vector(tla_Arena *a, tla_Vector *base) {
  tla_Matrix *m = tla_matrix_create(a, base->size, 1);
  for (size_t i = 0; i < base->size; i++) {
    tla_matrix_set_value(m, i, 0, tla_vector_get_value(base, i));
  }
  return m;
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

tla_Matrix *tla_matrix_transpose_new(tla_Arena *a, tla_Matrix *m) {
  tla_Matrix *new_matrix = tla_matrix_create(a, m->cols, m->rows);
  for (size_t row = 0; row < m->rows; row++) {
    for (size_t col = 0; col < m->cols; col++) {
      tla_matrix_set_value(new_matrix, col, row,
                           tla_matrix_get_value(m, row, col));
    }
  }
  return new_matrix;
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

double tla_vector_dot(tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  double acum = 0;
  for (size_t i = 0; i < v1->size; i++) {
    acum += tla_vector_get_value(v1, i) * tla_vector_get_value(v2, i);
  }
  return acum;
}

// ------

void tla_vector_vec(tla_Vector *vout, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  assert(v1->size == vout->size);
  assert(v1->size == 3);
  double ux = tla_vector_get_value(v1, 0);
  double uy = tla_vector_get_value(v1, 1);
  double uz = tla_vector_get_value(v1, 2);

  double vx = tla_vector_get_value(v2, 0);
  double vy = tla_vector_get_value(v2, 1);
  double vz = tla_vector_get_value(v2, 2);

  tla_vector_set_value(vout, 0, uy * vz - vy * uz);
  tla_vector_set_value(vout, 1, uz * vx - ux * vz);
  tla_vector_set_value(vout, 2, ux * vy - uy * vx);
}

tla_Vector *tla_vector_vec_new(tla_Arena *a, tla_Vector *v1, tla_Vector *v2) {
  assert(v1->size == v2->size);
  assert(v1->size == 3);
  tla_Vector *out = tla_vector_of_shape(a, v1, 0);
  tla_vector_vec(out, v1, v2);
  return out;
}

// ------

double tla_vector_norm2(tla_Vector *v1) { return tla_vector_dot(v1, v1); }
double tla_vector_norm(tla_Vector *v1) { return sqrt(tla_vector_dot(v1, v1)); }

// ------

void tla_vector_normalize(tla_Vector *out, tla_Vector *v) {
  double norm = sqrt(tla_vector_norm2(v));
  for (size_t i = 0; i < v->size; i++) {
    tla_vector_set_value(out, i, tla_vector_get_value(v, i) / norm);
  }
}

tla_Vector *tla_vector_normalize_new(tla_Arena *a, tla_Vector *v) {
  tla_Vector *res = tla_vector_clone(a, v);
  tla_vector_normalize(res, v);
  return res;
}

// ------
void tla_matrix_vector_mul(tla_Vector *out, tla_Matrix *m, tla_Vector *v) {
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
tla_Vector *tla_matrix_vector_mul_new(tla_Arena *a, tla_Matrix *m,
                                      tla_Vector *v) {
  assert(m->cols == v->size);
  tla_Vector *res = tla_vector_create(a, m->rows);
  tla_matrix_vector_mul(res, m, v);
  return res;
}

// ------

void tla_matrix_matrix_mul(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2) {
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
tla_Matrix *tla_matrix_matrix_mul_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  tla_Matrix *res = tla_matrix_create(a, m1->rows, m2->cols);
  tla_matrix_matrix_mul(res, m1, m2);
  return res;
}

// ------

void tla_matrix_matrix_add(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  assert(out->cols == m2->cols);
  assert(out->rows == m1->rows);
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      tla_matrix_set_value(out, i, j,
                           tla_matrix_get_value(m1, i, j) +
                               tla_matrix_get_value(m2, i, j));
    }
  }
}
tla_Matrix *tla_matrix_matrix_add_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  tla_Matrix *res = tla_matrix_create(a, m1->rows, m2->cols);
  tla_matrix_matrix_add(res, m1, m2);
  return res;
}

// ------

void tla_matrix_matrix_sub(tla_Matrix *out, tla_Matrix *m1, tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  assert(out->cols == m2->cols);
  assert(out->rows == m1->rows);
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      tla_matrix_set_value(out, i, j,
                           tla_matrix_get_value(m1, i, j) -
                               tla_matrix_get_value(m2, i, j));
    }
  }
}
tla_Matrix *tla_matrix_matrix_sub_new(tla_Arena *a, tla_Matrix *m1,
                                      tla_Matrix *m2) {
  assert(m1->cols == m2->rows);
  tla_Matrix *res = tla_matrix_create(a, m1->rows, m2->cols);
  tla_matrix_matrix_sub(res, m1, m2);
  return res;
}

// ------

void tla_matrix_scalar_mul(tla_Matrix *out, tla_Matrix *m, double scalar) {
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      tla_matrix_set_value(out, i, j, tla_matrix_get_value(m, i, j) * scalar);
    }
  }
}
tla_Matrix *tla_matrix_scalar_mul_new(tla_Arena *a, tla_Matrix *m,
                                      double scalar) {
  tla_Matrix *res = tla_matrix_create(a, m->rows, m->cols);
  tla_matrix_scalar_mul(res, m, scalar);
  return res;
}

//=============================
//
//   Linear Algebra
//
//=============================

// -------- Gauss -------------

int tla_gauss_solve(tla_Vector *x, tla_Matrix *aug) {

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
tla_Vector *tla_gauss_solve_new(tla_Arena *a, tla_Matrix *aug, int *code) {
  tla_Vector *x = tla_vector_create(a, aug->rows);
  if (!x)
    return NULL;
  size_t scratch = tla_arena_save(a);
  tla_Matrix *aug_clone = tla_matrix_clone(a, aug);
  *code = tla_gauss_solve(x, aug_clone);
  tla_arena_restore(a, scratch);
  return x;
}

// -------- LU -------------

void tla_swap_indices(size_t *indices, size_t i, size_t j) {
  size_t tmp = indices[i];
  indices[i] = indices[j];
  indices[j] = tmp;
}
int tla_plu(size_t *p, tla_Matrix *L, tla_Matrix *U, tla_Matrix *A) {
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
      tla_swap_indices(p, pivot_idx, start_row);

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
void tla_lu_forward(tla_Matrix *L, tla_Vector *Pb, tla_Vector *y) {
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
void tla_lu_backward(tla_Matrix *U, tla_Vector *y, tla_Vector *x) {
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

tla_PLUFactorization tla_plu_factor(tla_Arena *a, tla_Matrix *A) {
  tla_PLUFactorization factor;
  factor.p = tla_arena_alloc(a, A->rows * sizeof(size_t));
  factor.L = tla_matrix_of_value(a, A->rows, A->cols, 0);
  factor.U = tla_matrix_of_value(a, A->rows, A->cols, 0);
  tla_plu(factor.p, factor.L, factor.U, A);
  return factor;
}

tla_Vector *tla_lu_solve(tla_Arena *a, tla_Vector *x,
                         tla_PLUFactorization factor, tla_Vector *b) {
  size_t scratch = tla_arena_save(a);
  tla_Vector *y = tla_vector_create(a, b->size);
  tla_Vector *Pb = tla_vector_apply_permutation_new(a, factor.p, b);
  tla_lu_forward(factor.L, Pb, y);
  tla_lu_backward(factor.U, y, x);
  tla_arena_restore(a, scratch);
  return x;
}

// ----------- Eigenvalues -----------

// u must be a unit vector
tla_Matrix *tla_householder(tla_Arena *a, tla_Vector *u) {
  tla_Matrix *H = tla_matrix_create(a, u->size, u->size);

  for (size_t row = 0; row < u->size; row++) {
    for (size_t col = 0; col < u->size; col++) {
      // Identity part
      double val = (row == col) ? 1.0 : 0.0;
      // Subtract the outer product part: 2 * u * u^T
      val -= 2.0 * tla_vector_get_value(u, row) * tla_vector_get_value(u, col);

      tla_matrix_set_value(H, row, col, val);
    }
  }
  return H;
}

// Applies the Householder reflection H = I - 2uu^T directly to matrix A
// in-place u must be a unit vector with size equal to A->rows
void tla_apply_householder_left(tla_Matrix *A, tla_Vector *u, size_t row_start,
                                size_t col_start) {
  for (size_t col = col_start; col < A->cols; col++) {
    double dot = 0.0;
    for (size_t i = 0; i < u->size; i++) {
      dot += tla_vector_get_value(u, i) *
             tla_matrix_get_value(A, row_start + i, col);
    }
    for (size_t i = 0; i < u->size; i++) {
      double val = tla_matrix_get_value(A, row_start + i, col);
      tla_matrix_set_value(A, row_start + i, col,
                           val - 2.0 * dot * tla_vector_get_value(u, i));
    }
  }
}

// Applies the Householder reflection H = I - 2uu^T to matrix A on the right
// in-place u must be a unit vector with size equal to A->rows
void tla_apply_householder_right(tla_Matrix *A, tla_Vector *u, size_t row_start,
                                 size_t col_start) {
  for (size_t row = row_start; row < A->rows; row++) {
    double dot = 0.0;
    for (size_t i = 0; i < u->size; i++) {
      dot += tla_matrix_get_value(A, row, col_start + i) *
             tla_vector_get_value(u, i);
    }
    for (size_t i = 0; i < u->size; i++) {
      double val = tla_matrix_get_value(A, row, col_start + i);
      tla_matrix_set_value(A, row, col_start + i,
                           val - 2.0 * dot * tla_vector_get_value(u, i));
    }
  }
}

tla_Vector *tla_vector_from_matrix_column(tla_Arena *a, tla_Matrix *m,
                                          size_t column, size_t start_row,
                                          size_t end_row) {
  assert(end_row >= start_row);
  assert(column < m->cols);
  size_t n = end_row - start_row + 1;
  tla_Vector *v = tla_vector_create(a, n);
  for (size_t i = 0; i < n; i++) {
    tla_vector_set_value(v, i, tla_matrix_get_value(m, start_row + i, column));
  }
  return v;
}

void tla_givens_rotation2(double *a, double *b, double *s, double *c) {
  // G = | c   s |
  //     | -s  c |
  //
  // Operating on
  // | a |
  // | b |
  //
  // Giving
  // | r |
  // | 0 |
  //
  // The following relations hold:
  //
  // s = b/r
  // c = a/r
  //
  // Also, we define tau = b/a if |a| >= |b|, or tau = a/b if |b| > |a|
  // The point of defining tau here is to prevent overflow when calculating
  // the norm of (a,b) when a and/or b are large;
  if (fabs(*b) < 1e-14) {
    *c = 1.0;
    *s = 0.0;
    return;
  }
  if (fabs(*b) > fabs(*a)) {
    double tau = *a / *b;
    double rb = sqrt(1 + tau * tau);
    *s = 1 / rb;
    if (*b < 0)
      *s = -*s;
    *c = *s * tau;
    *a = *b / *s;
    *b = 0;
  } else {
    double tau = *b / *a;
    double ra = sqrt(1 + tau * tau);
    *c = 1 / ra;
    if (*a < 0)
      *c = -*c;
    *s = *c * tau;
    *a = *a / *c;
    *b = 0.0;
  }
}

void tla_upper_hessenberg(tla_Arena *a, tla_Matrix *m) {
  for (size_t i = 0; i + 2 < m->cols; i++) {
    size_t scratch = tla_arena_save(a);
    tla_Vector *u = tla_vector_from_matrix_column(a, m, i, i + 1, m->rows - 1);

    double norm = tla_vector_norm(u);
    if (norm <= 1e-14)
      continue;

    double sign = tla_vector_get_value(u, 0) >= 0.0 ? 1.0 : -1.0;
    double alpha = -sign * tla_vector_norm(u);
    tla_vector_set_value(u, 0, tla_vector_get_value(u, 0) - alpha);
    tla_vector_normalize(u, u);
    tla_apply_householder_left(m, u, i + 1, i);
    tla_apply_householder_right(m, u, 0, i + 1);
    tla_arena_restore(a, scratch);
  }
  for (size_t i = 0; i + 2 < m->cols; i++) {
    for (size_t j = i + 2; j < m->rows; j++) {
      tla_matrix_set_value(m, j, i, 0);
    }
  }
}

// Performs a single QR iteration step on an Upper Hessenberg matrix
void tla_hessenberg_qr_step(tla_Arena *arena, tla_Matrix *m) {

  size_t scratch = tla_arena_save(arena);
  double *c_values = tla_arena_alloc(arena, (m->cols - 1) * sizeof(double));
  double *s_values = tla_arena_alloc(arena, (m->cols - 1) * sizeof(double));

  // Left-apply QT * H -> R
  for (size_t i = 0; i < m->cols - 1; i++) {
    double c, s;
    double a = tla_matrix_get_value(m, i, i);
    double b = tla_matrix_get_value(m, i + 1, i);
    tla_givens_rotation2(&a, &b, &s, &c);
    s_values[i] = s;
    c_values[i] = c;
    tla_matrix_set_value(m, i, i, a);
    tla_matrix_set_value(m, i + 1, i, b);

    for (size_t k = i + 1; k < m->cols; k++) {
      double top = tla_matrix_get_value(m, i, k);
      double bot = tla_matrix_get_value(m, i + 1, k);

      tla_matrix_set_value(m, i, k, c * top + s * bot);
      tla_matrix_set_value(m, i + 1, k, -s * top + c * bot);
    }
  }

  // Right-apply R * Q -> Hnext
  for (size_t i = 0; i < m->cols - 1; i++) {
    double c = c_values[i];
    double s = s_values[i];

    for (size_t k = 0; k <= i + 1; k++) {
      double left = tla_matrix_get_value(m, k, i);
      double right = tla_matrix_get_value(m, k, i + 1);

      tla_matrix_set_value(m, k, i, c * left + s * right);
      tla_matrix_set_value(m, k, i + 1, -s * left + c * right);
    }
  }

  tla_arena_restore(arena, scratch);
}

// Returns a flat Vector of only the [real] parts of the eigenvalues
tla_Vector *tla_eigenvalues(tla_Arena *a, tla_Matrix *m) {
  tla_Vector *eig = tla_vector_create(a, m->cols);
  size_t scratch = tla_arena_save(a);

  tla_Matrix *new_m = tla_matrix_clone(a, m);
  tla_upper_hessenberg(a, new_m);

  double biggest = 1.0;
  size_t step = 0;

  while (biggest > 1e-12 && step < 1000) {
    biggest = 0.0;
    for (size_t k = 0; k < new_m->cols - 1; k++) {
      double val = fabs(tla_matrix_get_value(new_m, k + 1, k));
      if (val > biggest)
        biggest = val;
    }
    if (biggest <= 1e-12)
      break;
    tla_hessenberg_qr_step(a, new_m);
    step++;
  }

  // Extract purely real parts
  for (size_t i = 0; i < new_m->cols; i++) {
    if (i < new_m->cols - 1 &&
        fabs(tla_matrix_get_value(new_m, i + 1, i)) > 1e-12) {
      double a_val = tla_matrix_get_value(new_m, i, i);
      double d_val = tla_matrix_get_value(new_m, i + 1, i + 1);
      double real_part = (a_val + d_val) / 2.0;

      tla_vector_set_value(eig, i, real_part);
      tla_vector_set_value(eig, i + 1, real_part);
      i++; // Skip the second half of the 2x2 block
    } else {
      tla_vector_set_value(eig, i, tla_matrix_get_value(new_m, i, i));
    }
  }

  tla_arena_restore(a, scratch);
  return eig;
}

#endif // TINY_LA_IMPLEMENTATION

#endif // TINY_LA_H
