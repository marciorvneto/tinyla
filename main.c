#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

//=============================
//
//   Arena
//
//=============================

typedef struct {
  char *base;
  size_t offset;
  size_t capacity;
} Arena;

Arena arena_create(size_t capacity) {
  Arena a = {0};
  a.capacity = capacity;
  a.base = malloc(capacity);
  return a;
}

void *arena_alloc(Arena *a, size_t size) {
  // 8-bit aligned
  size_t aligned = (size + 7) & ~7;
  if (a->offset + aligned >= a->capacity) {
    return NULL;
  }
  char *addr = a->base + a->offset;
  a->offset += aligned;
  return addr;
}

void arena_destroy(Arena *a) { free(a->base); }

//=============================
//
//   Vectors and Matrices
//
//=============================

typedef struct {
  size_t rows;
  size_t cols;
  double *values;
} Matrix;
typedef struct {
  size_t size;
  double *values;
} Vector;

Matrix *matrix_create(Arena *a, size_t rows, size_t cols) {
  Matrix *m = arena_alloc(a, sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->values = arena_alloc(a, rows * cols * sizeof(double));
  return m;
}

Vector *vector_create(Arena *a, size_t size) {
  Vector *v = arena_alloc(a, sizeof(Vector));
  v->size = size;
  v->values = arena_alloc(a, size * sizeof(double));
  return v;
}

//=============================
//
//   Operations
//
//=============================

Matrix *matrix_append_column(Arena *a, Matrix *m, Vector *column) {
  assert(m->rows == column->size);
  Matrix *new_m = matrix_create(a, m->rows, m->cols + 1);
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

//=============================
//
//   Helpers
//
//=============================

// ------------ Matrices ------------------

void matrix_set_value(Matrix *m, size_t row, size_t col, double value) {
  m->values[row * m->cols + col] = value;
}
double matrix_get_value(Matrix *m, size_t row, size_t col) {
  return m->values[row * m->cols + col];
}

Matrix *matrix_of_value(Arena *a, size_t rows, size_t cols, double value) {
  Matrix *m = matrix_create(a, rows, cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      m->values[m->cols * i + j] = value;
    }
  }
  return m;
}

Matrix *matrix_eye(Arena *a, size_t size) {
  Matrix *m = matrix_create(a, size, size);
  for (size_t i = 0; i < m->rows; i++) {
    size_t j = i;
    m->values[m->cols * i + j] = 1.0;
  }
  return m;
}

// ------------ Vectors ------------------

void vector_set_value(Vector *v, size_t idx, double value) {
  v->values[idx] = value;
}
double vector_get_value(Vector *v, size_t idx) { return v->values[idx]; }

Vector *vector_of_value(Arena *a, size_t size, double value) {
  Vector *v = vector_create(a, size);
  for (size_t i = 0; i < v->size; i++) {
    v->values[i] = value;
  }
  return v;
}
void print_vector(Vector *v) {
  for (size_t i = 0; i < v->size; i++) {
    printf("%.3f\n", v->values[i]);
  }
  printf("\n");
}

void print_matrix(Matrix *m) {
  for (size_t i = 0; i < m->rows; i++) {
    printf("[");
    for (size_t j = 0; j < m->cols; j++) {
      printf("%.3f", m->values[i * m->cols + j]);
      if (j != m->cols - 1) {
        printf(", ");
      }
    }
    printf("]\n");
  }
  printf("\n");
}

//=============================
//
//   Linear Algebra
//
//=============================

void swap_rows(Matrix *m, size_t row1, size_t row2) {
  for (size_t col = 0; col < m->cols; col++) {
    double tmp = matrix_get_value(m, row1, col);
    matrix_set_value(m, row1, col, matrix_get_value(m, row2, col));
    matrix_set_value(m, row2, col, tmp);
  }
}

void lu_combine_rows(Matrix *m, size_t row2, double k, size_t row1) {
  // row2 <- row2 - k * row1
  for (size_t col = 0; col < m->cols; col++) {
    double r1 = matrix_get_value(m, row1, col);
    double r2 = matrix_get_value(m, row2, col);
    matrix_set_value(m, row2, col, r2 - k * r1);
  }
}

int lu(Arena *a, Matrix *A, Vector *b, Vector **x) {
  *x = vector_create(a, b->size);
  Matrix *aug = matrix_append_column(a, A, b);

  print_matrix(aug);

  // First pass
  for (size_t col = 0; col < aug->cols - 1; col++) {
    size_t start_row = col;

    // Find pivot
    double max_pivot_value = matrix_get_value(aug, col, start_row);
    size_t pivot_idx = start_row;
    for (size_t row = start_row; row < aug->rows; row++) {
      double value = matrix_get_value(aug, row, col);
      if (fabs(value) > max_pivot_value) {
        max_pivot_value = value;
        pivot_idx = row;
      }
    }
    if (fabs(max_pivot_value) < 1e-16)
      return -1;
    swap_rows(aug, pivot_idx, start_row);

    // Substract rows
    for (size_t row = start_row + 1; row < aug->rows; row++) {
      // The pivot has been swapped to start_row
      double k = matrix_get_value(aug, row, col) / max_pivot_value;
      lu_combine_rows(aug, row, k, start_row);
    }
  }

  // Solve for variables

  for (size_t col = aug->cols - 2; col < aug->cols; col--) {
    size_t row = col;
    // ax + by +cz = b => x = 1/a * (b -by -cz)
    double a = matrix_get_value(aug, row, col);
    double b = matrix_get_value(aug, row, aug->cols - 1);
    double acum = 0;
    for (size_t j = col; j < aug->cols - 1; j++) {
      acum += vector_get_value(*x, j) * matrix_get_value(aug, row, j);
    }
    vector_set_value(*x, row, 1.0 / a * (b - acum));
  }
  return 0;
}

int main() {
  Arena a = arena_create(1024 * 1024); // 1MB

  Matrix *eye = matrix_eye(&a, 5);
  Vector *new_col = vector_of_value(&a, 5, 7);
  vector_set_value(new_col, 1, -2);
  swap_rows(eye, 2, 4);
  matrix_set_value(eye, 2, 0, 5);

  Vector *x;
  lu(&a, eye, new_col, &x);
  print_vector(x);

  arena_destroy(&a);
  return 0;
}
