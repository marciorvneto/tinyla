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

Matrix *matrix_clone(Arena *a, Matrix *m) {
  Matrix *new = matrix_create(a, m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      matrix_set_value(new, i, j, matrix_get_value(m, i, j));
    }
  }
  return new;
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

Vector *vector_clone(Arena *a, Vector *v) {
  Vector *new = vector_create(a, v->size);
  for (size_t i = 0; i < v->size; i++) {
    vector_set_value(v, i, vector_get_value(v, i));
  }
  return new;
}

Vector *vector_of_value(Arena *a, size_t size, double value) {
  Vector *v = vector_create(a, size);
  for (size_t i = 0; i < v->size; i++) {
    v->values[i] = value;
  }
  return v;
}

#define PRINT_SCI_WIDTH 11
#define PRINT_WIDTH 7
#define PRINT_PREC 3

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

void print_vector(Vector *v) {
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
      printf("%*.*e\n", PRINT_SCI_WIDTH, PRINT_PREC, value);
    } else {
      printf("%*.*f\n", PRINT_WIDTH, PRINT_PREC, value);
    }
  }
  printf("\n");
}

void print_matrix(Matrix *m) {
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
        printf("%*.*e", PRINT_SCI_WIDTH, PRINT_PREC, value);
      } else {
        printf("%*.*f", PRINT_WIDTH, PRINT_PREC, value);
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

// ------

void vector_add(Vector *out, Vector *v1, Vector *v2) {
  assert(v1->size == v2->size);
  assert(out->size == v2->size);
  for (size_t i = 0; i < v1->size; i++) {
    double value = vector_get_value(v1, i) + vector_get_value(v2, i);
    vector_set_value(out, i, value);
  }
}
Vector *vector_add_new(Arena *a, Vector *v1, Vector *v2) {
  assert(v1->size == v2->size);
  Vector *res = vector_create(a, v1->size);
  vector_add(res, v1, v2);
  return res;
}

// ------

void vector_sub(Vector *out, Vector *v1, Vector *v2) {
  assert(v1->size == v2->size);
  assert(out->size == v2->size);
  for (size_t i = 0; i < v1->size; i++) {
    double value = vector_get_value(v1, i) - vector_get_value(v2, i);
    vector_set_value(out, i, value);
  }
}
Vector *vector_sub_new(Arena *a, Vector *v1, Vector *v2) {
  assert(v1->size == v2->size);
  Vector *res = vector_create(a, v1->size);
  vector_sub(res, v1, v2);
  return res;
}

// ------

void matrix_vector_mul(Vector *out, Matrix *m, Vector *v) {
  assert(m->cols == v->size);
  assert(out->size == m->rows);
  for (size_t i = 0; i < m->rows; i++) {
    double acum = 0;
    for (size_t j = 0; j < m->cols; j++) {
      acum += matrix_get_value(m, i, j) * vector_get_value(v, j);
    }
    vector_set_value(out, i, acum);
  }
}
Vector *matrix_vector_mul_new(Arena *a, Matrix *m, Vector *v) {
  assert(m->cols == v->size);
  Vector *res = vector_create(a, m->rows);
  matrix_vector_mul(res, m, v);
  return res;
}

// ------

void matrix_matrix_mul(Matrix *out, Matrix *m1, Matrix *m2) {
  assert(m1->cols == m2->rows);
  assert(out->cols == m2->cols);
  assert(out->rows == m1->rows);
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      double acum = 0;
      for (size_t k = 0; k < m1->cols; k++) {
        acum += matrix_get_value(m1, i, k) * matrix_get_value(m2, k, j);
      }
      matrix_set_value(out, i, j, acum);
    }
  }
}
Matrix *matrix_matrix_mul_new(Arena *a, Matrix *m1, Matrix *m2) {
  assert(m1->cols == m2->rows);
  Matrix *res = matrix_create(a, m1->rows, m2->cols);
  matrix_matrix_mul(res, m1, m2);
  return res;
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

int lu_solve(Vector *x, Matrix *aug) {

  // First pass
  for (size_t col = 0; col < aug->cols - 1; col++) {
    size_t start_row = col;

    // Find pivot
    double max_pivot_value = matrix_get_value(aug, start_row, col);
    size_t pivot_idx = start_row;
    for (size_t row = start_row; row < aug->rows; row++) {
      double value = matrix_get_value(aug, row, col);
      if (fabs(value) > fabs(max_pivot_value)) {
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
      double pivot = matrix_get_value(aug, start_row, col);
      double k = matrix_get_value(aug, row, col) / pivot;
      lu_combine_rows(aug, row, k, start_row);
    }
  }

  // Solve for variables

  size_t num_vars = aug->cols - 1;
  for (size_t row = num_vars; row-- > 0;) {
    // ax + by +cz = rhs_b => x = 1/a * (b_rhs -by -cz)
    double a = matrix_get_value(aug, row, row);
    double b = matrix_get_value(aug, row, aug->cols - 1);
    double acum = 0;

    // We start accumulating from row + 1
    // [1  2  3  4 | 8]
    // [0  2  3  4 | 1]
    // [0  0  3  4 | 2]
    // [0  0  0  4 | 3]
    //     ^ row
    for (size_t j = row + 1; j < aug->cols - 1; j++) {
      acum += vector_get_value(x, j) * matrix_get_value(aug, row, j);
    }
    vector_set_value(x, row, 1.0 / a * (b - acum));
  }
  return 0;
}
Vector *lu_solve_new(Arena *a, Matrix *aug, int *code) {
  Vector *x = vector_create(a, aug->rows);
  if (!x)
    return NULL;
  *code = lu_solve(x, aug);
  return x;
}

int main() {
  Arena a = arena_create(1024 * 1024); // 1MB

  Matrix *A = matrix_eye(&a, 5);
  Vector *b = vector_of_value(&a, 5, 7);
  vector_set_value(b, 1, -2);
  swap_rows(A, 2, 4);
  matrix_set_value(A, 2, 0, 5);

  Vector *x = vector_of_value(&a, A->rows, 0);
  Matrix *Ab = matrix_append_column(&a, A, b);
  lu_solve(x, Ab);
  print_vector(x);

  Vector *check = vector_of_value(&a, A->rows, 0);
  matrix_vector_mul(check, A, x);
  print_vector(check);
  vector_sub(check, check, b);
  print_vector(check);

  arena_destroy(&a);
  return 0;
}
