# TinyLA

A lightweight, single-file, header-only linear algebra library in C.

TinyLA is designed for high-performance numerical computing. Instead of relying on standard `malloc`/`free` for thousands of temporary matrix allocations, it uses a custom **Arena Allocator**. This guarantees zero memory fragmentation, fast bump-pointer allocations, and instantly scrubbable scratch memory, which is very convenient for heavy iterative loops and complex calculations.

## Features

- **Header-only (STB-style):** Drop `tinyla.h` into your project and you're done. No complex build systems or dependencies.
- **Arena-backed Memory:** Fast, contiguous memory allocation.
- **Matrix Operations:** Addition, subtraction, multiplication, row permutations, and transposition ($M^T$).
- **Vector Operations:** Addition, subtraction, dot product, cross product, $L^2$ norm, and vector-to-matrix conversions for inner/outer products.
- **Solvers:** Gaussian elimination and PLU Factorization ($PA = LU$).

## Quick Start

### 1. Include the Library

In **one** C file, define `TINY_LA_IMPLEMENTATION` before including the header to compile the implementation. In all other files, just include the header normally.

```c
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"

```

### 2. Solving a System ($Ax = b$)

Here is a quick example of setting up an arena, defining a system, and solving it using PLU factorization.

```c
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

int main(void) {
    // 1. Initialize a 1MB arena
    tla_Arena arena = tla_arena_create(1024 * 1024);

    // 2. Create A and b
    tla_Matrix *A = tla_matrix_eye(&arena, 3);
    tla_Vector *b = tla_vector_of_value(&arena, 3, 7.0);

    // Modify some values
    tla_matrix_set_value(A, 0, 1, 2.0);
    tla_vector_set_value(b, 1, -2.0);

    // 3. Factorize and solve
    tla_PLUFactorization plu = plu_factor(&arena, A);
    tla_Vector *x = tla_vector_of_shape(&arena, b, 0.0);

    tla_lu_solve(&arena, x, plu, b);

    // 4. Output results
    printf("Solution x:\n");
    tla_print_vector(x);

    // 5. Clean up all memory in one shot
    tla_arena_destroy(&arena);
    return 0;
}

```

## Building the Examples

The repository includes a simple `Makefile` and heavily commented examples demonstrating both Gaussian elimination and LU factorization.

To build and run the examples:

```bash
make
./out/gauss
./out/lu

```

## Memory Management (The Arena)

When performing complex linear algebra, intermediate allocations can bottleneck performance. TinyLA allows you to save and restore arena states, acting as a high-speed scratchpad.

```c
size_t scratch = tla_arena_save(&arena);

// ... perform allocations for intermediate matrices ...

// Instantly free all intermediate memory, keeping earlier data intact
tla_arena_restore(&arena, scratch);

```

## Advanced: Memory Alignment and SIMD

By default, TinyLA aligns memory allocations to 8-byte boundaries. However, if you are working with large matrices and want to take advantage of CPU auto-vectorization (SIMD instructions), modern architectures often require stricter alignment (e.g., 16 bytes for SSE, 32 bytes for AVX2, or 64 bytes for AVX-512).

You can override the default alignment by defining `TLA_ALIGNMENT` before including the implementation:

```c
#define TLA_ALIGNMENT 32 // Align to 32 bytes for AVX2
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"

```

_Note: `TLA_ALIGNMENT` must be a power of 2. If you use strict alignment requirements, ensure the memory block you pass to your arena (e.g., via `aligned_alloc`) is also aligned to at least the same boundary._
