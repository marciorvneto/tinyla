# TinyLA

Single-file, header-only linear algebra for C.

Most temporary matrices go through a small **arena allocator** instead of
`malloc`/`free`. That keeps allocation cheap, avoids fragmentation in tight
loops, and lets you discard scratch work with a single restore.

## Author

**Márcio R. V. Neto** (sole author and maintainer)  
ORCID: [0000-0003-4574-9173](https://orcid.org/0000-0003-4574-9173)

## Projects Using This Library

<p align="left">
  <a href="https://voimatoolbox.com/pt-br">
    <img src="docs/assets/voima-toolbox-logo.png" alt="Voima Toolbox logo" height="72">
  </a>
</p>

**[Voima Toolbox](https://voimatoolbox.com/pt-br)** is an engineering calculation
platform (hydraulics, process systems, reports, interactive diagrams). TinyLA is
used there for numerical linear algebra.

If you use TinyLA in a public project and want it listed here, open a PR or an
issue.

## Features

- **Header-only (STB-style):** define `TINY_LA_IMPLEMENTATION` in one `.c` file
- **Arena allocator:** bump allocation and cheap scratch restore
- **Matrices:** add, sub, mul, transpose, row swap / combine, permutations
- **Vectors:** add, sub, dot, cross, norms, scale, normalize
- **Solvers:** Gaussian elimination and PLU ($PA = LU$)
- **Extras:** Householder / Hessenberg helpers and real eigenvalue estimates

## Quick Start

### 1. Include the library

In **one** C file, define `TINY_LA_IMPLEMENTATION` before the include. Everywhere
else, include the header as usual.

```c
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
```

### 2. Solve $Ax = b$

```c
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
#include <stdio.h>

int main(void) {
    tla_Arena arena = tla_arena_create(1024 * 1024);

    tla_Matrix *A = TLA_MATRIX(&arena, 3, 3,
        1.0, 2.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
    tla_Vector *b = TLA_VECTOR(&arena, 7.0, -2.0, 7.0);

    tla_Vector *x;
    TLA_SOLVE(&arena, A, b, x);

    printf("Solution x:\n");
    tla_print_vector(x);

    tla_arena_destroy(&arena);
    return 0;
}
```

## Building Examples and Tests

```bash
make          # examples -> ./out
make test     # build and run tests
./out/gauss
./out/lu
```

## Convenience Macros

Optional sugar on top of the normal API:

| Macro                                  | Purpose                              |
| -------------------------------------- | ------------------------------------ |
| `TLA_M(m, r, c)` / `TLA_V(v, i)`       | Element access (assignable)          |
| `TLA_VECTOR(arena, ...)`               | Arena vector from a list             |
| `TLA_MATRIX(arena, rows, cols, ...)`   | Arena matrix, row-major list         |
| `TLA_VECTOR_LIT(...)` / `tla_vec2/3/4` | Stack temporaries (by value)         |
| `TLA_SCRATCH(arena) { ... }`           | Restore arena offset at end of block |
| `TLA_SOLVE(arena, A, b, x)`            | PLU factor + solve                   |
| `MGET` / `MSET` / `VGET` / `VSET`      | Short names for get/set helpers      |

```c
// Element access
TLA_M(A, 0, 3) = -2.0;
double x0 = TLA_V(x, 0);

// Arena-backed literals
tla_Vector *b = TLA_VECTOR(&arena, 1.0, 2.0, 3.0);
tla_Matrix *A = TLA_MATRIX(&arena, 2, 2, 2.0, 1.0, 1.0, 1.0);

// Stack temporaries (valid until the end of the block)
tla_Vector e1 = tla_vec3(1.0, 0.0, 0.0);
tla_Vector tmp = TLA_VECTOR_LIT(1.0, 2.0, 3.0);

// Scratch
TLA_SCRATCH(&arena) {
    tla_Matrix *tmp = tla_matrix_of_value(&arena, n, n, 0.0);
    // ...
}

// One-shot solve
tla_Vector *x;
TLA_SOLVE(&arena, A, b, x);
```

`TLA_VECTOR` / `TLA_MATRIX` wrap `tla_vector_from_list` / `tla_matrix_from_list`
if you already have a `double *` buffer.

## Memory Management (The Arena)

Save the arena offset, allocate temps, then restore to free them in one shot:

```c
size_t scratch = tla_arena_save(&arena);

// intermediate matrices...

tla_arena_restore(&arena, scratch);

// same idea, scoped:
TLA_SCRATCH(&arena) {
    // temporary work
}
```

## Alignment and SIMD

Default allocation alignment is controlled by `TLA_ALIGNMENT` (16 bytes unless
you override it). Stricter values help when the compiler vectorizes large
kernels (e.g. 32 for AVX2, 64 for AVX-512):

```c
#define TLA_ALIGNMENT 32
#define TINY_LA_IMPLEMENTATION
#include "tinyla.h"
```

`TLA_ALIGNMENT` must be a power of 2. If you tighten it, the arena's backing
memory (from `aligned_alloc` or your own buffer) needs to meet the same
requirement.

## Citing TinyLA

If TinyLA shows up in a paper, thesis, or report, a citation is appreciated.
GitHub can also emit formats from [`CITATION.cff`](CITATION.cff) via **Cite this
repository**.

**APA-style:**

> Vianna Neto, M. R. (2026). _TinyLA_ [Computer software]. https://github.com/marciorvneto/tinyla

**BibTeX:**

```bibtex
@software{neto_tinyla,
  author  = {Neto, Márcio R. V.},
  title   = {TinyLA},
  year    = {2026},
  url     = {https://github.com/marciorvneto/tinyla},
  license = {MIT},
  orcid   = {0000-0003-4574-9173}
}
```
