CC := gcc
CFLAGS := -g -Wall -Wextra -I. -Itests
LDFLAGS := -lm

OUT_DIR := ./out
EXAMPLES_DIR := ./examples
TESTS_DIR := ./tests

SRCS := $(wildcard $(EXAMPLES_DIR)/*.c)
BINS := $(patsubst $(EXAMPLES_DIR)/%.c, $(OUT_DIR)/%, $(SRCS))

TEST_SRCS := $(wildcard $(TESTS_DIR)/*.c)
TEST_BINS := $(patsubst $(TESTS_DIR)/%.c, $(OUT_DIR)/test_%, $(TEST_SRCS))

.PHONY: all clean test

all: $(BINS)

test: $(TEST_BINS)
	@failed=0; \
	for t in $(TEST_BINS); do \
		echo ">>> $$t"; \
		$$t || failed=1; \
		echo; \
	done; \
	exit $$failed

$(OUT_DIR)/%: $(EXAMPLES_DIR)/%.c tinyla.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(OUT_DIR)/test_%: $(TESTS_DIR)/%.c tinyla.h $(TESTS_DIR)/tinytest.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

clean:
	@rm -rf $(OUT_DIR)
