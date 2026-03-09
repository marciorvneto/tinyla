CC := gcc
CFLAGS := -g -lm -I.

OUT_DIR := ./out
EXAMPLES_DIR := ./examples

SRCS := $(wildcard $(EXAMPLES_DIR)/*.c)
BINS := $(patsubst $(EXAMPLES_DIR)/%.c, $(OUT_DIR)/%, $(SRCS))

.PHONY: all clean

all: $(BINS)

$(OUT_DIR)/%: $(EXAMPLES_DIR)/%.c | $(OUT_DIR)
	$(CC) -o $@ $< $(CFLAGS)

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

clean:
	@rm -rf $(OUT_DIR)
