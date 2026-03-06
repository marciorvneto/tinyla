OUT_DIR := ./out
EXAMPLES_DIR := ./examples

all: $(OUT_DIR)/gauss $(OUT_DIR)/lu

$(OUT_DIR)/gauss: $(EXAMPLES_DIR)/gauss.c | $(OUT_DIR)
	$(CC) -o $@ $< -g -lm -I.

$(OUT_DIR)/lu: $(EXAMPLES_DIR)/lu.c | $(OUT_DIR)
	$(CC) -o $@ $< -g -lm -I.

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

clean:
	@rm -rf $(OUT_DIR)
