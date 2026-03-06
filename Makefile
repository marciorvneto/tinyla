OUT_DIR := ./out

$(OUT_DIR)/main: main.c | $(OUT_DIR)
	@echo $@
	$(CC) -o $@ $< -g -lm 

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

clean:
	@rm -rf $(OUT_DIR)

