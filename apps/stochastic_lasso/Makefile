LASSO_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
PETUUM_ROOT = $(LASSO_DIR)/../../

include $(PETUUM_ROOT)/defns.mk

LASSO_SRC = $(wildcard $(LASSO_DIR)/src/*.cpp)
LASSO_HDR = $(wildcard $(LASSO_DIR)/src/*.hpp)
LASSO_BIN = $(LASSO_DIR)/bin
LASSO_OBJ = $(LASSO_SRC:.cpp=.o)
NDEBUG = -DNDEBUG

all: $(LASSO_BIN)/lasso_main

$(LASSO_BIN):
	mkdir -p $(LASSO_BIN)

$(LASSO_BIN)/lasso_main: $(LASSO_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(LASSO_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$(LASSO_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(PETUUM_LDFLAGS) -o $@

$(LASSO_OBJ): %.o: %.cpp $(LASSO_HDR)
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
		$(PETUUM_INCFLAGS) -c $< -o $@

$(LASSO_BIN)/gen_data_sparse: $(LASSO_DIR)/src/tools/gen_data_sparse.cpp $(LASSO_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$< $(PETUUM_PS_LIB) $(PETUUM_LDFLAGS) -o $@

clean:
	rm -rf $(LASSO_OBJ)
	rm -rf $(LASSO_BIN)

.PHONY: clean
