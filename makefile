EXEC_NAME=myneuralnet

CC=gcc
CCFLAGS=-march=native

CXX=g++
CXXFLAGS=-std=c++20

SDIR=src
BDIR=build
ODIR=$(BDIR)/obj
LDIR=lib

LIBS=-lm

SRC=$(wildcard $(SDIR)/*.c) $(wildcard $(SDIR)/*.cc)
DEP=$(wildcard $(SDIR)/*.h) $(wildcard $(SDIR)/*.hh)
OBJ=$(patsubst $(SDIR)/%,$(ODIR)/%.o,$(SRC))

$(BDIR)/$(EXEC_NAME): $(OBJ)
	@mkdir -p build
	$(CC) -o $@ $^ $(LIBS)

$(ODIR)/%.cc.o: $(SDIR)/%.cc $(DEP)
	@mkdir -p $(ODIR)
	$(CXX) -c -o $@ $< $(INCS) $(CXXFLAGS)

$(ODIR)/%.c.o: $(SDIR)/%.c $(DEP)
	@mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(INCS) $(CCFLAGS)

.PHONY: run
run: $(BDIR)/$(EXEC_NAME)
	@./$<

clean:
	@rm -rf $(BDIR)
