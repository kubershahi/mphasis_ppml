# Privacy-Preserving Machine Learning — SecureML (C++)
# Build from the repository root so dataset paths resolve correctly.

CXX      ?= g++
# Set EIGEN_INCLUDE if Eigen is not on the default compiler search path.
# e.g. make EIGEN_INCLUDE=/opt/homebrew/include/eigen3
EIGEN_INCLUDE ?=
EIGEN_FLAGS   := $(if $(EIGEN_INCLUDE),-I$(EIGEN_INCLUDE),)
CXXFLAGS      ?= -std=c++14 -O2 -Wall -Isecureml/include $(EIGEN_FLAGS)
BUILD    := build
BIN      := bin

COMMON_OBJS := \
	$(BUILD)/read_data.o \
	$(BUILD)/utils.o

.PHONY: all linear logistic clean

all: linear logistic

linear: $(BIN)/linear

logistic: $(BIN)/logistic

$(BIN)/linear: $(COMMON_OBJS) $(BUILD)/linear_regression.o $(BUILD)/linear.o | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN)/logistic: $(COMMON_OBJS) $(BUILD)/logistic_regression.o $(BUILD)/logistic.o | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/read_data.o: secureml/src/read_data.cpp secureml/include/read_data.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/utils.o: secureml/src/utils.cpp secureml/include/utils.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/linear_regression.o: secureml/src/linear_regression.cpp secureml/include/linear_regression.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/logistic_regression.o: secureml/src/logistic_regression.cpp secureml/include/logistic_regression.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/linear.o: secureml/src/linear.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/logistic.o: secureml/src/logistic.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD) $(BIN):
	mkdir -p $@

clean:
	rm -rf $(BUILD) $(BIN)
