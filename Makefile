CXX ?= c++
OBJCXX ?= $(CXX)
CPPFLAGS ?=
CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?=

CPPFLAGS += -I.
CXXFLAGS += -std=c++17

TARGET := chat
BUILD_DIR := build
METAL_OBJECT := $(BUILD_DIR)/metal_llm.o
METAL_LDLIBS := -framework Metal -framework Foundation
METAL_CXXFLAGS := -fobjc-arc
LINKER := $(OBJCXX)

MODEL ?= ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf
# TOKENS is the canonical name; TOKEN is accepted as a compatibility alias.
TOKENS ?= $(if $(TOKEN),$(TOKEN),32)
MAX_SEQUENCE ?=
RAW_FLAG = $(if $(RAW),--raw,)
GPU_FLAG = $(if $(GPU),--gpu,)
MAX_SEQUENCE_FLAG = $(if $(MAX_SEQUENCE),--max-sequence "$(MAX_SEQUENCE)",)
SYSTEM_FLAG = $(if $(SYSTEM),--system "$(SYSTEM)",)
PROFILE_FLAG = $(if $(NO_PROFILE),--no-profile,--profile-csv "$(PROFILE_CSV)")
PROFILE_CSV ?= output/llm_profile.csv

OBJECTS := \
	$(BUILD_DIR)/chat.o \
	$(BUILD_DIR)/cpu_llm.o \
	$(BUILD_DIR)/model.o \
	$(BUILD_DIR)/profiler.o \
	$(METAL_OBJECT)

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINKER) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) $(METAL_LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/chat.o: chat.cpp chat.h memory_stats.h cpu_llm.h metal_llm.h model.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/cpu_llm.o: cpu_llm.cpp cpu_llm.h memory_stats.h model.h profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/model.o: model.cpp model.h cpu_llm.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/profiler.o: profiler.cpp profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/metal_llm.o: metal_llm.mm metal_llm.h memory_stats.h metal_llm.metal model.h cpu_llm.h profiler.h | $(BUILD_DIR)
	$(OBJCXX) $(CPPFLAGS) $(CXXFLAGS) $(METAL_CXXFLAGS) -MMD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

run: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --tokens "$(TOKENS)" $(MAX_SEQUENCE_FLAG) $(RAW_FLAG) $(GPU_FLAG) $(SYSTEM_FLAG) $(PROFILE_FLAG)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

help:
	@printf '%s\n' \
		'make                       Build the standalone chat executable' \
		'make run MODEL=...         Build and run with a model path' \
		'make run TOKENS=4          Limit greedy generation to 4 tokens' \
		'make run TOKEN=4           TOKEN is an alias for TOKENS' \
		'make run MAX_SEQUENCE=2048  Set the fixed model sequence capacity' \
		'make run RAW=1             Skip ChatML wrapping' \
		'make run GPU=1              Require the Metal GPU backend' \
		'make run SYSTEM="..."     Set the ChatML system message' \
		'make run PROFILE_CSV=...   Select a profile file under output/' \
		'make run NO_PROFILE=1      Disable profiling for the run' \
		'make clean                 Remove the executable and build objects'
