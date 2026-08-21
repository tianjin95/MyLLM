CXX ?= c++
CPPFLAGS ?=
CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?=

CPPFLAGS += -I.
CXXFLAGS += -std=c++17

TARGET := chat
BUILD_DIR := build

MODEL ?= ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf
# TOKENS is the canonical name; TOKEN is accepted as a compatibility alias.
TOKENS ?= $(if $(TOKEN),$(TOKEN),32)
RAW_FLAG = $(if $(RAW),--raw,)
KV_FLAG = $(if $(KV),--kv,)
SYSTEM_FLAG = $(if $(SYSTEM),--system "$(SYSTEM)",)
PROFILE_FLAG = $(if $(NO_PROFILE),--no-profile,--profile-csv "$(PROFILE_CSV)")
PROFILE_CSV ?= output/llm_profile.csv

OBJECTS := \
	$(BUILD_DIR)/chat.o \
	$(BUILD_DIR)/llm.o \
	$(BUILD_DIR)/model.o \
	$(BUILD_DIR)/runtime.o \
	$(BUILD_DIR)/profiler.o

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/chat.o: chat.cpp chat.h kvc.h runtime.h model.h llm.h profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/llm.o: llm.cpp llm.h model.h profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/model.o: model.cpp model.h llm.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/runtime.o: runtime.cpp runtime.h model.h llm.h profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/profiler.o: profiler.cpp profiler.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

run: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --tokens "$(TOKENS)" $(RAW_FLAG) $(KV_FLAG) $(SYSTEM_FLAG) $(PROFILE_FLAG)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

help:
	@printf '%s\n' \
		'make                       Build the standalone chat executable' \
		'make run MODEL=...         Build and run with a model path' \
		'make run TOKENS=4          Limit greedy generation to 4 tokens' \
		'make run TOKEN=4           TOKEN is an alias for TOKENS' \
		'make run RAW=1             Skip ChatML wrapping' \
		'make run KV=1              Use per-turn KV-cache prefill/decode' \
		'make run SYSTEM="..."     Set the ChatML system message' \
		'make run PROFILE_CSV=...   Select a profile file under output/' \
		'make run NO_PROFILE=1      Disable profiling for the run' \
		'make clean                 Remove the executable and build objects'
