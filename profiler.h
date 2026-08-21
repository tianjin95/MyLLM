#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// These are logical tensor traffic estimates. They describe the data needed
// by an operator, not hardware-observed DRAM traffic.
struct ProfileMetrics {
    // These values are logical estimates derived from tensor shapes. They are
    // useful for comparing operators, but are not hardware performance-counter
    // measurements.
    uint64_t flops = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    // Estimated temporary/output storage associated with intermediate objects.
    // It is reported separately and is not part of the primary logical traffic
    // denominator used for estimated_gbps and arithmetic_intensity.
    uint64_t temporary_bytes = 0;
    uint64_t allocations = 0;
};

ProfileMetrics operator+(ProfileMetrics left, const ProfileMetrics& right);
ProfileMetrics& operator+=(ProfileMetrics& left, const ProfileMetrics& right);

ProfileMetrics profile_gemm_metrics(size_t m,
                                    size_t n,
                                    size_t k);
ProfileMetrics profile_gemmb_metrics(size_t m,
                                     size_t n,
                                     size_t k);
ProfileMetrics profile_gemv_metrics(size_t input_size,
                                    size_t output_size);
ProfileMetrics profile_elementwise_metrics(size_t elements,
                                           uint64_t flops_per_element,
                                           uint64_t read_arrays,
                                           uint64_t write_arrays,
                                           uint64_t temporary_bytes = 0,
                                           uint64_t allocations = 0);
ProfileMetrics profile_copy_metrics(uint64_t bytes,
                                    uint64_t allocations = 0);

class Profiler {
public:
    class Scope {
    public:
        Scope(Profiler* profiler,
              std::string name,
              ProfileMetrics metrics = {});
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        Profiler* profiler_ = nullptr;
        std::string name_;
        ProfileMetrics metrics_;
        std::chrono::steady_clock::time_point start_;
    };

    class ForwardScope {
    public:
        ForwardScope(Profiler* profiler, size_t sequence_length);
        ~ForwardScope();

        ForwardScope(const ForwardScope&) = delete;
        ForwardScope& operator=(const ForwardScope&) = delete;

    private:
        Profiler* profiler_ = nullptr;
    };

    class LayerScope {
    public:
        LayerScope(Profiler* profiler, size_t layer_index);
        ~LayerScope();

        LayerScope(const LayerScope&) = delete;
        LayerScope& operator=(const LayerScope&) = delete;

    private:
        Profiler* profiler_ = nullptr;
    };

    explicit Profiler(const std::string& csv_path);
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    void flush();

private:
    struct Aggregate {
        uint64_t calls = 0;
        uint64_t elapsed_ns = 0;
        ProfileMetrics metrics;
    };

    struct LayerSummary {
        size_t layer_index = 0;
        uint64_t elapsed_ns = 0;
        ProfileMetrics metrics;
        std::map<std::string, Aggregate> stages;
    };

    using AggregateMap = std::map<std::string, Aggregate>;

    void begin_forward(size_t sequence_length);
    void finish_forward();
    void begin_layer(size_t layer_index);
    void finish_layer();
    void record_scope(const std::string& name,
                      uint64_t elapsed_ns,
                      const ProfileMetrics& metrics);
    static void add_aggregate(Aggregate& destination,
                              uint64_t elapsed_ns,
                              const ProfileMetrics& metrics);
    static void add_map(AggregateMap& destination,
                        const std::string& name,
                        uint64_t elapsed_ns,
                        const ProfileMetrics& metrics);
    static bool is_inclusive_scope(const std::string& name);
    static std::string csv_escape(const std::string& value);
    void write_csv_row(const std::string& record_type,
                       std::optional<size_t> forward_index,
                       std::optional<size_t> sequence_length,
                       std::optional<size_t> layer_index,
                       const std::string& stage,
                       const Aggregate& aggregate);
    void write_final_summary();

    std::ofstream output_;
    std::string csv_path_;
    AggregateMap global_;
    AggregateMap current_forward_;
    AggregateMap current_layer_;
    ProfileMetrics current_forward_metrics_;
    ProfileMetrics current_layer_metrics_;
    std::vector<LayerSummary> current_layers_;
    size_t forward_index_ = 0;
    size_t sequence_length_ = 0;
    size_t current_layer_index_ = 0;
    bool forward_active_ = false;
    bool layer_active_ = false;
    bool final_summary_written_ = false;
    std::chrono::steady_clock::time_point forward_start_;
    std::chrono::steady_clock::time_point layer_start_;

    friend class Scope;
    friend class ForwardScope;
    friend class LayerScope;
};

} // namespace llm
