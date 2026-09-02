#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
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

// Hardware-facing Metal records are kept separate from the generic operator
// profiler. The command record is inexpensive and is emitted for every Metal
// forward. Kernel records are populated only by the opt-in counter mode.
struct MetalKernelProfileRecord {
    size_t order = 0;
    std::optional<size_t> layer_index;
    std::optional<size_t> head_index;
    std::string operation;
    std::string pipeline;
    std::string value_type;
    std::string dispatch_type;
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    uint64_t grid_x = 0;
    uint64_t grid_y = 0;
    uint64_t grid_z = 0;
    uint64_t threads_x = 0;
    uint64_t threads_y = 0;
    uint64_t threads_z = 0;
    uint64_t threadgroups = 0;
    uint64_t dispatched_threads = 0;
    uint64_t cpu_encode_ns = 0;
    size_t start_sample_index = std::numeric_limits<size_t>::max();
    size_t end_sample_index = std::numeric_limits<size_t>::max();
    uint64_t start_timestamp = 0;
    uint64_t end_timestamp = 0;
    bool timestamp_valid = false;
    ProfileMetrics metrics;
    uint64_t weight_bytes = 0;
    uint64_t shader_read_bytes = 0;
};

struct MetalCommandProfileRecord {
    size_t command_index = 0;
    std::string phase;
    size_t sequence_tokens = 0;
    uint64_t kernel_count = 0;
    uint64_t threadgroup_count = 0;
    uint64_t dispatched_threads = 0;
    uint64_t single_threadgroup_kernels = 0;
    bool counter_requested = false;
    bool counter_active = false;
    std::string counter_status;
    uint64_t sample_count = 0;
    double timestamp_ns_per_tick = 0.0;
    uint64_t cpu_command_create_ns = 0;
    uint64_t cpu_encode_ns = 0;
    uint64_t cpu_commit_ns = 0;
    uint64_t cpu_wait_ns = 0;
    uint64_t cpu_counter_resolve_ns = 0;
    uint64_t cpu_total_ns = 0;
    std::optional<double> commit_to_scheduled_callback_ms;
    std::optional<double> commit_to_completed_callback_ms;
    std::optional<double> completed_callback_to_wait_return_ms;
    std::optional<double> commit_to_gpu_start_ms;
    std::optional<double> gpu_start_to_kernel_start_ms;
    std::optional<double> kernel_window_ms;
    std::optional<double> kernel_end_to_gpu_end_ms;
    std::optional<double> gpu_duration_ms;
    std::optional<double> gpu_end_to_wait_return_ms;
};

class MetalProfiler {
public:
    MetalProfiler(const std::string& primary_csv_path,
                  bool detailed_kernel_timestamps);
    ~MetalProfiler();

    MetalProfiler(const MetalProfiler&) = delete;
    MetalProfiler& operator=(const MetalProfiler&) = delete;

    size_t acquire_command_index() noexcept;
    bool detailed_kernel_timestamps() const noexcept;
    const std::string& command_csv_path() const noexcept;
    const std::string& kernel_csv_path() const noexcept;
    const std::string& operation_csv_path() const noexcept;

    void write(MetalCommandProfileRecord command,
               const std::vector<MetalKernelProfileRecord>& kernels);
    void flush();

private:
    std::ofstream command_output_;
    std::ofstream kernel_output_;
    std::ofstream operation_output_;
    std::string command_csv_path_;
    std::string kernel_csv_path_;
    std::string operation_csv_path_;
    size_t next_command_index_ = 0;
    bool detailed_kernel_timestamps_ = false;
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
