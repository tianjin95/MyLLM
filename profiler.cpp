#include "profiler.h"

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llm {
namespace {

uint64_t saturating_add(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

uint64_t saturating_mul(uint64_t left, uint64_t right) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t to_u64(size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(value);
}

uint64_t matrix_bytes(size_t rows, size_t cols) {
    return saturating_mul(saturating_mul(to_u64(rows), to_u64(cols)), 4);
}

uint64_t matrix_allocations(size_t rows) {
    // Matrix owns one vector of rows and each Vector owns one float buffer.
    return saturating_add(to_u64(rows), 1);
}

std::string csv_escape_value(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string derived_profile_path(const std::string& primary_path,
                                 const char* suffix) {
    namespace fs = std::filesystem;
    const fs::path primary(primary_path);
    const fs::path parent = primary.parent_path();
    std::string stem = primary.stem().string();
    if (stem.empty()) {
        stem = "llm_profile";
    }
    return (parent / (stem + suffix + ".csv")).string();
}

double ns_to_ms(uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1.0e6;
}

double ns_to_us(double nanoseconds) {
    return nanoseconds / 1.0e3;
}

void write_optional(std::ostream& output,
                    const std::optional<double>& value,
                    int precision = 6) {
    if (value.has_value() && std::isfinite(*value)) {
        output << std::fixed << std::setprecision(precision) << *value;
    }
}

std::optional<double> kernel_duration_ns(
    const MetalKernelProfileRecord& kernel,
    double timestamp_ns_per_tick) {
    if (!kernel.timestamp_valid || timestamp_ns_per_tick <= 0.0 ||
        kernel.end_timestamp < kernel.start_timestamp) {
        return std::nullopt;
    }
    return static_cast<double>(
               kernel.end_timestamp - kernel.start_timestamp) *
           timestamp_ns_per_tick;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct MetalOperationAggregate {
    uint64_t calls = 0;
    double gpu_ns = 0.0;
    double gap_before_ns = 0.0;
    uint64_t cpu_encode_ns = 0;
    ProfileMetrics metrics;
    uint64_t weight_bytes = 0;
    uint64_t shader_read_bytes = 0;
    double minimum_kernel_ns = std::numeric_limits<double>::infinity();
    double maximum_kernel_ns = 0.0;
};

} // namespace

MetalProfiler::MetalProfiler(const std::string& primary_csv_path,
                             bool detailed_kernel_timestamps)
    : command_csv_path_(derived_profile_path(
          primary_csv_path, "_metal_commands")),
      kernel_csv_path_(derived_profile_path(
          primary_csv_path, "_metal_kernels")),
      operation_csv_path_(derived_profile_path(
          primary_csv_path, "_metal_ops")),
      detailed_kernel_timestamps_(detailed_kernel_timestamps) {
    command_output_.open(command_csv_path_, std::ios::out | std::ios::trunc);
    kernel_output_.open(kernel_csv_path_, std::ios::out | std::ios::trunc);
    operation_output_.open(
        operation_csv_path_, std::ios::out | std::ios::trunc);
    if (!command_output_ || !kernel_output_ || !operation_output_) {
        throw std::runtime_error(
            "Cannot open one or more Metal profile CSV files");
    }

    command_output_
        << "command_index,phase,sequence_tokens,kernel_count,threadgroups,"
           "dispatched_threads,single_threadgroup_kernels,counter_requested,"
           "counter_active,counter_status,sample_count,timestamp_ns_per_tick,"
           "cpu_command_create_ms,cpu_encode_ms,cpu_encoder_sum_ms,"
           "cpu_encode_unattributed_ms,cpu_commit_ms,cpu_wait_ms,"
           "cpu_counter_resolve_ms,cpu_total_ms,"
           "commit_to_scheduled_callback_ms,"
           "commit_to_completed_callback_ms,"
           "completed_callback_to_wait_return_ms,"
           "commit_to_gpu_start_ms,gpu_start_to_kernel_start_ms,"
           "kernel_window_ms,kernel_end_to_gpu_end_ms,gpu_duration_ms,"
           "gpu_end_to_wait_return_ms,sampled_kernel_ms,sampled_gap_ms,"
           "sampled_span_ms,sampled_unattributed_ms,kernel_p50_us,"
           "kernel_p90_us,kernel_p99_us,kernel_max_us,kernel_max_operation\n";
    kernel_output_
        << "command_index,phase,sequence_tokens,order,layer_index,head_index,"
           "operation,pipeline,value_type,m,n,k,dispatch_type,grid_x,grid_y,"
           "grid_z,threads_x,threads_y,threads_z,threadgroups,"
           "dispatched_threads,cpu_encode_us,start_timestamp,end_timestamp,"
           "gpu_duration_us,gap_from_previous_us,flops,read_bytes,"
           "write_bytes,weight_bytes,shader_read_bytes,gflops,"
           "minimum_gbps,weight_gbps,shader_requested_gbps\n";
    operation_output_
        << "command_index,phase,sequence_tokens,operation,pipeline,value_type,"
           "calls,gpu_time_ms,gpu_percent,average_us,min_us,max_us,"
           "gap_before_ms,cpu_encode_ms,flops,read_bytes,write_bytes,"
           "weight_bytes,shader_read_bytes,gflops,minimum_gbps,"
           "weight_gbps,shader_requested_gbps\n";
}

MetalProfiler::~MetalProfiler() {
    flush();
}

size_t MetalProfiler::acquire_command_index() noexcept {
    return next_command_index_++;
}

bool MetalProfiler::detailed_kernel_timestamps() const noexcept {
    return detailed_kernel_timestamps_;
}

const std::string& MetalProfiler::command_csv_path() const noexcept {
    return command_csv_path_;
}

const std::string& MetalProfiler::kernel_csv_path() const noexcept {
    return kernel_csv_path_;
}

const std::string& MetalProfiler::operation_csv_path() const noexcept {
    return operation_csv_path_;
}

void MetalProfiler::write(
    MetalCommandProfileRecord command,
    const std::vector<MetalKernelProfileRecord>& kernels) {
    double sampled_kernel_ns = 0.0;
    double sampled_gap_ns = 0.0;
    double sampled_span_ns = 0.0;
    uint64_t cpu_encoder_sum_ns = 0;
    std::vector<double> durations;
    std::string maximum_operation;
    double maximum_duration_ns = 0.0;
    std::map<std::tuple<std::string, std::string, std::string>,
             MetalOperationAggregate> operations;

    uint64_t first_timestamp = 0;
    uint64_t previous_end_timestamp = 0;
    uint64_t last_timestamp = 0;
    bool have_timestamp_span = false;

    for (const MetalKernelProfileRecord& kernel : kernels) {
        cpu_encoder_sum_ns = saturating_add(
            cpu_encoder_sum_ns, kernel.cpu_encode_ns);
        const std::optional<double> duration = kernel_duration_ns(
            kernel, command.timestamp_ns_per_tick);
        std::optional<double> gap;
        if (kernel.timestamp_valid && command.timestamp_ns_per_tick > 0.0) {
            if (!have_timestamp_span) {
                first_timestamp = kernel.start_timestamp;
                have_timestamp_span = true;
            } else if (kernel.start_timestamp >= previous_end_timestamp) {
                gap = static_cast<double>(
                          kernel.start_timestamp - previous_end_timestamp) *
                      command.timestamp_ns_per_tick;
                sampled_gap_ns += *gap;
            }
            previous_end_timestamp = kernel.end_timestamp;
            last_timestamp = kernel.end_timestamp;
        }
        if (duration.has_value()) {
            sampled_kernel_ns += *duration;
            durations.push_back(*duration);
            if (*duration > maximum_duration_ns) {
                maximum_duration_ns = *duration;
                maximum_operation = kernel.operation;
            }
        }

        const auto key = std::make_tuple(
            kernel.operation, kernel.pipeline, kernel.value_type);
        MetalOperationAggregate& aggregate = operations[key];
        aggregate.calls = saturating_add(aggregate.calls, 1);
        aggregate.cpu_encode_ns = saturating_add(
            aggregate.cpu_encode_ns, kernel.cpu_encode_ns);
        aggregate.metrics += kernel.metrics;
        aggregate.weight_bytes = saturating_add(
            aggregate.weight_bytes, kernel.weight_bytes);
        aggregate.shader_read_bytes = saturating_add(
            aggregate.shader_read_bytes, kernel.shader_read_bytes);
        if (duration.has_value()) {
            aggregate.gpu_ns += *duration;
            aggregate.minimum_kernel_ns = std::min(
                aggregate.minimum_kernel_ns, *duration);
            aggregate.maximum_kernel_ns = std::max(
                aggregate.maximum_kernel_ns, *duration);
        }
        if (gap.has_value()) {
            aggregate.gap_before_ns += *gap;
        }

        kernel_output_ << command.command_index << ','
                       << csv_escape_value(command.phase) << ','
                       << command.sequence_tokens << ',' << kernel.order << ',';
        if (kernel.layer_index.has_value()) {
            kernel_output_ << *kernel.layer_index;
        }
        kernel_output_ << ',';
        if (kernel.head_index.has_value()) {
            kernel_output_ << *kernel.head_index;
        }
        kernel_output_ << ',' << csv_escape_value(kernel.operation)
                       << ',' << csv_escape_value(kernel.pipeline)
                       << ',' << csv_escape_value(kernel.value_type)
                       << ',' << kernel.m << ',' << kernel.n << ',' << kernel.k
                       << ',' << csv_escape_value(kernel.dispatch_type)
                       << ',' << kernel.grid_x << ',' << kernel.grid_y
                       << ',' << kernel.grid_z << ',' << kernel.threads_x
                       << ',' << kernel.threads_y << ',' << kernel.threads_z
                       << ',' << kernel.threadgroups
                       << ',' << kernel.dispatched_threads
                       << ',' << std::fixed << std::setprecision(3)
                       << ns_to_us(static_cast<double>(kernel.cpu_encode_ns))
                       << ',';
        if (kernel.timestamp_valid) {
            kernel_output_ << kernel.start_timestamp;
        }
        kernel_output_ << ',';
        if (kernel.timestamp_valid) {
            kernel_output_ << kernel.end_timestamp;
        }
        kernel_output_ << ',';
        if (duration.has_value()) {
            kernel_output_ << ns_to_us(*duration);
        }
        kernel_output_ << ',';
        if (gap.has_value()) {
            kernel_output_ << ns_to_us(*gap);
        }
        kernel_output_ << ',' << kernel.metrics.flops
                       << ',' << kernel.metrics.read_bytes
                       << ',' << kernel.metrics.write_bytes
                       << ',' << kernel.weight_bytes
                       << ',' << kernel.shader_read_bytes << ',';
        const double seconds = duration.has_value() ? *duration / 1.0e9 : 0.0;
        if (seconds > 0.0) {
            kernel_output_ << std::setprecision(3)
                           << static_cast<double>(kernel.metrics.flops) /
                                  seconds / 1.0e9
                           << ','
                           << static_cast<double>(saturating_add(
                                  kernel.metrics.read_bytes,
                                  kernel.metrics.write_bytes)) /
                                  seconds / 1.0e9
                           << ','
                           << static_cast<double>(kernel.weight_bytes) /
                                  seconds / 1.0e9
                           << ','
                           << static_cast<double>(kernel.shader_read_bytes) /
                                  seconds / 1.0e9;
        } else {
            kernel_output_ << ",,,";
        }
        kernel_output_ << '\n';
    }

    if (have_timestamp_span && last_timestamp >= first_timestamp) {
        sampled_span_ns = static_cast<double>(
                              last_timestamp - first_timestamp) *
                          command.timestamp_ns_per_tick;
    }
    const double sampled_unattributed_ns =
        command.gpu_duration_ms.has_value()
            ? *command.gpu_duration_ms * 1.0e6 - sampled_span_ns
            : 0.0;
    const uint64_t encode_unattributed_ns = command.cpu_encode_ns >=
            cpu_encoder_sum_ns
        ? command.cpu_encode_ns - cpu_encoder_sum_ns
        : 0;

    command_output_ << command.command_index << ','
                    << csv_escape_value(command.phase) << ','
                    << command.sequence_tokens << ',' << command.kernel_count
                    << ',' << command.threadgroup_count
                    << ',' << command.dispatched_threads
                    << ',' << command.single_threadgroup_kernels
                    << ',' << (command.counter_requested ? 1 : 0)
                    << ',' << (command.counter_active ? 1 : 0)
                    << ',' << csv_escape_value(command.counter_status)
                    << ',' << command.sample_count
                    << ',' << std::fixed << std::setprecision(9)
                    << command.timestamp_ns_per_tick
                    << ',' << std::setprecision(6)
                    << ns_to_ms(command.cpu_command_create_ns)
                    << ',' << ns_to_ms(command.cpu_encode_ns)
                    << ',' << ns_to_ms(cpu_encoder_sum_ns)
                    << ',' << ns_to_ms(encode_unattributed_ns)
                    << ',' << ns_to_ms(command.cpu_commit_ns)
                    << ',' << ns_to_ms(command.cpu_wait_ns)
                    << ',' << ns_to_ms(command.cpu_counter_resolve_ns)
                    << ',' << ns_to_ms(command.cpu_total_ns) << ',';
    write_optional(
        command_output_, command.commit_to_scheduled_callback_ms);
    command_output_ << ',';
    write_optional(
        command_output_, command.commit_to_completed_callback_ms);
    command_output_ << ',';
    write_optional(
        command_output_, command.completed_callback_to_wait_return_ms);
    command_output_ << ',';
    write_optional(command_output_, command.commit_to_gpu_start_ms);
    command_output_ << ',';
    write_optional(command_output_, command.gpu_start_to_kernel_start_ms);
    command_output_ << ',';
    write_optional(command_output_, command.kernel_window_ms);
    command_output_ << ',';
    write_optional(command_output_, command.kernel_end_to_gpu_end_ms);
    command_output_ << ',';
    write_optional(command_output_, command.gpu_duration_ms);
    command_output_ << ',';
    write_optional(command_output_, command.gpu_end_to_wait_return_ms);
    command_output_ << ',' << sampled_kernel_ns / 1.0e6
                    << ',' << sampled_gap_ns / 1.0e6
                    << ',' << sampled_span_ns / 1.0e6
                    << ',' << sampled_unattributed_ns / 1.0e6
                    << ',' << ns_to_us(percentile(durations, 0.50))
                    << ',' << ns_to_us(percentile(durations, 0.90))
                    << ',' << ns_to_us(percentile(durations, 0.99))
                    << ',' << ns_to_us(maximum_duration_ns)
                    << ',' << csv_escape_value(maximum_operation) << '\n';

    for (const auto& entry : operations) {
        const auto& key = entry.first;
        const MetalOperationAggregate& aggregate = entry.second;
        const double seconds = aggregate.gpu_ns / 1.0e9;
        const uint64_t minimum_bytes = saturating_add(
            aggregate.metrics.read_bytes, aggregate.metrics.write_bytes);
        operation_output_ << command.command_index << ','
                          << csv_escape_value(command.phase) << ','
                          << command.sequence_tokens << ','
                          << csv_escape_value(std::get<0>(key)) << ','
                          << csv_escape_value(std::get<1>(key)) << ','
                          << csv_escape_value(std::get<2>(key)) << ','
                          << aggregate.calls << ',' << std::fixed
                          << std::setprecision(6)
                          << aggregate.gpu_ns / 1.0e6 << ','
                          << (sampled_kernel_ns > 0.0
                                  ? aggregate.gpu_ns / sampled_kernel_ns * 100.0
                                  : 0.0)
                          << ','
                          << (aggregate.calls != 0
                                  ? aggregate.gpu_ns /
                                        static_cast<double>(aggregate.calls) /
                                        1.0e3
                                  : 0.0)
                          << ','
                          << (std::isfinite(aggregate.minimum_kernel_ns)
                                  ? aggregate.minimum_kernel_ns / 1.0e3
                                  : 0.0)
                          << ',' << aggregate.maximum_kernel_ns / 1.0e3
                          << ',' << aggregate.gap_before_ns / 1.0e6
                          << ',' << ns_to_ms(aggregate.cpu_encode_ns)
                          << ',' << aggregate.metrics.flops
                          << ',' << aggregate.metrics.read_bytes
                          << ',' << aggregate.metrics.write_bytes
                          << ',' << aggregate.weight_bytes
                          << ',' << aggregate.shader_read_bytes << ',';
        if (seconds > 0.0) {
            operation_output_
                << static_cast<double>(aggregate.metrics.flops) /
                       seconds / 1.0e9
                << ',' << static_cast<double>(minimum_bytes) /
                       seconds / 1.0e9
                << ',' << static_cast<double>(aggregate.weight_bytes) /
                       seconds / 1.0e9
                << ',' << static_cast<double>(aggregate.shader_read_bytes) /
                       seconds / 1.0e9;
        } else {
            operation_output_ << ",,,";
        }
        operation_output_ << '\n';
    }
    flush();
}

void MetalProfiler::flush() {
    command_output_.flush();
    kernel_output_.flush();
    operation_output_.flush();
}

ProfileMetrics operator+(ProfileMetrics left, const ProfileMetrics& right) {
    left += right;
    return left;
}

ProfileMetrics& operator+=(ProfileMetrics& left, const ProfileMetrics& right) {
    left.flops = saturating_add(left.flops, right.flops);
    left.read_bytes = saturating_add(left.read_bytes, right.read_bytes);
    left.write_bytes = saturating_add(left.write_bytes, right.write_bytes);
    left.temporary_bytes = saturating_add(
        left.temporary_bytes, right.temporary_bytes);
    left.allocations = saturating_add(left.allocations, right.allocations);
    return left;
}

ProfileMetrics profile_gemm_metrics(size_t m,
                                    size_t n,
                                    size_t k) {
    ProfileMetrics result;
    const uint64_t mm = to_u64(m);
    const uint64_t nn = to_u64(n);
    const uint64_t kk = to_u64(k);
    const uint64_t mn = saturating_mul(mm, nn);
    result.flops = saturating_mul(saturating_mul(mn, kk), 2);
    result.read_bytes = saturating_add(
        matrix_bytes(m, k), matrix_bytes(k, n));
    result.write_bytes = matrix_bytes(m, n);
    result.temporary_bytes = result.write_bytes;
    result.allocations = matrix_allocations(m);
    return result;
}

ProfileMetrics profile_gemmb_metrics(size_t m,
                                     size_t n,
                                     size_t k) {
    ProfileMetrics result = profile_gemm_metrics(m, n, k);
    result.flops = saturating_add(
        result.flops, saturating_mul(to_u64(m), to_u64(n)));
    result.read_bytes = saturating_add(
        result.read_bytes, saturating_mul(to_u64(n), 4));
    return result;
}

ProfileMetrics profile_gemv_metrics(size_t input_size,
                                    size_t output_size) {
    ProfileMetrics result;
    const uint64_t input = to_u64(input_size);
    const uint64_t output = to_u64(output_size);
    result.flops = saturating_mul(saturating_mul(input, output), 2);
    result.read_bytes = saturating_add(
        saturating_mul(saturating_mul(input, output), 4),
        saturating_mul(input, 4));
    result.write_bytes = saturating_mul(output, 4);
    result.temporary_bytes = result.write_bytes;
    result.allocations = 1;
    return result;
}

ProfileMetrics profile_elementwise_metrics(size_t elements,
                                           uint64_t flops_per_element,
                                           uint64_t read_arrays,
                                           uint64_t write_arrays,
                                           uint64_t temporary_bytes,
                                           uint64_t allocations) {
    ProfileMetrics result;
    const uint64_t count = to_u64(elements);
    result.flops = saturating_mul(count, flops_per_element);
    result.read_bytes = saturating_mul(
        saturating_mul(count, read_arrays), 4);
    result.write_bytes = saturating_mul(
        saturating_mul(count, write_arrays), 4);
    result.temporary_bytes = temporary_bytes;
    result.allocations = allocations;
    return result;
}

ProfileMetrics profile_copy_metrics(uint64_t bytes, uint64_t allocations) {
    ProfileMetrics result;
    result.read_bytes = bytes;
    result.write_bytes = bytes;
    result.temporary_bytes = bytes;
    result.allocations = allocations;
    return result;
}

void Profiler::add_aggregate(Profiler::Aggregate& destination,
                             uint64_t elapsed_ns,
                             const ProfileMetrics& metrics) {
    destination.calls = saturating_add(destination.calls, 1);
    destination.elapsed_ns = saturating_add(destination.elapsed_ns, elapsed_ns);
    destination.metrics.flops = saturating_add(
        destination.metrics.flops, metrics.flops);
    destination.metrics.read_bytes = saturating_add(
        destination.metrics.read_bytes, metrics.read_bytes);
    destination.metrics.write_bytes = saturating_add(
        destination.metrics.write_bytes, metrics.write_bytes);
    destination.metrics.temporary_bytes = saturating_add(
        destination.metrics.temporary_bytes, metrics.temporary_bytes);
    destination.metrics.allocations = saturating_add(
        destination.metrics.allocations, metrics.allocations);
}

void Profiler::add_map(Profiler::AggregateMap& destination,
                       const std::string& name,
                       uint64_t elapsed_ns,
                       const ProfileMetrics& metrics) {
    add_aggregate(destination[name], elapsed_ns, metrics);
}

bool Profiler::is_inclusive_scope(const std::string& name) {
    constexpr const char suffix[] = ".total";
    constexpr size_t suffix_length = sizeof(suffix) - 1;
    return name.size() >= suffix_length &&
           name.compare(name.size() - suffix_length, suffix_length, suffix) == 0;
}

Profiler::Scope::Scope(Profiler* profiler,
                       std::string name,
                       ProfileMetrics metrics)
    : profiler_(profiler),
      name_(std::move(name)),
      metrics_(metrics),
      start_(std::chrono::steady_clock::now()) {}

Profiler::Scope::~Scope() {
    if (profiler_ == nullptr) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_);
    profiler_->record_scope(
        name_, static_cast<uint64_t>(elapsed.count()), metrics_);
}

Profiler::ForwardScope::ForwardScope(Profiler* profiler,
                                     size_t sequence_length)
    : profiler_(profiler) {
    if (profiler_ != nullptr) {
        profiler_->begin_forward(sequence_length);
    }
}

Profiler::ForwardScope::~ForwardScope() {
    if (profiler_ != nullptr) {
        profiler_->finish_forward();
    }
}

Profiler::LayerScope::LayerScope(Profiler* profiler, size_t layer_index)
    : profiler_(profiler) {
    if (profiler_ != nullptr) {
        profiler_->begin_layer(layer_index);
    }
}

Profiler::LayerScope::~LayerScope() {
    if (profiler_ != nullptr) {
        profiler_->finish_layer();
    }
}

Profiler::Profiler(const std::string& csv_path) : csv_path_(csv_path) {
    if (csv_path.empty()) {
        throw std::invalid_argument("Profile CSV path cannot be empty");
    }
    output_.open(csv_path, std::ios::out | std::ios::trunc);
    if (!output_) {
        throw std::runtime_error("Cannot open profile CSV: " + csv_path);
    }
    output_ << "record_type,forward_index,sequence_tokens,layer_index,stage,"
               "calls,time_ms,flops,read_bytes,write_bytes,temporary_bytes,"
               "logical_bytes,estimated_bytes_with_temporaries,allocations,"
               "gflops,logical_gbps,estimated_gbps,arithmetic_intensity,"
               "arithmetic_intensity_with_temporaries\n";
}

Profiler::~Profiler() {
    if (layer_active_) {
        finish_layer();
    }
    if (forward_active_) {
        finish_forward();
    }
    write_final_summary();
}

void Profiler::flush() {
    output_.flush();
}

void Profiler::begin_forward(size_t sequence_length) {
    if (forward_active_) {
        finish_forward();
    }
    sequence_length_ = sequence_length;
    current_forward_.clear();
    current_forward_metrics_ = {};
    current_layers_.clear();
    forward_start_ = std::chrono::steady_clock::now();
    forward_active_ = true;
}

void Profiler::finish_forward() {
    if (!forward_active_) {
        return;
    }
    if (layer_active_) {
        finish_layer();
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - forward_start_);
    const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed.count());
    add_map(global_, "forward.total", elapsed_ns, current_forward_metrics_);

    Aggregate forward_total;
    forward_total.calls = 1;
    forward_total.elapsed_ns = elapsed_ns;
    forward_total.metrics = current_forward_metrics_;
    write_csv_row("forward_total", forward_index_, sequence_length_,
                  std::nullopt, "forward.total", forward_total);
    for (const auto& entry : current_forward_) {
        write_csv_row("forward_stage", forward_index_, sequence_length_,
                      std::nullopt, entry.first, entry.second);
    }
    for (const LayerSummary& layer : current_layers_) {
        Aggregate layer_total;
        layer_total.calls = 1;
        layer_total.elapsed_ns = layer.elapsed_ns;
        layer_total.metrics = layer.metrics;
        write_csv_row("layer_total", forward_index_, sequence_length_,
                      layer.layer_index, "layer.total", layer_total);
        for (const auto& entry : layer.stages) {
            write_csv_row("layer_stage", forward_index_, sequence_length_,
                          layer.layer_index, entry.first, entry.second);
        }
    }
    output_.flush();

    current_forward_.clear();
    current_forward_metrics_ = {};
    current_layers_.clear();
    forward_active_ = false;
    ++forward_index_;
}

void Profiler::begin_layer(size_t layer_index) {
    if (layer_active_) {
        finish_layer();
    }
    current_layer_index_ = layer_index;
    current_layer_.clear();
    current_layer_metrics_ = {};
    layer_start_ = std::chrono::steady_clock::now();
    layer_active_ = true;
}

void Profiler::finish_layer() {
    if (!layer_active_) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - layer_start_);
    const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed.count());
    add_map(global_, "layer.total", elapsed_ns, current_layer_metrics_);

    LayerSummary summary;
    summary.layer_index = current_layer_index_;
    summary.elapsed_ns = elapsed_ns;
    summary.metrics = current_layer_metrics_;
    summary.stages = current_layer_;
    if (forward_active_) {
        current_layers_.push_back(std::move(summary));
    }
    current_layer_.clear();
    current_layer_metrics_ = {};
    layer_active_ = false;
}

void Profiler::record_scope(const std::string& name,
                            uint64_t elapsed_ns,
                            const ProfileMetrics& metrics) {
    add_map(global_, name, elapsed_ns, metrics);
    if (forward_active_) {
        add_map(current_forward_, name, elapsed_ns, metrics);
    }
    if (layer_active_) {
        add_map(current_layer_, name, elapsed_ns, metrics);
    }
    if (!is_inclusive_scope(name)) {
        if (forward_active_) {
            current_forward_metrics_ += metrics;
        }
        if (layer_active_) {
            current_layer_metrics_ += metrics;
        }
    }
}

std::string Profiler::csv_escape(const std::string& value) {
    bool needs_quotes = false;
    for (const char character : value) {
        if (character == ',' || character == '"' || character == '\n' ||
            character == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

void Profiler::write_csv_row(const std::string& record_type,
                             std::optional<size_t> forward_index,
                             std::optional<size_t> sequence_length,
                             std::optional<size_t> layer_index,
                             const std::string& stage,
                             const Aggregate& aggregate) {
    const double seconds = static_cast<double>(aggregate.elapsed_ns) / 1.0e9;
    const uint64_t logical_bytes = saturating_add(
        aggregate.metrics.read_bytes, aggregate.metrics.write_bytes);
    const uint64_t estimated_bytes = saturating_add(
        logical_bytes, aggregate.metrics.temporary_bytes);
    const double gflops = seconds > 0.0
        ? static_cast<double>(aggregate.metrics.flops) / seconds / 1.0e9
        : 0.0;
    const double logical_gbps = seconds > 0.0
        ? static_cast<double>(logical_bytes) / seconds / 1.0e9
        : 0.0;
    const double estimated_gbps = seconds > 0.0
        ? static_cast<double>(estimated_bytes) / seconds / 1.0e9
        : 0.0;
    const double arithmetic_intensity = logical_bytes > 0
        ? static_cast<double>(aggregate.metrics.flops) /
              static_cast<double>(logical_bytes)
        : 0.0;
    const double arithmetic_intensity_with_temporaries = estimated_bytes > 0
        ? static_cast<double>(aggregate.metrics.flops) /
              static_cast<double>(estimated_bytes)
        : 0.0;

    output_ << csv_escape(record_type) << ',';
    if (forward_index.has_value()) {
        output_ << *forward_index;
    }
    output_ << ',';
    if (sequence_length.has_value()) {
        output_ << *sequence_length;
    }
    output_ << ',';
    if (layer_index.has_value()) {
        output_ << *layer_index;
    }
    output_ << ',' << csv_escape(stage)
            << ',' << aggregate.calls
            << ',' << std::fixed << std::setprecision(3)
            << static_cast<double>(aggregate.elapsed_ns) / 1.0e6
            << ',' << aggregate.metrics.flops
            << ',' << aggregate.metrics.read_bytes
            << ',' << aggregate.metrics.write_bytes
            << ',' << aggregate.metrics.temporary_bytes
            << ',' << logical_bytes
            << ',' << estimated_bytes
            << ',' << aggregate.metrics.allocations
            << ',' << std::setprecision(3) << gflops
            << ',' << logical_gbps
            << ',' << estimated_gbps
            << ',' << arithmetic_intensity
            << ',' << arithmetic_intensity_with_temporaries
            << '\n';
}

void Profiler::write_final_summary() {
    if (final_summary_written_ || !output_) {
        return;
    }
    for (const auto& entry : global_) {
        write_csv_row("global", std::nullopt, std::nullopt, std::nullopt,
                      entry.first, entry.second);
    }
    output_.flush();
    final_summary_written_ = true;
}

} // namespace llm
