#include "profiler.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
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

} // namespace

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
