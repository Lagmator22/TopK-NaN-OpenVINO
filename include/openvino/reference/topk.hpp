// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// POC: Enhanced reference implementation for TopK with NaN handling.
//
// In the real implementation this would EXTEND the existing file:
//   src/core/reference/include/openvino/reference/topk.hpp
//
// Strategy: add a new overload of the topk() template that accepts TopKNanMode.
// The existing overload (without nan_mode) remains UNCHANGED for full backward compat.
// When nan_mode == NONE, the new overload delegates to the original comparator
// with zero overhead (no isnan checks in the hot loop).

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include "../op/topk_nan_mode.hpp"
#include "../op/topk_v17.hpp"

namespace ov {
namespace reference {

// ============================================================================
// EXISTING comparators (unchanged from current OpenVINO master)
// ============================================================================

/// Original comparator from OpenVINO reference/topk.hpp.
/// This used to be lambda expressions but MSVC had difficulty compiling it.
template <bool D, typename T, typename U>
inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    if (std::get<0>(a) != std::get<0>(b)) {
        return D ? std::get<0>(a) > std::get<0>(b) : std::get<0>(a) < std::get<0>(b);
    } else {
        return std::get<1>(a) < std::get<1>(b);
    }
}

template <typename T, typename U>
inline bool compare_indices_ascending(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    return std::get<1>(a) < std::get<1>(b);
}

// ============================================================================
// NEW: NaN-aware comparator for v17::TopK
// ============================================================================

/// NaN-aware comparator that satisfies strict weak ordering for all modes.
///
/// When nan_mode == NAN_AS_SMALLEST:
///   NaN < -inf < ... < +inf    (NaN treated as the smallest possible value)
/// When nan_mode == NAN_AS_LARGEST:
///   -inf < ... < +inf < NaN    (NaN treated as the largest possible value)
///
/// For equal values (including NaN == NaN under the mode's ordering),
/// ties are broken by ascending index for stability.
template <bool IsMax, typename T, typename U>
class NanAwareComparator {
public:
    explicit NanAwareComparator(ov::op::TopKNanMode mode) : m_mode(mode) {}

    bool operator()(const std::tuple<T, U>& a, const std::tuple<T, U>& b) const {
        const T val_a = std::get<0>(a);
        const T val_b = std::get<0>(b);

        const bool a_nan = std::isnan(val_a);
        const bool b_nan = std::isnan(val_b);

        if (a_nan || b_nan) {
            if (a_nan && b_nan) {
                // Both NaN: use index for stable ordering
                return std::get<1>(a) < std::get<1>(b);
            }
            if (m_mode == ov::op::TopKNanMode::NAN_AS_SMALLEST) {
                // NaN is "smallest": in MAX mode, NaN loses; in MIN mode, NaN wins
                return a_nan ? !IsMax : IsMax;
            } else {
                // NAN_AS_LARGEST: in MAX mode, NaN wins; in MIN mode, NaN loses
                return a_nan ? IsMax : !IsMax;
            }
        }

        // Normal path: no NaN
        if (val_a != val_b) {
            return IsMax ? val_a > val_b : val_a < val_b;
        }
        return std::get<1>(a) < std::get<1>(b);  // Stable tie-break
    }

private:
    ov::op::TopKNanMode m_mode;
};

// ============================================================================
// EXISTING topk() — UNCHANGED (backward compatibility)
// ============================================================================

/// Original reference topk (no nan_mode). Identical to current OpenVINO master.
template <typename T, typename U>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const size_t in_axis_size,
          const size_t k,
          const bool compute_max,
          const ov::op::TopKSortType sort = ov::op::TopKSortType::NONE) {
    // Create workspace
    std::vector<std::tuple<T, U>> workspace(in_axis_size);
    for (size_t i = 0; i < in_axis_size; ++i) {
        std::get<0>(workspace[i]) = arg[i];
        std::get<1>(workspace[i]) = static_cast<U>(i);
    }

    const auto cmp_func = compute_max ? compare_max<true, T, U> : compare_max<false, T, U>;

    std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp_func);

    // Apply sort if requested
    switch (sort) {
    case ov::op::TopKSortType::SORT_VALUES:
        std::sort(workspace.begin(), workspace.begin() + k, cmp_func);
        break;
    case ov::op::TopKSortType::SORT_INDICES:
        std::sort(workspace.begin(), workspace.begin() + k, compare_indices_ascending<T, U>);
        break;
    default:
        break;
    }

    for (size_t j = 0; j < k; ++j) {
        out_values[j] = std::get<0>(workspace[j]);
        out_indices[j] = std::get<1>(workspace[j]);
    }
}

// ============================================================================
// NEW: topk() overload with nan_mode for v17::TopK
// ============================================================================

/// Enhanced reference topk with NaN handling.
///
/// When nan_mode == NONE, this uses the original comparator (zero overhead).
/// When nan_mode is NAN_AS_SMALLEST or NAN_AS_LARGEST, NaN-aware comparator is used.
///
/// The `stable` flag uses std::stable_sort instead of nth_element + sort.
template <typename T, typename U>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const size_t in_axis_size,
          const size_t k,
          const bool compute_max,
          const ov::op::TopKSortType sort,
          const ov::op::TopKNanMode nan_mode,
          const bool stable = false) {
    // Delegate to original when nan_mode is NONE (zero overhead path)
    if (nan_mode == ov::op::TopKNanMode::NONE && !stable) {
        topk(arg, out_indices, out_values, in_axis_size, k, compute_max, sort);
        return;
    }

    // Create workspace
    std::vector<std::tuple<T, U>> workspace(in_axis_size);
    for (size_t i = 0; i < in_axis_size; ++i) {
        std::get<0>(workspace[i]) = arg[i];
        std::get<1>(workspace[i]) = static_cast<U>(i);
    }

    // Select comparator
    using Entry = std::tuple<T, U>;
    std::function<bool(const Entry&, const Entry&)> cmp_func;

    if (nan_mode != ov::op::TopKNanMode::NONE) {
        if (compute_max) {
            cmp_func = NanAwareComparator<true, T, U>(nan_mode);
        } else {
            cmp_func = NanAwareComparator<false, T, U>(nan_mode);
        }
    } else {
        // NONE + stable: use original comparator
        cmp_func = compute_max ? compare_max<true, T, U> : compare_max<false, T, U>;
    }

    // Sorting strategy
    if (stable) {
        std::stable_sort(workspace.begin(), workspace.end(), cmp_func);
    } else {
        std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp_func);
    }

    // Apply secondary sort
    switch (sort) {
    case ov::op::TopKSortType::SORT_VALUES:
        if (!stable) {
            std::sort(workspace.begin(), workspace.begin() + k, cmp_func);
        }
        // stable_sort already sorted by value
        break;
    case ov::op::TopKSortType::SORT_INDICES:
        std::sort(workspace.begin(), workspace.begin() + k, compare_indices_ascending<T, U>);
        break;
    default:
        break;
    }

    for (size_t j = 0; j < k; ++j) {
        out_values[j] = std::get<0>(workspace[j]);
        out_indices[j] = std::get<1>(workspace[j]);
    }
}

}  // namespace reference
}  // namespace ov
