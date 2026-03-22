// Enhanced reference implementation with NaN handling
// This would go in src/core/reference/include/openvino/reference/topk.hpp

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include <tuple>
#include <vector>

#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {

// NaN handling mode enum (duplicated from op definition for reference impl)
enum class TopKNaNMode {
    NONE,
    NAN_AS_SMALLEST,
    NAN_AS_LARGEST
};

// NaN-aware comparison function for v17
template <bool IsMaxMode, typename T, typename U>
class NaNAwareComparatorV17 {
private:
    TopKNaNMode nan_mode;

public:
    explicit NaNAwareComparatorV17(TopKNaNMode mode = TopKNaNMode::NONE) : nan_mode(mode) {}

    bool operator()(const std::tuple<T, U>& a, const std::tuple<T, U>& b) const {
        T val_a = std::get<0>(a);
        T val_b = std::get<0>(b);
        U idx_a = std::get<1>(a);
        U idx_b = std::get<1>(b);

        // Only do NaN checks if not in NONE mode (performance optimization)
        if (nan_mode != TopKNaNMode::NONE) {
            bool a_is_nan = std::isnan(val_a);
            bool b_is_nan = std::isnan(val_b);

            if (a_is_nan || b_is_nan) {
                if (nan_mode == TopKNaNMode::NAN_AS_SMALLEST) {
                    if (a_is_nan && b_is_nan) {
                        return idx_a < idx_b;  // Stable sort for multiple NaN
                    }
                    if (a_is_nan) {
                        return !IsMaxMode;  // NaN is smallest
                    }
                    return IsMaxMode;  // Regular value > NaN
                } else {  // NAN_AS_LARGEST
                    if (a_is_nan && b_is_nan) {
                        return idx_a < idx_b;  // Stable sort for multiple NaN
                    }
                    if (a_is_nan) {
                        return IsMaxMode;  // NaN is largest
                    }
                    return !IsMaxMode;  // Regular value < NaN
                }
            }
        }

        // Normal comparison (no NaN or NONE mode)
        if (val_a != val_b) {
            return IsMaxMode ? val_a > val_b : val_a < val_b;
        }
        return idx_a < idx_b;  // Stable sort for equal values
    }
};

// Updated topk function with NaN handling
template <typename T, typename U>
void topk_v17(const T* arg,
              U* out_indices,
              T* out_values,
              const Shape& in_shape,
              const Shape& out_shape,
              const size_t axis,
              const size_t k,
              const bool compute_max,
              const op::TopKSortType sort = op::TopKSortType::NONE,
              const TopKNaNMode nan_mode = TopKNaNMode::NONE,
              const bool stable = false) {

    // Create workspace
    std::vector<std::tuple<T, U>> workspace(in_shape[axis]);
    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);
    const auto in_axis_stride = in_strides[axis];
    const auto out_axis_stride = out_strides[axis];

    // Select comparator based on nan_mode
    std::function<bool(const std::tuple<T, U>&, const std::tuple<T, U>&)> cmp_func;

    if (nan_mode != TopKNaNMode::NONE) {
        cmp_func = compute_max ?
            NaNAwareComparatorV17<true, T, U>(nan_mode) :
            NaNAwareComparatorV17<false, T, U>(nan_mode);
    } else {
        // Original comparator for backward compatibility
        cmp_func = compute_max ?
            [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) > std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
            } :
            [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) < std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
            };
    }

    // Determine sort function
    decltype(cmp_func) sort_func = nullptr;
    switch (sort) {
        case op::TopKSortType::SORT_INDICES:
            sort_func = [](const auto& a, const auto& b) {
                return std::get<1>(a) < std::get<1>(b);
            };
            break;
        case op::TopKSortType::SORT_VALUES:
            sort_func = cmp_func;
            break;
        default:
            break;
    }

    // Process each slice
    auto traverse_shape = in_shape;
    traverse_shape[axis] = 1;
    CoordinateTransformBasic traverse_transform(traverse_shape);

    for (const auto& coord : traverse_transform) {
        auto arg_index = coordinate_index(coord, in_shape);
        auto out_index = coordinate_index(coord, out_shape);

        // Fill workspace
        U i = 0;
        for (auto& entry : workspace) {
            std::get<0>(entry) = arg[arg_index];
            std::get<1>(entry) = i;
            arg_index += in_axis_stride;
            ++i;
        }

        // Sort or partial sort based on stable flag
        if (stable && sort_func) {
            std::stable_sort(workspace.begin(), workspace.end(), cmp_func);
            // Take first k elements
            if (k < workspace.size()) {
                workspace.resize(k);
            }
        } else {
            std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp_func);
            if (sort_func) {
                std::sort(workspace.begin(), workspace.begin() + k, sort_func);
            }
        }

        // Extract results
        for (size_t j = 0; j < k; ++j) {
            out_values[out_index] = std::get<0>(workspace[j]);
            out_indices[out_index] = std::get<1>(workspace[j]);
            out_index += out_axis_stride;
        }
    }
}

} // namespace reference
} // namespace ov