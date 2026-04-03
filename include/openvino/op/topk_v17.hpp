// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// POC: Proposed v17::TopK operator definition.
//
// In the real implementation this would be added to:
//   src/core/include/openvino/op/topk.hpp  (inside the existing file, as a new namespace v17)
//
// Pattern follows v11::TopK exactly, adding only the nan_mode attribute.
// Inherits from util::TopKBase, reusing all existing shape inference,
// axis normalization, mode/sort handling.

#pragma once

#include "topk_nan_mode.hpp"

// ---- Minimal stubs for standalone compilation ----
// In the real OpenVINO build these come from the framework headers.
// We stub them here so the POC compiles without the full OpenVINO tree.
#ifndef OPENVINO_FULL_BUILD

#include <cstdint>
#include <string>

namespace ov {
namespace op {

enum class TopKSortType { NONE, SORT_INDICES, SORT_VALUES };
enum class TopKMode { MAX, MIN };

inline TopKMode topk_mode_from_string(const std::string& s) {
    return (s == "max" || s == "MAX") ? TopKMode::MAX : TopKMode::MIN;
}
inline TopKSortType topk_sort_from_string(const std::string& s) {
    if (s == "value" || s == "SORT_VALUES") return TopKSortType::SORT_VALUES;
    if (s == "index" || s == "SORT_INDICES") return TopKSortType::SORT_INDICES;
    return TopKSortType::NONE;
}

}  // namespace op
}  // namespace ov

#endif  // OPENVINO_FULL_BUILD

namespace ov {
namespace op {
namespace v17 {

/// \brief Computes the top K elements with explicit NaN handling.
///
/// Extends v11::TopK with a configurable `nan_mode` attribute that controls
/// how NaN values participate in the sorting / partial-sort step.
///
/// \ingroup ov_ops_cpp_api
///
/// Opset registration (would go in opset17_tbl.hpp):
///   _OPENVINO_OP_REG(TopK, ov::op::v17)
///
/// Attributes (inherited from TopKBase + new):
///   - axis              int64_t         (from TopKBase)
///   - mode              TopKMode        (from TopKBase)
///   - sort              TopKSortType    (from TopKBase)
///   - index_element_type element::Type  (from TopKBase)
///   - stable            bool            (from v11)
///   - nan_mode          TopKNanMode     (NEW in v17)
///
/// Default nan_mode = NONE preserves exact backward compatibility with v11::TopK.
///
/// Reference PRs studied for this pattern:
///   - https://github.com/openvinotoolkit/openvino/pull/22796
///   - https://github.com/openvinotoolkit/openvino/pull/28698
class TopK {
public:
    // In real impl: OPENVINO_OP("TopK", "opset17", op::util::TopKBase);

    TopK() = default;

    /// \brief Full constructor (enum-based)
    TopK(TopKMode mode,
         TopKSortType sort,
         int64_t axis,
         bool stable = false,
         TopKNanMode nan_mode = TopKNanMode::NONE)
        : m_mode(mode),
          m_sort(sort),
          m_axis(axis),
          m_stable(stable),
          m_nan_mode(nan_mode) {}

    /// \brief String constructor (for IR deserialization)
    TopK(const std::string& mode,
         const std::string& sort,
         int64_t axis,
         bool stable = false,
         const std::string& nan_mode = "none")
        : m_mode(topk_mode_from_string(mode)),
          m_sort(topk_sort_from_string(sort)),
          m_axis(axis),
          m_stable(stable),
          m_nan_mode(topk_nan_mode_from_string(nan_mode)) {}

    // --- Accessors (match v11::TopK API) ---

    TopKMode get_mode() const { return m_mode; }
    void set_mode(TopKMode mode) { m_mode = mode; }

    TopKSortType get_sort_type() const { return m_sort; }
    void set_sort_type(TopKSortType sort) { m_sort = sort; }

    int64_t get_axis() const { return m_axis; }
    void set_axis(int64_t axis) { m_axis = axis; }

    bool get_stable() const { return m_stable; }
    void set_stable(bool stable) { m_stable = stable; }

    // --- NEW in v17 ---

    TopKNanMode get_nan_mode() const { return m_nan_mode; }
    void set_nan_mode(TopKNanMode nan_mode) { m_nan_mode = nan_mode; }

    // POC note: In the real implementation the following methods would be overridden
    // from TopKBase / v11::TopK:
    //
    //   void validate_and_infer_types() override {
    //       // Same as v11 + stable-sort validation
    //       if (m_stable) {
    //           NODE_VALIDATION_CHECK(this,
    //               m_sort == TopKSortType::SORT_VALUES || m_sort == TopKSortType::SORT_INDICES,
    //               "Stable sort requires sort mode VALUE or INDEX.");
    //       }
    //       util::TopKBase::validate_and_infer_types();
    //   }
    //
    //   bool visit_attributes(AttributeVisitor& visitor) override {
    //       util::TopKBase::visit_attributes(visitor);
    //       visitor.on_attribute("stable", m_stable);
    //       visitor.on_attribute("nan_mode", m_nan_mode);  // <-- new attribute serialized to IR
    //       return true;
    //   }
    //
    //   std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
    //       check_new_args_count(this, new_args);
    //       return std::make_shared<TopK>(new_args[0], new_args[1], m_axis, m_mode, m_sort,
    //                                     m_index_element_type, m_stable, m_nan_mode);
    //   }
    //
    //   bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    //   bool has_evaluate() const override;

private:
    TopKMode m_mode = TopKMode::MAX;
    TopKSortType m_sort = TopKSortType::NONE;
    int64_t m_axis = 0;
    bool m_stable = false;
    TopKNanMode m_nan_mode = TopKNanMode::NONE;
};

}  // namespace v17
}  // namespace op
}  // namespace ov
