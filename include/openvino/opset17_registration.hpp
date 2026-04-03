// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// POC: Shows exactly what files need to be created/modified to register opset17
// and add v17::TopK. This is NOT executable code — it's a reference showing
// the exact diffs that would be applied to the OpenVINO tree.
//
// Based on the pattern from:
//   - opset16.hpp / opset16_tbl.hpp (latest opset in current master)
//   - PR #22796 (SearchSorted added to opset15)
//   - PR #28698 (AvgPool v16)

#pragma once

// ============================================================================
// FILE 1: NEW — src/core/include/openvino/opsets/opset17.hpp
// ============================================================================
//
//   // Copyright (C) 2018-2026 Intel Corporation
//   // SPDX-License-Identifier: Apache-2.0
//   //
//
//   #pragma once
//
//   #include "openvino/op/ops.hpp"
//
//   namespace ov {
//   namespace opset17 {
//   #define _OPENVINO_OP_REG(a, b) using b::a;
//   #include "openvino/opsets/opset17_tbl.hpp"
//   #undef _OPENVINO_OP_REG
//   }  // namespace opset17
//   }  // namespace ov

// ============================================================================
// FILE 2: NEW — src/core/include/openvino/opsets/opset17_tbl.hpp
// ============================================================================
//
//   // Copyright (C) 2018-2026 Intel Corporation
//   // SPDX-License-Identifier: Apache-2.0
//   //
//
//   // opset17 inherits all operations from opset16
//   #include "openvino/opsets/opset16_tbl.hpp"
//
//   // New operations added in opset17
//   // Override TopK from v11 to v17 (with nan_mode attribute)
//   #undef _OPENVINO_OP_REG_TopK
//   _OPENVINO_OP_REG(TopK, ov::op::v17)
//
//   NOTE: The above is a simplified illustration. In practice, the opset_tbl
//   files use a flat list and the TopK line from opset16_tbl.hpp
//   (which says `_OPENVINO_OP_REG(TopK, ov::op::v11)`) would be
//   replaced with `_OPENVINO_OP_REG(TopK, ov::op::v17)` in opset17_tbl.hpp.

// ============================================================================
// FILE 3: MODIFIED — src/core/include/openvino/op/topk.hpp
// ============================================================================
//
//   Add at the end of the file, after the v11 namespace:
//
//   namespace v17 {
//   class OPENVINO_API TopK : public util::TopKBase {
//   public:
//       OPENVINO_OP("TopK", "opset17", op::util::TopKBase);
//
//       TopK() = default;
//
//       TopK(const Output<Node>& data,
//            const Output<Node>& k,
//            const int64_t axis,
//            const std::string& mode,
//            const std::string& sort,
//            const element::Type& index_element_type = element::i32,
//            const bool stable = false,
//            const std::string& nan_mode = "none");
//
//       TopK(const Output<Node>& data,
//            const Output<Node>& k,
//            const int64_t axis,
//            const TopKMode mode,
//            const TopKSortType sort,
//            const element::Type& index_element_type = element::i32,
//            const bool stable = false,
//            const TopKNanMode nan_mode = TopKNanMode::NONE);
//
//       void validate_and_infer_types() override;
//       bool visit_attributes(AttributeVisitor& visitor) override;
//       std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
//
//       bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
//       bool has_evaluate() const override;
//
//       bool get_stable() const { return m_stable; }
//       void set_stable(const bool stable) { m_stable = stable; }
//
//       TopKNanMode get_nan_mode() const { return m_nan_mode; }
//       void set_nan_mode(const TopKNanMode nan_mode) { m_nan_mode = nan_mode; }
//
//   private:
//       bool m_stable = false;
//       TopKNanMode m_nan_mode = TopKNanMode::NONE;
//   };
//   }  // namespace v17

// ============================================================================
// FILE 4: MODIFIED — src/core/include/openvino/op/util/attr_types.hpp
// ============================================================================
//
//   Add after the TopKMode enum (line ~161):
//
//   enum class TopKNanMode {
//       NONE,            // Backward compatible - implementation-defined NaN ordering
//       NAN_AS_SMALLEST, // NaN treated as smallest value (NumPy semantics)
//       NAN_AS_LARGEST,  // NaN treated as largest value (PyTorch semantics)
//   };
//
//   OPENVINO_API
//   std::ostream& operator<<(std::ostream& s, const TopKNanMode& mode);
//
//   And add the AttributeAdapter (after line ~298):
//
//   template <>
//   class OPENVINO_API AttributeAdapter<op::TopKNanMode>
//       : public EnumAttributeAdapterBase<op::TopKNanMode> {
//   public:
//       AttributeAdapter(op::TopKNanMode& value)
//           : EnumAttributeAdapterBase<op::TopKNanMode>(value) {}
//       ~AttributeAdapter() override;
//       OPENVINO_RTTI("AttributeAdapter<TopKNanMode>");
//   };

// ============================================================================
// FILE 5: MODIFIED — src/core/src/op/topk.cpp
// ============================================================================
//
//   Add v17 implementation after v11 namespace (follows exact same pattern):
//
//   namespace v17 {
//   TopK::TopK(const Output<Node>& data,
//              const Output<Node>& k,
//              const int64_t axis,
//              const std::string& mode,
//              const std::string& sort,
//              const element::Type& index_element_type,
//              const bool stable,
//              const std::string& nan_mode)
//       : TopK(data, k, axis, as_enum<TopKMode>(mode), as_enum<TopKSortType>(sort),
//              index_element_type, stable, topk_nan_mode_from_string(nan_mode)) {}
//
//   TopK::TopK(const Output<Node>& data,
//              const Output<Node>& k,
//              const int64_t axis,
//              const TopKMode mode,
//              const TopKSortType sort,
//              const element::Type& index_element_type,
//              const bool stable,
//              const TopKNanMode nan_mode)
//       : util::TopKBase{data, k, axis, mode, sort, index_element_type},
//         m_stable{stable},
//         m_nan_mode{nan_mode} {
//       constructor_validate_and_infer_types();
//   }
//
//   void TopK::validate_and_infer_types() {
//       OV_OP_SCOPE(v17_TopK_validate_and_infer_types);
//       if (m_stable) {
//           NODE_VALIDATION_CHECK(this,
//               m_sort == TopKSortType::SORT_VALUES || m_sort == TopKSortType::SORT_INDICES,
//               "Stable sort requires sort mode VALUE or INDEX.");
//       }
//       util::TopKBase::validate_and_infer_types();
//   }
//
//   bool TopK::visit_attributes(AttributeVisitor& visitor) {
//       OV_OP_SCOPE(v17_TopK_visit_attributes);
//       util::TopKBase::visit_attributes(visitor);
//       visitor.on_attribute("stable", m_stable);
//       visitor.on_attribute("nan_mode", m_nan_mode);
//       return true;
//   }
//
//   std::shared_ptr<Node> TopK::clone_with_new_inputs(const OutputVector& new_args) const {
//       OV_OP_SCOPE(v17_TopK_clone_with_new_inputs);
//       check_new_args_count(this, new_args);
//       return std::make_shared<TopK>(new_args[0], new_args[1], m_axis, m_mode, m_sort,
//                                     m_index_element_type, m_stable, m_nan_mode);
//   }
//
//   bool TopK::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
//       OV_OP_SCOPE(v17_TopK_evaluate);
//       // Modified evaluate that passes nan_mode to reference::topk
//       return topk::evaluate_v17(this, outputs, inputs);
//   }
//
//   bool TopK::has_evaluate() const {
//       OV_OP_SCOPE(v17_TopK_has_evaluate);
//       return topk::validate::data_type(get_input_element_type(0));
//   }
//   }  // namespace v17

// ============================================================================
// FILE 6: MODIFIED — src/core/reference/include/openvino/reference/topk.hpp
// ============================================================================
//
//   Add new NanAwareComparator and topk() overload with nan_mode parameter.
//   See include/openvino/reference/topk.hpp in this POC for the full implementation.
//   Key: existing topk() overload remains UNCHANGED for backward compatibility.
