// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// POC: This file demonstrates the new TopKNanMode enum that would be added
// to src/core/include/openvino/op/util/attr_types.hpp in the real implementation.
//
// Pattern follows existing enums: TopKSortType, TopKMode, FillMode

#pragma once

#include <ostream>
#include <string>
#include <stdexcept>

namespace ov {
namespace op {

/// \brief Specifies how NaN values are handled during TopK sorting.
///
/// This enum controls the position of NaN values in the sorted output:
/// - NONE:             Backward-compatible default. NaN ordering is implementation-defined
///                     (same as v11::TopK and earlier). No extra NaN checks are performed,
///                     so there is zero overhead for existing users.
/// - NAN_AS_SMALLEST:  NaN is treated as smaller than any finite value (equivalent to -inf).
///                     Matches NumPy np.sort / np.argpartition NaN semantics.
///                     Use case: exclude corrupted scores from top-K results.
/// - NAN_AS_LARGEST:   NaN is treated as larger than any finite value (equivalent to +inf).
///                     Matches PyTorch torch.topk NaN semantics.
///                     Use case: surface corrupted scores for debugging / monitoring.
enum class TopKNanMode {
    NONE = 0,
    NAN_AS_SMALLEST,
    NAN_AS_LARGEST,
};

inline std::ostream& operator<<(std::ostream& s, const TopKNanMode& mode) {
    switch (mode) {
    case TopKNanMode::NONE:
        return s << "NONE";
    case TopKNanMode::NAN_AS_SMALLEST:
        return s << "NAN_AS_SMALLEST";
    case TopKNanMode::NAN_AS_LARGEST:
        return s << "NAN_AS_LARGEST";
    default:
        return s << "UNKNOWN";
    }
}

/// \brief Convert string to TopKNanMode (for IR deserialization / string constructors)
inline TopKNanMode topk_nan_mode_from_string(const std::string& mode) {
    if (mode == "none" || mode == "NONE")
        return TopKNanMode::NONE;
    if (mode == "nan_as_smallest" || mode == "NAN_AS_SMALLEST")
        return TopKNanMode::NAN_AS_SMALLEST;
    if (mode == "nan_as_largest" || mode == "NAN_AS_LARGEST")
        return TopKNanMode::NAN_AS_LARGEST;
    throw std::invalid_argument("Unknown TopKNanMode: " + mode);
}

/// \brief Convert TopKNanMode to string (for IR serialization / visit_attributes)
inline std::string topk_nan_mode_to_string(TopKNanMode mode) {
    switch (mode) {
    case TopKNanMode::NONE:
        return "none";
    case TopKNanMode::NAN_AS_SMALLEST:
        return "nan_as_smallest";
    case TopKNanMode::NAN_AS_LARGEST:
        return "nan_as_largest";
    default:
        return "none";
    }
}

// POC note: In the real implementation, the following AttributeAdapter would be added
// to attr_types.hpp, following the exact same pattern as TopKSortType and TopKMode:
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

}  // namespace op
}  // namespace ov
