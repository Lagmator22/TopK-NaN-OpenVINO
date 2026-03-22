// Proposed v17::TopK implementation with NaN handling
// This would go in src/core/include/openvino/op/v17/topk.hpp

#pragma once

#include "openvino/op/util/topk_base.hpp"

namespace ov {
namespace op {
namespace v17 {

/// \brief Specifies how TopK should handle NaN values
enum class TopKNaNMode {
    NONE,            ///< Keep backward compatibility - undefined behavior (default)
    NAN_AS_SMALLEST, ///< Treat NaN as smallest value (-inf)
    NAN_AS_LARGEST   ///< Treat NaN as largest value (+inf)
};

/// \brief Computes the top K elements with explicit NaN handling
/// \ingroup ov_ops_cpp_api
class TopK : public util::TopKBase {
public:
    OPENVINO_OP("TopK", "opset17", op::util::TopKBase);

    /// \brief Constructs a TopK operation
    TopK() = default;

    /// \brief Constructs a TopK operation with NaN handling
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    /// \param axis The axis along which the TopK operation should be executed
    /// \param mode Specifies whether TopK selects the largest or the smallest elements
    /// \param sort Specifies the order of corresponding elements of the output tensor
    /// \param index_element_type Specifies the data type of the elements in the 'indices' output
    /// \param stable Specifies whether equivalent elements maintain their relative order
    /// \param nan_mode Specifies how to handle NaN values in the input tensor
    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const TopKMode mode,
         const TopKSortType sort,
         const element::Type& index_element_type = element::i32,
         const bool stable = false,
         const TopKNaNMode nan_mode = TopKNaNMode::NONE);

    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const std::string& mode,
         const std::string& sort,
         const element::Type& index_element_type = element::i32,
         const bool stable = false,
         const std::string& nan_mode = "none");

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    bool get_stable() const { return m_stable; }
    void set_stable(const bool stable) { m_stable = stable; }

    TopKNaNMode get_nan_mode() const { return m_nan_mode; }
    void set_nan_mode(const TopKNaNMode nan_mode) { m_nan_mode = nan_mode; }

private:
    bool m_stable = false;
    TopKNaNMode m_nan_mode = TopKNaNMode::NONE;

    static TopKNaNMode nan_mode_from_string(const std::string& mode) {
        if (mode == "nan_as_smallest") return TopKNaNMode::NAN_AS_SMALLEST;
        if (mode == "nan_as_largest") return TopKNaNMode::NAN_AS_LARGEST;
        return TopKNaNMode::NONE;
    }

    static std::string nan_mode_to_string(TopKNaNMode mode) {
        switch (mode) {
            case TopKNaNMode::NAN_AS_SMALLEST: return "nan_as_smallest";
            case TopKNaNMode::NAN_AS_LARGEST: return "nan_as_largest";
            default: return "none";
        }
    }
};

} // namespace v17
} // namespace op
} // namespace ov