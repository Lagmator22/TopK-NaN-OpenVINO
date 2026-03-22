// Test program to understand current TopK NaN behavior
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>

#include "openvino/reference/topk.hpp"
#include "openvino/core/shape.hpp"

using namespace ov;
using namespace std;

void print_results(const string& test_name, const vector<float>& values, const vector<int64_t>& indices, size_t k) {
    cout << "\n" << test_name << ":\n";
    cout << "  Top " << k << " values: ";
    for (size_t i = 0; i < k; ++i) {
        if (isnan(values[i])) {
            cout << "NaN ";
        } else if (isinf(values[i])) {
            cout << (values[i] > 0 ? "inf" : "-inf") << " ";
        } else {
            cout << values[i] << " ";
        }
    }
    cout << "\n  Indices: ";
    for (size_t i = 0; i < k; ++i) {
        cout << indices[i] << " ";
    }
    cout << endl;
}

void test_topk_with_nan() {
    const size_t k = 3;

    // Test 1: NaN at the beginning
    {
        vector<float> input = {
            numeric_limits<float>::quiet_NaN(),
            5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f
        };
        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        Shape in_shape = {8};
        Shape out_shape = {k};

        reference::topk(input.data(), out_indices.data(), out_values.data(),
                       in_shape, out_shape, 0, k, true, op::TopKSortType::SORT_VALUES);

        print_results("Test 1: NaN at beginning (MAX mode)", out_values, out_indices, k);
    }

    // Test 2: Multiple NaNs
    {
        vector<float> input = {
            numeric_limits<float>::quiet_NaN(),
            5.0f,
            numeric_limits<float>::quiet_NaN(),
            7.0f, 1.0f,
            numeric_limits<float>::quiet_NaN(),
            2.0f, 4.0f
        };
        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        Shape in_shape = {8};
        Shape out_shape = {k};

        reference::topk(input.data(), out_indices.data(), out_values.data(),
                       in_shape, out_shape, 0, k, true, op::TopKSortType::SORT_VALUES);

        print_results("Test 2: Multiple NaNs (MAX mode)", out_values, out_indices, k);
    }

    // Test 3: NaN with infinity
    {
        vector<float> input = {
            numeric_limits<float>::quiet_NaN(),
            numeric_limits<float>::infinity(),
            5.0f,
            -numeric_limits<float>::infinity(),
            7.0f, 1.0f, 2.0f, 4.0f
        };
        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        Shape in_shape = {8};
        Shape out_shape = {k};

        reference::topk(input.data(), out_indices.data(), out_values.data(),
                       in_shape, out_shape, 0, k, true, op::TopKSortType::SORT_VALUES);

        print_results("Test 3: NaN with infinity (MAX mode)", out_values, out_indices, k);
    }

    // Test 4: MIN mode with NaN
    {
        vector<float> input = {
            numeric_limits<float>::quiet_NaN(),
            5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f
        };
        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        Shape in_shape = {8};
        Shape out_shape = {k};

        reference::topk(input.data(), out_indices.data(), out_values.data(),
                       in_shape, out_shape, 0, k, false, op::TopKSortType::SORT_VALUES);

        print_results("Test 4: NaN at beginning (MIN mode)", out_values, out_indices, k);
    }

    // Test 5: All NaNs
    {
        vector<float> input(8, numeric_limits<float>::quiet_NaN());
        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        Shape in_shape = {8};
        Shape out_shape = {k};

        reference::topk(input.data(), out_indices.data(), out_values.data(),
                       in_shape, out_shape, 0, k, true, op::TopKSortType::SORT_VALUES);

        print_results("Test 5: All NaNs (MAX mode)", out_values, out_indices, k);
    }
}

int main() {
    cout << "Testing OpenVINO TopK NaN behavior\n";
    cout << "===================================\n";

    test_topk_with_nan();

    cout << "\n\nAnalysis:\n";
    cout << "----------\n";
    cout << "The current TopK implementation uses std::nth_element and std::sort,\n";
    cout << "which have implementation-defined behavior for NaN values.\n";
    cout << "NaNs are typically treated inconsistently in comparisons.\n";

    return 0;
}