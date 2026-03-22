// Test program for enhanced TopK with NaN handling
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include "topk_nan_enhanced.hpp"

using namespace std;
using namespace enhanced_topk;

void print_results(const string& test_name, const vector<float>& values, const vector<int64_t>& indices, size_t k) {
    cout << test_name << ":\n";
    cout << "  Values: ";
    for (size_t i = 0; i < k; ++i) {
        if (isnan(values[i])) {
            cout << "NaN ";
        } else if (isinf(values[i])) {
            cout << (values[i] > 0 ? "inf " : "-inf ");
        } else {
            cout << fixed << setprecision(1) << values[i] << " ";
        }
    }
    cout << "\n  Indices: ";
    for (size_t i = 0; i < k; ++i) {
        cout << indices[i] << " ";
    }
    cout << "\n";
}

void run_test_suite() {
    const size_t k = 3;
    float nan = numeric_limits<float>::quiet_NaN();
    float inf = numeric_limits<float>::infinity();

    cout << "Enhanced TopK with NaN Handling - Test Suite\n";
    cout << "============================================\n\n";

    // Test data set 1: Single NaN
    {
        cout << "Test Set 1: Single NaN at beginning\n";
        cout << "------------------------------------\n";
        vector<float> input = {nan, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f};
        cout << "Input: [NaN, 5, 3, 7, 1, 9, 2, 4]\n\n";

        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        // Test with NONE mode (backward compatible)
        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NONE);
        print_results("MAX, NaNMode::NONE", out_values, out_indices, k);

        // Test with NAN_AS_SMALLEST
        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_SMALLEST);
        print_results("MAX, NaNMode::NAN_AS_SMALLEST", out_values, out_indices, k);

        // Test with NAN_AS_LARGEST
        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_LARGEST);
        print_results("MAX, NaNMode::NAN_AS_LARGEST", out_values, out_indices, k);

        // MIN mode tests
        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, false, NaNMode::NAN_AS_SMALLEST);
        print_results("MIN, NaNMode::NAN_AS_SMALLEST", out_values, out_indices, k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, false, NaNMode::NAN_AS_LARGEST);
        print_results("MIN, NaNMode::NAN_AS_LARGEST", out_values, out_indices, k);

        cout << "\n";
    }

    // Test data set 2: Multiple NaNs
    {
        cout << "Test Set 2: Multiple NaNs\n";
        cout << "-------------------------\n";
        vector<float> input = {nan, 5.0f, nan, 7.0f, 1.0f, nan, 2.0f, 4.0f};
        cout << "Input: [NaN, 5, NaN, 7, 1, NaN, 2, 4]\n\n";

        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_SMALLEST);
        print_results("MAX, NaNMode::NAN_AS_SMALLEST", out_values, out_indices, k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_LARGEST);
        print_results("MAX, NaNMode::NAN_AS_LARGEST", out_values, out_indices, k);

        cout << "\n";
    }

    // Test data set 3: NaN with infinity
    {
        cout << "Test Set 3: NaN with infinity\n";
        cout << "-----------------------------\n";
        vector<float> input = {nan, inf, 5.0f, -inf, 7.0f, 1.0f, 2.0f, 4.0f};
        cout << "Input: [NaN, inf, 5, -inf, 7, 1, 2, 4]\n\n";

        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_SMALLEST);
        print_results("MAX, NaNMode::NAN_AS_SMALLEST", out_values, out_indices, k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_LARGEST);
        print_results("MAX, NaNMode::NAN_AS_LARGEST", out_values, out_indices, k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, false, NaNMode::NAN_AS_SMALLEST);
        print_results("MIN, NaNMode::NAN_AS_SMALLEST", out_values, out_indices, k);

        cout << "\n";
    }

    // Test data set 4: All NaNs
    {
        cout << "Test Set 4: All NaNs\n";
        cout << "--------------------\n";
        vector<float> input(8, nan);
        cout << "Input: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]\n\n";

        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), k, true, NaNMode::NAN_AS_LARGEST);
        print_results("MAX, NaNMode::NAN_AS_LARGEST", out_values, out_indices, k);

        cout << "\n";
    }

    // Test stable sort
    {
        cout << "Test Set 5: Stable sort with equal values\n";
        cout << "-----------------------------------------\n";
        vector<float> input = {nan, 5.0f, 5.0f, nan, 5.0f, 7.0f, 5.0f, 8.0f};
        cout << "Input: [NaN, 5, 5, NaN, 5, 7, 5, 8]\n\n";

        vector<float> out_values(5);
        vector<int64_t> out_indices(5);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), 5, true, NaNMode::NAN_AS_LARGEST, false);
        print_results("MAX, NaN_AS_LARGEST, unstable", out_values, out_indices, 5);

        topk_with_nan_handling(input.data(), out_indices.data(), out_values.data(),
                              input.size(), 5, true, NaNMode::NAN_AS_LARGEST, true);
        print_results("MAX, NaN_AS_LARGEST, stable", out_values, out_indices, 5);

        cout << "\n";
    }
}

void performance_test() {
    cout << "Performance Test\n";
    cout << "================\n";

    const size_t sizes[] = {1000, 10000, 100000};
    const size_t k = 100;

    for (size_t size : sizes) {
        vector<float> data(size);
        // Fill with random data including some NaNs
        for (size_t i = 0; i < size; ++i) {
            if (i % 100 == 0) {
                data[i] = numeric_limits<float>::quiet_NaN();
            } else {
                data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
            }
        }

        vector<float> out_values(k);
        vector<int64_t> out_indices(k);

        auto start = chrono::high_resolution_clock::now();
        topk_with_nan_handling(data.data(), out_indices.data(), out_values.data(),
                              data.size(), k, true, NaNMode::NAN_AS_SMALLEST);
        auto end = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        cout << "Size: " << setw(7) << size << ", Top-" << k << " time: "
             << setw(6) << duration << " μs\n";
    }
    cout << "\n";
}

int main() {
    run_test_suite();
    performance_test();

    cout << "Summary:\n";
    cout << "--------\n";
    cout << "- NaN handling modes provide predictable behavior\n";
    cout << "- NAN_AS_SMALLEST treats NaN as -infinity\n";
    cout << "- NAN_AS_LARGEST treats NaN as +infinity\n";
    cout << "- NONE mode maintains backward compatibility\n";
    cout << "- Stable sort option preserves relative order\n";

    return 0;
}