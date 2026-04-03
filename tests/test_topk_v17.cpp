// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Comprehensive test suite for v17::TopK NaN handling.
//
// Tests cover:
//   1. Backward compatibility (NONE mode matches v11 behavior)
//   2. NAN_AS_SMALLEST correctness (MAX + MIN modes)
//   3. NAN_AS_LARGEST correctness (MAX + MIN modes)
//   4. Edge cases: all-NaN, single-NaN, NaN with inf, empty-ish inputs
//   5. Stable sort with NaN
//   6. Multi-dimensional axis behavior (simulated)
//   7. Strict weak ordering verification

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../include/openvino/reference/topk.hpp"

using namespace ov::op;
using namespace ov::reference;

// ===== Test infrastructure =====

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_CASE(name)                                     \
    static void test_##name();                              \
    static struct Register_##name {                         \
        Register_##name() { test_registry().push_back({#name, test_##name}); } \
    } register_##name;                                      \
    static void test_##name()

struct TestEntry {
    const char* name;
    void (*func)();
};

static std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> r;
    return r;
}

static void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::cerr << "  FAIL: " << expr << " at " << file << ":" << line << "\n";
        g_tests_failed++;
    } else {
        g_tests_passed++;
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

static bool is_nan(float v) { return std::isnan(v); }

// ===== Helper =====

struct TopKResult {
    std::vector<float> values;
    std::vector<int64_t> indices;
};

TopKResult run_topk(const std::vector<float>& input,
                    size_t k,
                    bool compute_max,
                    TopKSortType sort = TopKSortType::SORT_VALUES,
                    TopKNanMode nan_mode = TopKNanMode::NONE,
                    bool stable = false) {
    TopKResult r;
    r.values.resize(k);
    r.indices.resize(k);
    ov::reference::topk(input.data(), r.indices.data(), r.values.data(),
                        input.size(), k, compute_max, sort, nan_mode, stable);
    return r;
}

TopKResult run_topk_legacy(const std::vector<float>& input,
                           size_t k,
                           bool compute_max,
                           TopKSortType sort = TopKSortType::SORT_VALUES) {
    TopKResult r;
    r.values.resize(k);
    r.indices.resize(k);
    ov::reference::topk(input.data(), r.indices.data(), r.values.data(),
                        input.size(), k, compute_max, sort);
    return r;
}

// ============================================================================
// Test 1: Backward compatibility — NONE mode matches legacy (no NaN in input)
// ============================================================================
TEST_CASE(backward_compat_no_nan) {
    std::vector<float> input = {5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f, 8.0f};

    auto legacy = run_topk_legacy(input, 3, true, TopKSortType::SORT_VALUES);
    auto v17    = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NONE);

    for (size_t i = 0; i < 3; ++i) {
        CHECK(legacy.values[i] == v17.values[i]);
        CHECK(legacy.indices[i] == v17.indices[i]);
    }
}

TEST_CASE(backward_compat_min_mode) {
    std::vector<float> input = {5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f, 8.0f};

    auto legacy = run_topk_legacy(input, 3, false, TopKSortType::SORT_VALUES);
    auto v17    = run_topk(input, 3, false, TopKSortType::SORT_VALUES, TopKNanMode::NONE);

    for (size_t i = 0; i < 3; ++i) {
        CHECK(legacy.values[i] == v17.values[i]);
        CHECK(legacy.indices[i] == v17.indices[i]);
    }
}

// ============================================================================
// Test 2: NAN_AS_SMALLEST — MAX mode (NaN should NOT appear in top results)
// ============================================================================
TEST_CASE(nan_as_smallest_max_single_nan) {
    //                idx: 0    1     2     3     4     5     6     7
    std::vector<float> input = {NAN, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX top-3 should be: 9.0(5), 7.0(3), 5.0(1) — NaN excluded
    CHECK(r.values[0] == 9.0f);
    CHECK(r.values[1] == 7.0f);
    CHECK(r.values[2] == 5.0f);
    CHECK(r.indices[0] == 5);
    CHECK(r.indices[1] == 3);
    CHECK(r.indices[2] == 1);
}

TEST_CASE(nan_as_smallest_max_multiple_nan) {
    std::vector<float> input = {NAN, 5.0f, NAN, 7.0f, 1.0f, NAN, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX top-3: 7.0(3), 5.0(1), 4.0(7)
    CHECK(r.values[0] == 7.0f);
    CHECK(r.values[1] == 5.0f);
    CHECK(r.values[2] == 4.0f);
    CHECK(r.indices[0] == 3);
    CHECK(r.indices[1] == 1);
    CHECK(r.indices[2] == 7);
}

// ============================================================================
// Test 3: NAN_AS_SMALLEST — MIN mode (NaN should appear first)
// ============================================================================
TEST_CASE(nan_as_smallest_min) {
    std::vector<float> input = {NAN, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, false, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MIN bottom-3 with NaN as smallest: NaN(0), 1.0(4), 2.0(6)
    CHECK(is_nan(r.values[0]));
    CHECK(r.indices[0] == 0);
    CHECK(r.values[1] == 1.0f);
    CHECK(r.values[2] == 2.0f);
}

// ============================================================================
// Test 4: NAN_AS_LARGEST — MAX mode (NaN should appear first in results)
// ============================================================================
TEST_CASE(nan_as_largest_max_single_nan) {
    std::vector<float> input = {NAN, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    // MAX top-3 with NaN as largest: NaN(0), 9.0(5), 7.0(3)
    CHECK(is_nan(r.values[0]));
    CHECK(r.indices[0] == 0);
    CHECK(r.values[1] == 9.0f);
    CHECK(r.indices[1] == 5);
    CHECK(r.values[2] == 7.0f);
    CHECK(r.indices[2] == 3);
}

TEST_CASE(nan_as_largest_max_multiple_nan) {
    std::vector<float> input = {NAN, 5.0f, NAN, 7.0f, 1.0f, NAN, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    // MAX top-3: NaN(0), NaN(2), NaN(5) — all NaNs surface first
    CHECK(is_nan(r.values[0]));
    CHECK(is_nan(r.values[1]));
    CHECK(is_nan(r.values[2]));
    // Indices should be stable (ascending): 0, 2, 5
    CHECK(r.indices[0] == 0);
    CHECK(r.indices[1] == 2);
    CHECK(r.indices[2] == 5);
}

// ============================================================================
// Test 5: NAN_AS_LARGEST — MIN mode (NaN should NOT appear in bottom results)
// ============================================================================
TEST_CASE(nan_as_largest_min) {
    std::vector<float> input = {NAN, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, false, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    // MIN bottom-3: 1.0(4), 2.0(6), 3.0(2) — NaN excluded from bottom
    CHECK(r.values[0] == 1.0f);
    CHECK(r.values[1] == 2.0f);
    CHECK(r.values[2] == 3.0f);
    CHECK(r.indices[0] == 4);
    CHECK(r.indices[1] == 6);
    CHECK(r.indices[2] == 2);
}

// ============================================================================
// Test 6: All NaN input
// ============================================================================
TEST_CASE(all_nan_as_smallest) {
    std::vector<float> input(8, NAN);

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // All NaN: should return NaN values with stable index order 0,1,2
    for (size_t i = 0; i < 3; ++i) {
        CHECK(is_nan(r.values[i]));
    }
    CHECK(r.indices[0] == 0);
    CHECK(r.indices[1] == 1);
    CHECK(r.indices[2] == 2);
}

TEST_CASE(all_nan_as_largest) {
    std::vector<float> input(8, NAN);

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(is_nan(r.values[i]));
    }
    CHECK(r.indices[0] == 0);
    CHECK(r.indices[1] == 1);
    CHECK(r.indices[2] == 2);
}

// ============================================================================
// Test 7: NaN with infinity
// ============================================================================
TEST_CASE(nan_with_inf_smallest) {
    float inf = std::numeric_limits<float>::infinity();
    std::vector<float> input = {NAN, inf, 5.0f, -inf, 7.0f, 1.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX top-3: inf(1), 7.0(4), 5.0(2) — NaN smaller than -inf
    CHECK(r.values[0] == inf);
    CHECK(r.indices[0] == 1);
    CHECK(r.values[1] == 7.0f);
    CHECK(r.indices[1] == 4);
    CHECK(r.values[2] == 5.0f);
    CHECK(r.indices[2] == 2);
}

TEST_CASE(nan_with_inf_largest) {
    float inf = std::numeric_limits<float>::infinity();
    std::vector<float> input = {NAN, inf, 5.0f, -inf, 7.0f, 1.0f, 2.0f, 4.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    // MAX top-3: NaN(0), inf(1), 7.0(4) — NaN larger than +inf
    CHECK(is_nan(r.values[0]));
    CHECK(r.indices[0] == 0);
    CHECK(r.values[1] == inf);
    CHECK(r.indices[1] == 1);
    CHECK(r.values[2] == 7.0f);
    CHECK(r.indices[2] == 4);
}

// ============================================================================
// Test 8: Stable sort preserves insertion order for equal values
// ============================================================================
TEST_CASE(stable_sort_with_nan) {
    // Two NaN at idx 0,3 + duplicates at idx 1,2,4
    std::vector<float> input = {NAN, 5.0f, 5.0f, NAN, 5.0f, 7.0f, 5.0f, 8.0f};

    auto r = run_topk(input, 6, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST, true);
    // Stable MAX: NaN(0), NaN(3), 8.0(7), 7.0(5), 5.0(1), 5.0(2)
    CHECK(is_nan(r.values[0]));
    CHECK(r.indices[0] == 0);
    CHECK(is_nan(r.values[1]));
    CHECK(r.indices[1] == 3);
    CHECK(r.values[2] == 8.0f);
    CHECK(r.values[3] == 7.0f);
    CHECK(r.values[4] == 5.0f);
    CHECK(r.indices[4] == 1);  // First 5.0
    CHECK(r.values[5] == 5.0f);
    CHECK(r.indices[5] == 2);  // Second 5.0
}

// ============================================================================
// Test 9: k == input size (return everything)
// ============================================================================
TEST_CASE(k_equals_input_size) {
    std::vector<float> input = {NAN, 3.0f, 1.0f, NAN, 2.0f};

    auto r = run_topk(input, 5, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX sorted: 3.0, 2.0, 1.0, NaN, NaN
    CHECK(r.values[0] == 3.0f);
    CHECK(r.values[1] == 2.0f);
    CHECK(r.values[2] == 1.0f);
    CHECK(is_nan(r.values[3]));
    CHECK(is_nan(r.values[4]));
    // NaN indices should be stable: 0 before 3
    CHECK(r.indices[3] == 0);
    CHECK(r.indices[4] == 3);
}

// ============================================================================
// Test 10: k = 1
// ============================================================================
TEST_CASE(k_equals_one_max_nan_smallest) {
    std::vector<float> input = {NAN, 5.0f, NAN, 3.0f};
    auto r = run_topk(input, 1, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    CHECK(r.values[0] == 5.0f);
    CHECK(r.indices[0] == 1);
}

TEST_CASE(k_equals_one_max_nan_largest) {
    std::vector<float> input = {NAN, 5.0f, NAN, 3.0f};
    auto r = run_topk(input, 1, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
    CHECK(is_nan(r.values[0]));
    CHECK(r.indices[0] == 0);
}

// ============================================================================
// Test 11: No NaN in input — NaN modes should not change results
// ============================================================================
TEST_CASE(no_nan_all_modes_same) {
    std::vector<float> input = {5.0f, 3.0f, 7.0f, 1.0f, 9.0f};

    auto r_none     = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NONE);
    auto r_smallest = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    auto r_largest  = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);

    for (size_t i = 0; i < 3; ++i) {
        CHECK(r_none.values[i] == r_smallest.values[i]);
        CHECK(r_none.values[i] == r_largest.values[i]);
        CHECK(r_none.indices[i] == r_smallest.indices[i]);
        CHECK(r_none.indices[i] == r_largest.indices[i]);
    }
}

// ============================================================================
// Test 12: SORT_INDICES mode
// ============================================================================
TEST_CASE(sort_indices_mode) {
    std::vector<float> input = {NAN, 5.0f, 3.0f, 7.0f, 1.0f, 9.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_INDICES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX top-3 by value: 9.0(5), 7.0(3), 5.0(1), but sorted by INDEX: 1,3,5
    CHECK(r.indices[0] == 1);
    CHECK(r.indices[1] == 3);
    CHECK(r.indices[2] == 5);
    CHECK(r.values[0] == 5.0f);
    CHECK(r.values[1] == 7.0f);
    CHECK(r.values[2] == 9.0f);
}

// ============================================================================
// Test 13: Negative values with NaN
// ============================================================================
TEST_CASE(negative_values_with_nan) {
    std::vector<float> input = {-5.0f, NAN, -3.0f, -7.0f, -1.0f, -9.0f};

    auto r = run_topk(input, 3, true, TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
    // MAX top-3: -1.0(4), -3.0(2), -5.0(0)
    CHECK(r.values[0] == -1.0f);
    CHECK(r.values[1] == -3.0f);
    CHECK(r.values[2] == -5.0f);
}

// ============================================================================
// Test 14: Operator class construction and attribute access
// ============================================================================
TEST_CASE(operator_class_attributes) {
    ov::op::v17::TopK op(TopKMode::MAX, TopKSortType::SORT_VALUES, 0, true, TopKNanMode::NAN_AS_SMALLEST);

    CHECK(op.get_mode() == TopKMode::MAX);
    CHECK(op.get_sort_type() == TopKSortType::SORT_VALUES);
    CHECK(op.get_axis() == 0);
    CHECK(op.get_stable() == true);
    CHECK(op.get_nan_mode() == TopKNanMode::NAN_AS_SMALLEST);

    // Test string constructor
    ov::op::v17::TopK op2("max", "value", 1, false, "nan_as_largest");
    CHECK(op2.get_mode() == TopKMode::MAX);
    CHECK(op2.get_nan_mode() == TopKNanMode::NAN_AS_LARGEST);
}

// ============================================================================
// Test 15: Enum string conversion round-trip
// ============================================================================
TEST_CASE(enum_string_roundtrip) {
    CHECK(topk_nan_mode_from_string("none") == TopKNanMode::NONE);
    CHECK(topk_nan_mode_from_string("nan_as_smallest") == TopKNanMode::NAN_AS_SMALLEST);
    CHECK(topk_nan_mode_from_string("nan_as_largest") == TopKNanMode::NAN_AS_LARGEST);
    CHECK(topk_nan_mode_from_string("NONE") == TopKNanMode::NONE);
    CHECK(topk_nan_mode_from_string("NAN_AS_SMALLEST") == TopKNanMode::NAN_AS_SMALLEST);
    CHECK(topk_nan_mode_from_string("NAN_AS_LARGEST") == TopKNanMode::NAN_AS_LARGEST);

    CHECK(topk_nan_mode_to_string(TopKNanMode::NONE) == "none");
    CHECK(topk_nan_mode_to_string(TopKNanMode::NAN_AS_SMALLEST) == "nan_as_smallest");
    CHECK(topk_nan_mode_to_string(TopKNanMode::NAN_AS_LARGEST) == "nan_as_largest");

    // ostream
    std::ostringstream ss;
    ss << TopKNanMode::NAN_AS_SMALLEST;
    CHECK(ss.str() == "NAN_AS_SMALLEST");
}

// ============================================================================
// Test 16: Default nan_mode is NONE (backward compat)
// ============================================================================
TEST_CASE(default_nan_mode_is_none) {
    ov::op::v17::TopK op;
    CHECK(op.get_nan_mode() == TopKNanMode::NONE);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "v17::TopK NaN Handling — Test Suite\n";
    std::cout << "====================================\n\n";

    for (auto& [name, func] : test_registry()) {
        std::cout << "  Running: " << name << " ... ";
        int before_fail = g_tests_failed;
        func();
        if (g_tests_failed == before_fail) {
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED\n";
        }
    }

    std::cout << "\n------------------------------------\n";
    std::cout << "Results: " << g_tests_passed << " checks passed, "
              << g_tests_failed << " checks failed\n";

    if (g_tests_failed > 0) {
        std::cout << "\nFAILURE\n";
        return 1;
    }
    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}
