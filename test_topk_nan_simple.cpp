// Simple standalone test to understand NaN behavior in sorting
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>

using namespace std;

// This mimics the comparison function used in OpenVINO's TopK
template <bool D, typename T, typename U>
inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    if (std::get<0>(a) != std::get<0>(b)) {
        return D ? std::get<0>(a) > std::get<0>(b) : std::get<0>(a) < std::get<0>(b);
    } else {
        return std::get<1>(a) < std::get<1>(b);
    }
}

void test_nan_behavior() {
    cout << "Testing NaN behavior in C++ comparisons and sorting\n";
    cout << "=====================================================\n\n";

    float nan = numeric_limits<float>::quiet_NaN();
    float inf = numeric_limits<float>::infinity();

    // Test basic comparisons
    cout << "Basic NaN comparisons:\n";
    cout << "  NaN < 5.0: " << (nan < 5.0) << "\n";
    cout << "  NaN > 5.0: " << (nan > 5.0) << "\n";
    cout << "  NaN == NaN: " << (nan == nan) << "\n";
    cout << "  NaN != NaN: " << (nan != nan) << "\n";
    cout << "  NaN < inf: " << (nan < inf) << "\n";
    cout << "  NaN > inf: " << (nan > inf) << "\n\n";

    // Test sorting with NaN using tuples (like OpenVINO does)
    {
        cout << "Test 1: Sorting with NaN at beginning (MAX mode):\n";
        vector<tuple<float, int>> data = {
            {nan, 0}, {5.0f, 1}, {3.0f, 2}, {7.0f, 3}, {1.0f, 4}, {9.0f, 5}, {2.0f, 6}, {4.0f, 7}
        };

        cout << "  Original: ";
        for (const auto& [val, idx] : data) {
            if (isnan(val)) cout << "NaN ";
            else cout << val << " ";
        }
        cout << "\n";

        // Simulate nth_element for top 3
        auto cmp = compare_max<true, float, int>;
        nth_element(data.begin(), data.begin() + 3, data.end(), cmp);
        sort(data.begin(), data.begin() + 3, cmp);

        cout << "  Top 3: ";
        for (int i = 0; i < 3; ++i) {
            auto [val, idx] = data[i];
            if (isnan(val)) cout << "NaN(idx=" << idx << ") ";
            else cout << val << "(idx=" << idx << ") ";
        }
        cout << "\n\n";
    }

    {
        cout << "Test 2: Multiple NaNs (MAX mode):\n";
        vector<tuple<float, int>> data = {
            {nan, 0}, {5.0f, 1}, {nan, 2}, {7.0f, 3}, {1.0f, 4}, {nan, 5}, {2.0f, 6}, {4.0f, 7}
        };

        cout << "  Original: ";
        for (const auto& [val, idx] : data) {
            if (isnan(val)) cout << "NaN ";
            else cout << val << " ";
        }
        cout << "\n";

        auto cmp = compare_max<true, float, int>;
        nth_element(data.begin(), data.begin() + 3, data.end(), cmp);
        sort(data.begin(), data.begin() + 3, cmp);

        cout << "  Top 3: ";
        for (int i = 0; i < 3; ++i) {
            auto [val, idx] = data[i];
            if (isnan(val)) cout << "NaN(idx=" << idx << ") ";
            else cout << val << "(idx=" << idx << ") ";
        }
        cout << "\n\n";
    }

    {
        cout << "Test 3: NaN with infinity (MAX mode):\n";
        vector<tuple<float, int>> data = {
            {nan, 0}, {inf, 1}, {5.0f, 2}, {-inf, 3}, {7.0f, 4}, {1.0f, 5}, {2.0f, 6}, {4.0f, 7}
        };

        cout << "  Original: ";
        for (const auto& [val, idx] : data) {
            if (isnan(val)) cout << "NaN ";
            else if (isinf(val)) cout << (val > 0 ? "inf " : "-inf ");
            else cout << val << " ";
        }
        cout << "\n";

        auto cmp = compare_max<true, float, int>;
        nth_element(data.begin(), data.begin() + 3, data.end(), cmp);
        sort(data.begin(), data.begin() + 3, cmp);

        cout << "  Top 3: ";
        for (int i = 0; i < 3; ++i) {
            auto [val, idx] = data[i];
            if (isnan(val)) cout << "NaN(idx=" << idx << ") ";
            else if (isinf(val)) cout << (val > 0 ? "inf" : "-inf") << "(idx=" << idx << ") ";
            else cout << val << "(idx=" << idx << ") ";
        }
        cout << "\n\n";
    }

    {
        cout << "Test 4: MIN mode with NaN:\n";
        vector<tuple<float, int>> data = {
            {nan, 0}, {5.0f, 1}, {3.0f, 2}, {7.0f, 3}, {1.0f, 4}, {9.0f, 5}, {2.0f, 6}, {4.0f, 7}
        };

        cout << "  Original: ";
        for (const auto& [val, idx] : data) {
            if (isnan(val)) cout << "NaN ";
            else cout << val << " ";
        }
        cout << "\n";

        auto cmp = compare_max<false, float, int>;  // false for MIN mode
        nth_element(data.begin(), data.begin() + 3, data.end(), cmp);
        sort(data.begin(), data.begin() + 3, cmp);

        cout << "  Bottom 3: ";
        for (int i = 0; i < 3; ++i) {
            auto [val, idx] = data[i];
            if (isnan(val)) cout << "NaN(idx=" << idx << ") ";
            else cout << val << "(idx=" << idx << ") ";
        }
        cout << "\n\n";
    }

    cout << "Analysis:\n";
    cout << "---------\n";
    cout << "1. NaN comparisons always return false (NaN < x, NaN > x, NaN == x are all false)\n";
    cout << "2. This causes undefined/inconsistent behavior in sorting algorithms\n";
    cout << "3. The exact placement of NaN values depends on implementation details\n";
    cout << "4. We need explicit NaN handling to get predictable behavior\n";
}

int main() {
    test_nan_behavior();
    return 0;
}