// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone C++ demo: RAG pipeline with NaN-corrupted similarity scores.
//
// Demonstrates the practical impact of each nan_mode on a retrieval task.
// No external dependencies required — compiles with just a C++17 compiler.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../include/openvino/reference/topk.hpp"

using namespace ov::op;
using namespace ov::reference;

struct Document {
    std::string title;
    float score;
};

void print_topk(const std::string& label,
                const float* values,
                const int64_t* indices,
                size_t k,
                const std::vector<Document>& docs) {
    std::cout << "\n  " << label << ":\n";
    int nan_count = 0;
    for (size_t i = 0; i < k; ++i) {
        int idx = static_cast<int>(indices[i]);
        std::cout << "    #" << (i + 1) << ": [" << std::setw(2) << idx << "] ";
        if (std::isnan(values[i])) {
            std::cout << "   NaN  " << docs[idx].title << "  <-- CORRUPTED\n";
            nan_count++;
        } else {
            std::cout << std::fixed << std::setprecision(4) << values[i]
                      << "  " << docs[idx].title << "\n";
        }
    }
    if (nan_count > 0) {
        std::cout << "    WARNING: " << nan_count << " corrupted result(s) shown to user!\n";
    }
}

int main() {
    std::cout << "v17::TopK — RAG Pipeline NaN Handling Demo\n";
    std::cout << "============================================\n";

    // Simulated document corpus with similarity scores
    std::vector<Document> docs = {
        {"Introduction to neural networks",       0.42f},  // 0
        {"Python basics tutorial",                0.38f},  // 1
        {"Web development with Django",           0.31f},  // 2
        {"Machine learning fundamentals",         0.92f},  // 3  HIGH
        {"Database design patterns",              0.35f},  // 4
        {"Cloud computing overview",              0.44f},  // 5
        {"Kubernetes deployment guide",           0.29f},  // 6
        {"OpenVINO inference optimization",       0.88f},  // 7  HIGH
        {"Data visualization with matplotlib",    0.41f},  // 8
        {"REST API best practices",               0.33f},  // 9
        {"Microservices architecture",            0.37f},  // 10
        {"DevOps CI/CD pipelines",                0.40f},  // 11
        {"Model quantization for edge",           0.85f},  // 12 HIGH
        {"Natural language processing",           0.47f},  // 13
        {"Computer vision fundamentals",          0.51f},  // 14
        {"Real-time detection with OpenVINO",     0.91f},  // 15 HIGH
        {"Graph neural networks",                 0.39f},  // 16
        {"Reinforcement learning intro",          0.36f},  // 17
        {"Transfer learning techniques",          0.48f},  // 18
        {"Federated learning overview",           0.34f},  // 19
    };

    const size_t n = docs.size();
    const size_t k = 5;

    // Extract scores
    std::vector<float> scores(n);
    for (size_t i = 0; i < n; ++i) scores[i] = docs[i].score;

    std::cout << "\n[1] Clean pipeline (no NaN) — all modes produce same result\n";
    {
        std::vector<float> out_v(k);
        std::vector<int64_t> out_i(k);
        topk(scores.data(), out_i.data(), out_v.data(), n, k, true,
             TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
        print_topk("Top-5 (clean data)", out_v.data(), out_i.data(), k, docs);
    }

    // Inject NaN corruption
    std::cout << "\n[2] Corrupted pipeline (NaN at indices 7, 11, 15)\n";
    std::cout << "    Cause: FP16 overflow in embedding projection layer\n";

    std::vector<float> corrupted = scores;
    corrupted[7]  = std::numeric_limits<float>::quiet_NaN();  // Was 0.88
    corrupted[11] = std::numeric_limits<float>::quiet_NaN();  // Was 0.40
    corrupted[15] = std::numeric_limits<float>::quiet_NaN();  // Was 0.91

    // Show all three modes
    {
        std::vector<float> out_v(k);
        std::vector<int64_t> out_i(k);

        // NONE mode (current v11 behavior — undefined with NaN)
        topk(corrupted.data(), out_i.data(), out_v.data(), n, k, true,
             TopKSortType::SORT_VALUES, TopKNanMode::NONE);
        print_topk("nan_mode=NONE (v11 compat — undefined NaN order)", out_v.data(), out_i.data(), k, docs);

        // NAN_AS_SMALLEST (production mode)
        topk(corrupted.data(), out_i.data(), out_v.data(), n, k, true,
             TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_SMALLEST);
        print_topk("nan_mode=NAN_AS_SMALLEST (production — exclude corrupted)", out_v.data(), out_i.data(), k, docs);

        // NAN_AS_LARGEST (debug mode)
        topk(corrupted.data(), out_i.data(), out_v.data(), n, k, true,
             TopKSortType::SORT_VALUES, TopKNanMode::NAN_AS_LARGEST);
        print_topk("nan_mode=NAN_AS_LARGEST (debug — surface corrupted)", out_v.data(), out_i.data(), k, docs);
    }

    // Framework comparison
    std::cout << "\n[3] Framework alignment\n";
    std::cout << "  +-----------------------+-------------------+-------------------+\n";
    std::cout << "  | Framework             | NaN in MAX TopK   | nan_mode equiv    |\n";
    std::cout << "  +-----------------------+-------------------+-------------------+\n";
    std::cout << "  | NumPy np.argpartition | Last (smallest)   | NAN_AS_SMALLEST   |\n";
    std::cout << "  | PyTorch torch.topk    | First (largest)   | NAN_AS_LARGEST    |\n";
    std::cout << "  | TensorFlow tf.math    | Last (smallest)   | NAN_AS_SMALLEST   |\n";
    std::cout << "  | Current OpenVINO v11  | Undefined         | NONE              |\n";
    std::cout << "  +-----------------------+-------------------+-------------------+\n";

    std::cout << "\n  With v17::TopK, users importing models from any framework can\n";
    std::cout << "  select the matching nan_mode to preserve original semantics.\n\n";

    return 0;
}
