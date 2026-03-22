// Example: How enhanced TopK would improve OvaSearch retrieval
// This shows practical benefits of NaN handling in RAG systems

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include "topk_nan_enhanced.hpp"

using namespace std;
using namespace enhanced_topk;

struct Document {
    string content;
    float similarity_score;
    size_t index;
};

void demonstrate_rag_scenario() {
    cout << "OvaSearch Integration Example: TopK with NaN Handling\n";
    cout << "====================================================\n\n";

    cout << "Scenario: Multimodal RAG with missing/corrupted embeddings\n";
    cout << "-----------------------------------------------------------\n\n";

    // Simulate similarity scores from vector search
    // Some documents might have NaN scores due to:
    // 1. Corrupted embeddings
    // 2. Division by zero in cosine similarity
    // 3. Missing data in multimodal pipeline
    // 4. Numerical instability in embedding models

    vector<Document> documents = {
        {"Document 1: Introduction to OpenVINO", 0.92f, 0},
        {"Document 2: [CORRUPTED EMBEDDING]", numeric_limits<float>::quiet_NaN(), 1},
        {"Document 3: TopK operator details", 0.87f, 2},
        {"Document 4: Neural network basics", 0.75f, 3},
        {"Document 5: [MISSING IMAGE DATA]", numeric_limits<float>::quiet_NaN(), 4},
        {"Document 6: Performance optimization", 0.88f, 5},
        {"Document 7: Model quantization guide", 0.91f, 6},
        {"Document 8: [FAILED VLM PROCESSING]", numeric_limits<float>::quiet_NaN(), 7},
        {"Document 9: Hardware acceleration", 0.83f, 8},
        {"Document 10: Deployment strategies", 0.79f, 9}
    };

    cout << "Document Similarity Scores:\n";
    for (const auto& doc : documents) {
        cout << "  [" << doc.index << "] " << left << setw(40) << doc.content << " : ";
        if (isnan(doc.similarity_score)) {
            cout << "NaN (corrupted/missing)";
        } else {
            cout << fixed << setprecision(2) << doc.similarity_score;
        }
        cout << "\n";
    }
    cout << "\n";

    const size_t k = 5;  // Retrieve top 5 documents
    vector<float> scores;
    vector<int64_t> indices_storage(k);

    // Extract scores for TopK
    for (const auto& doc : documents) {
        scores.push_back(doc.similarity_score);
    }

    vector<float> out_values(k);

    cout << "Retrieval Results with Different NaN Modes:\n";
    cout << "-------------------------------------------\n\n";

    // Test 1: Current behavior (NONE mode)
    {
        topk_with_nan_handling(scores.data(), indices_storage.data(), out_values.data(),
                              scores.size(), k, true, NaNMode::NONE);

        cout << "1. NONE mode (current OpenVINO behavior - undefined):\n";
        cout << "   Retrieved documents:\n";
        for (size_t i = 0; i < k; ++i) {
            size_t idx = indices_storage[i];
            cout << "     " << (i+1) << ". " << documents[idx].content;
            if (isnan(out_values[i])) {
                cout << " (NaN - UNRELIABLE!)";
            } else {
                cout << " (score: " << fixed << setprecision(2) << out_values[i] << ")";
            }
            cout << "\n";
        }
        cout << "   WARNING: Corrupted documents might appear in results!\n\n";
    }

    // Test 2: NAN_AS_SMALLEST mode
    {
        topk_with_nan_handling(scores.data(), indices_storage.data(), out_values.data(),
                              scores.size(), k, true, NaNMode::NAN_AS_SMALLEST);

        cout << "2. NAN_AS_SMALLEST mode (exclude corrupted):\n";
        cout << "   Retrieved documents:\n";
        for (size_t i = 0; i < k; ++i) {
            size_t idx = indices_storage[i];
            cout << "     " << (i+1) << ". " << documents[idx].content;
            cout << " (score: " << fixed << setprecision(2) << out_values[i] << ")";
            cout << "\n";
        }
        cout << "   CLEAN RESULTS: Only valid documents retrieved\n\n";
    }

    // Test 3: NAN_AS_LARGEST mode (for debugging)
    {
        topk_with_nan_handling(scores.data(), indices_storage.data(), out_values.data(),
                              scores.size(), k, true, NaNMode::NAN_AS_LARGEST);

        cout << "3. NAN_AS_LARGEST mode (prioritize problematic docs for debugging):\n";
        cout << "   Retrieved documents:\n";
        for (size_t i = 0; i < k; ++i) {
            size_t idx = indices_storage[i];
            cout << "     " << (i+1) << ". " << documents[idx].content;
            if (isnan(out_values[i])) {
                cout << " (NaN - NEEDS ATTENTION!)";
            } else {
                cout << " (score: " << fixed << setprecision(2) << out_values[i] << ")";
            }
            cout << "\n";
        }
        cout << "   DEBUG MODE: Corrupted documents shown first for investigation\n\n";
    }
}

void demonstrate_llm_generation() {
    cout << "LLM Token Generation Scenario: Handling NaN in Logits\n";
    cout << "====================================================\n\n";

    // In LLM generation, NaN can appear in logits due to:
    // 1. Numerical overflow in attention scores
    // 2. Division by zero in softmax
    // 3. Gradient explosion during inference

    vector<string> tokens = {
        "the", "a", "is", "[NaN]", "cat", "dog", "[NaN]", "running", "walking", "sleeping"
    };

    vector<float> logits = {
        2.1f, 1.8f, 3.2f,
        numeric_limits<float>::quiet_NaN(),  // Numerical issue
        4.5f, 3.9f,
        numeric_limits<float>::quiet_NaN(),  // Another NaN
        2.7f, 3.1f, 2.3f
    };

    cout << "Token logits before sampling:\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        cout << "  " << setw(10) << tokens[i] << ": ";
        if (isnan(logits[i])) {
            cout << "NaN";
        } else {
            cout << fixed << setprecision(2) << logits[i];
        }
        cout << "\n";
    }
    cout << "\n";

    const size_t k = 3;  // Top-K sampling
    vector<float> top_logits(k);
    vector<int64_t> top_indices(k);

    // With NAN_AS_SMALLEST, we exclude NaN tokens from generation
    topk_with_nan_handling(logits.data(), top_indices.data(), top_logits.data(),
                          logits.size(), k, true, NaNMode::NAN_AS_SMALLEST);

    cout << "Top-3 tokens for generation (NAN_AS_SMALLEST):\n";
    for (size_t i = 0; i < k; ++i) {
        cout << "  " << (i+1) << ". \"" << tokens[top_indices[i]]
             << "\" (logit: " << fixed << setprecision(2) << top_logits[i] << ")\n";
    }
    cout << "\nSAFE GENERATION: NaN tokens excluded, preventing corrupted output\n";
}

int main() {
    demonstrate_rag_scenario();
    cout << "\n";
    demonstrate_llm_generation();

    cout << "\n\nConclusion:\n";
    cout << "===========\n";
    cout << "NaN handling in TopK is critical for production RAG systems:\n";
    cout << "1. Prevents corrupted documents from appearing in retrieval\n";
    cout << "2. Ensures consistent behavior across platforms\n";
    cout << "3. Enables debugging mode to identify problematic embeddings\n";
    cout << "4. Protects LLM generation from numerical instabilities\n";
    cout << "5. Essential for multimodal pipelines where some modalities may fail\n";

    return 0;
}