// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Run a Laya encoder GGUF through libllama (embeddings, pooling NONE) on pre-tokenized cases
// and dump per-token hidden states as raw float32: for each case, n x n_embd values.
// cases.txt is the same file the ORT harness reads.
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s enc.gguf cases.txt out.bin\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    auto mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 1;
    auto cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 1024;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cp);
    const int n_embd = llama_model_n_embd(model);

    std::ifstream in(argv[2]);
    std::FILE* out = std::fopen(argv[3], "wb");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int qtype;
        double temp;
        size_t k, n;
        ss >> qtype >> temp >> k;
        for (size_t i = 0, m; i < k; ++i)
            ss >> m;
        ss >> n;
        std::vector<llama_token> ids(n);
        for (auto& v : ids)
            ss >> v;
        llama_batch b = llama_batch_init((int)n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            b.token[i] = ids[i];
            b.pos[i] = (llama_pos)i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = (int)n;
        llama_memory_clear(llama_get_memory(ctx), true);
        auto t0 = std::chrono::steady_clock::now();
        if (llama_encode(ctx, b) != 0) {
            std::fprintf(stderr, "encode failed\n");
            return 1;
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
        for (size_t i = 0; i < n; ++i)
            std::fwrite(llama_get_embeddings_ith(ctx, (int)i), sizeof(float), n_embd, out);
        std::printf("%3zu tok  encode %6.1f ms\n", n, ms);
        llama_batch_free(b);
    }
    std::fclose(out);
    std::ifstream st("/proc/self/status");
    while (std::getline(st, line))
        if (line.rfind("VmHWM:", 0) == 0)
            std::printf("peak RSS %s\n", line.substr(6).c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
