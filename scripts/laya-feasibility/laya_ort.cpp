// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Laya feasibility harness: run the exported decision ONNX through the ORT C++ API on
// pre-tokenized cases and print calibrated probabilities next to the PyTorch reference.
// cases.txt, one case per line:
//   qtype temperature k marker_0..marker_{k-1} n id_0..id_{n-1} ref_p_0..ref_p_{k-1}
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static long peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0)
            return std::stol(line.substr(6));
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.onnx cases.txt [threads]\n", argv[0]);
        return 2;
    }
    const int threads = argc > 3 ? std::atoi(argv[3]) : 4;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "laya");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto t0 = std::chrono::steady_clock::now();
    Ort::Session sess(env, argv[1], so);
    double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("load %.0f ms, peak RSS after load %ld MB, threads %d\n", load_ms,
                peak_rss_kb() / 1024, threads);

    const char* in_names[] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
    const char* out_names[] = {"logits"};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::ifstream in(argv[2]);
    std::string line;
    double worst = 0;
    int n_cases = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int64_t qtype;
        double temp;
        size_t k, n;
        ss >> qtype >> temp >> k;
        std::vector<int64_t> markers(k);
        for (auto& m : markers)
            ss >> m;
        ss >> n;
        std::vector<int64_t> ids(n);
        for (auto& v : ids)
            ss >> v;
        std::vector<double> ref(k);
        for (auto& r : ref)
            ss >> r;

        std::vector<int64_t> att(n, 1);
        std::unique_ptr<bool[]> mmask(new bool[k]);
        std::fill(mmask.get(), mmask.get() + k, true);
        const int64_t s_ids[] = {1, (int64_t)n}, s_k[] = {1, (int64_t)k}, s_q[] = {1};
        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<int64_t>(mem, ids.data(), n, s_ids, 2),
            Ort::Value::CreateTensor<int64_t>(mem, att.data(), n, s_ids, 2),
            Ort::Value::CreateTensor<int64_t>(mem, markers.data(), k, s_k, 2),
            Ort::Value::CreateTensor<bool>(mem, mmask.get(), k, s_k, 2),
            Ort::Value::CreateTensor<int64_t>(mem, &qtype, 1, s_q, 1),
        };
        auto t1 = std::chrono::steady_clock::now();
        auto out = sess.Run(Ort::RunOptions{nullptr}, in_names, inputs, 5, out_names, 1);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1)
                        .count();
        const float* lg = out[0].GetTensorData<float>();

        std::vector<double> p(k);
        double mx = -1e30, sum = 0;
        for (size_t i = 0; i < k; ++i)
            mx = std::max(mx, lg[i] / temp);
        for (size_t i = 0; i < k; ++i)
            sum += p[i] = std::exp(lg[i] / temp - mx);
        std::printf("case %d  %3zu tok  %6.1f ms  p=[", n_cases, n, ms);
        for (size_t i = 0; i < k; ++i) {
            p[i] /= sum;
            worst = std::max(worst, std::fabs(p[i] - ref[i]));
            std::printf("%s%.4f", i ? ", " : "", p[i]);
        }
        std::printf("]  ref=[");
        for (size_t i = 0; i < k; ++i)
            std::printf("%s%.4f", i ? ", " : "", ref[i]);
        std::printf("]\n");
        ++n_cases;
    }
    std::printf("%d cases, max |p_cpp - p_torch| = %.2e, peak RSS %ld MB\n", n_cases, worst,
                peak_rss_kb() / 1024);
    return 0;
}
