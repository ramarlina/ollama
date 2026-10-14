// Saina Helm 2 routed residual head (routed_residual_v1), ported from the saina
// package's Apache-2.0 reference (saina/routed.py):
//
//   logits = shared(h) + sum_e p_e * up_e(gelu(down_e(h)))
//   p = top-2 routing of softmax(router(layer_norm(h)) / (2 - fraction))
//
// h is the backbone's final (RMS-normalized) hidden state at a question's
// readout token. All arithmetic is float32 storage with double accumulation.
#include "helm2.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct tensor {
    std::vector<int64_t> shape; // outermost first, as in PyTorch
    std::vector<float> data;
};
} // namespace

struct helm2_head::impl {
    int hidden = 0, codes = 0, experts = 0, bottleneck = 0;
    double fraction = 0, alpha = 0;
    tensor shared, norm_w, norm_b, router, down, up;

    explicit impl(const std::string &path) {
        FILE *file = ggml_fopen(path.c_str(), "rb");
        ggml_context *tensors = nullptr;
        gguf_context *meta = gguf_init_from_file(path.c_str(), {true, &tensors});
        try {
            require(meta && tensors && file, "Helm 2: cannot open model");
            const int arch_key = gguf_find_key(meta, "general.architecture");
            require(arch_key >= 0 && gguf_get_kv_type(meta, arch_key) == GGUF_TYPE_STRING,
                    "Helm 2: missing architecture");
            const std::string prefix = std::string(gguf_get_val_str(meta, arch_key)) + ".decision.";
            const int progress_key = gguf_find_key(meta, (prefix + "route_progress").c_str());
            require(progress_key >= 0, "Helm 2: missing routing configuration");
            double progress = 0;
            switch (gguf_get_kv_type(meta, progress_key)) {
            case GGUF_TYPE_FLOAT64:
                progress = gguf_get_val_f64(meta, progress_key);
                break;
            case GGUF_TYPE_FLOAT32:
                progress = gguf_get_val_f32(meta, progress_key);
                break;
            default:
                require(false, "Helm 2: invalid routing configuration");
            }
            require(std::isfinite(progress) && progress >= 0 && progress <= 1, "Helm 2: invalid routing progress");
            // The trained routing schedule, frozen at the saved progress.
            fraction = std::min(std::max(progress / .1, 0.), 1.);
            alpha = std::min(std::max((progress - .1) / .1, 0.), 1.);

            auto load = [&](const char *name, size_t dims) {
                const int index = gguf_find_tensor(meta, name);
                require(index >= 0, "Helm 2: head tensors are missing");
                const ggml_tensor *t = ggml_get_tensor(tensors, name);
                require(t->type == GGML_TYPE_F32, "Helm 2: head tensors must be float32");
                tensor out;
                for (int d = int(dims) - 1; d >= 0; --d)
                    out.shape.push_back(t->ne[d]);
                for (size_t d = dims; d < GGML_MAX_DIMS; ++d)
                    require(t->ne[d] == 1, "Helm 2: unexpected head tensor rank");
                out.data.resize(ggml_nelements(t));
                const size_t position = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index);
#if defined(_WIN32)
                const int seek_rc = _fseeki64(file, static_cast<__int64>(position), SEEK_SET);
#else
                const int seek_rc = fseeko(file, static_cast<off_t>(position), SEEK_SET);
#endif
                const size_t bytes = out.data.size() * sizeof(float);
                require(seek_rc == 0 && std::fread(out.data.data(), 1, bytes, file) == bytes,
                        "Helm 2: truncated tensor data");
                return out;
            };
            shared = load("helm2.shared.weight", 2);
            norm_w = load("helm2.norm.weight", 1);
            norm_b = load("helm2.norm.bias", 1);
            router = load("helm2.router.weight", 2);
            down = load("helm2.down", 3);
            up = load("helm2.up", 3);
            codes = int(shared.shape[0]);
            hidden = int(shared.shape[1]);
            experts = int(router.shape[0]);
            bottleneck = int(down.shape[1]);
            require(codes >= 2 && hidden > 0 && experts >= 2 && bottleneck > 0 && norm_w.shape[0] == hidden &&
                        norm_b.shape[0] == hidden && router.shape[1] == hidden && down.shape[0] == experts &&
                        down.shape[2] == hidden && up.shape[0] == experts && up.shape[1] == codes &&
                        up.shape[2] == bottleneck,
                    "Helm 2: head shape mismatch");
        } catch (...) {
            if (tensors)
                ggml_free(tensors);
            if (meta)
                gguf_free(meta);
            if (file)
                std::fclose(file);
            throw;
        }
        ggml_free(tensors);
        gguf_free(meta);
        std::fclose(file);
    }

    std::vector<float> logits(const std::vector<float> &h, int count) const {
        // Router input: LayerNorm(h), eps 1e-5, biased variance.
        double mean = 0, var = 0;
        for (float v : h)
            mean += v;
        mean /= hidden;
        for (float v : h)
            var += (v - mean) * (v - mean);
        var /= hidden;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        std::vector<float> normed(hidden);
        for (int c = 0; c < hidden; ++c)
            normed[c] = float((h[c] - mean) * inv) * norm_w.data[c] + norm_b.data[c];

        std::vector<double> p(experts);
        const double temperature = 2 - fraction;
        for (int e = 0; e < experts; ++e) {
            double z = 0;
            for (int c = 0; c < hidden; ++c)
                z += double(router.data[size_t(e) * hidden + c]) * normed[c];
            p[e] = float(z) / temperature;
        }
        const double peak = *std::max_element(p.begin(), p.end());
        double sum = 0;
        for (auto &v : p) {
            v = std::exp(v - peak);
            sum += v;
        }
        for (auto &v : p)
            v /= sum;
        if (alpha > 0) {
            // Top 2 experts; on equal weights the lower index comes first.
            int first = 0;
            for (int e = 1; e < experts; ++e)
                if (p[e] > p[first])
                    first = e;
            int second = first == 0 ? 1 : 0;
            for (int e = 0; e < experts; ++e)
                if (e != first && p[e] > p[second])
                    second = e;
            const double total = p[first] + p[second];
            std::vector<double> sparse(experts, 0.0);
            sparse[first] = p[first] / total;
            sparse[second] = p[second] / total;
            for (int e = 0; e < experts; ++e)
                p[e] = p[e] * (1 - alpha) + sparse[e] * alpha;
        }

        std::vector<double> out(count);
        for (int k = 0; k < count; ++k) {
            double v = 0;
            for (int c = 0; c < hidden; ++c)
                v += double(shared.data[size_t(k) * hidden + c]) * h[c];
            out[k] = v;
        }
        std::vector<double> mid(bottleneck);
        for (int e = 0; e < experts; ++e) {
            if (p[e] == 0)
                continue; // contributes exactly zero
            for (int b = 0; b < bottleneck; ++b) {
                double v = 0;
                const float *row = &down.data[(size_t(e) * bottleneck + b) * hidden];
                for (int c = 0; c < hidden; ++c)
                    v += double(row[c]) * h[c];
                mid[b] = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
            }
            for (int k = 0; k < count; ++k) {
                double v = 0;
                const float *row = &up.data[(size_t(e) * codes + k) * bottleneck];
                for (int b = 0; b < bottleneck; ++b)
                    v += double(row[b]) * mid[b];
                out[k] += p[e] * v;
            }
        }
        std::vector<float> result(count);
        for (int k = 0; k < count; ++k) {
            result[k] = float(out[k]);
            require(std::isfinite(result[k]), "Helm 2: non-finite logit");
        }
        return result;
    }

    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &states, const common_json &fields) {
        require(!states.empty(), "Helm 2: missing backbone hidden states");
        require(fields.is_array() && !fields.empty() && fields.size() <= 256, "Helm 2: expected 1-256 questions");
        std::vector<std::vector<float>> result;
        for (const auto &field : fields) {
            const auto &span = field.at("question");
            require(span.is_array() && span.size() == 2 && span[0].is_number_integer() &&
                        span[1].is_number_integer(),
                    "Helm 2: malformed readout span");
            const int start = span[0].get<int>(), end = span[1].get<int>();
            require(start >= 0 && start < end && size_t(end) <= states.size(), "Helm 2: invalid readout span");
            require(field.contains("count") && field.at("count").is_number_integer(), "Helm 2: missing option count");
            const int count = field.at("count").get<int>();
            require(count >= 2 && count <= codes, "Helm 2: unsupported option count");
            const auto &h = states[end - 1];
            require(h.size() == size_t(hidden), "Helm 2: hidden size mismatch");
            result.push_back(logits(h, count));
        }
        return result;
    }
};

helm2_head::helm2_head(const std::string &path) : p(new impl(path)) {}
helm2_head::~helm2_head() = default;
std::vector<std::vector<float>> helm2_head::score(const std::vector<std::vector<float>> &hidden,
                                                  const common_json &fields) {
    return p->score(hidden, fields);
}
