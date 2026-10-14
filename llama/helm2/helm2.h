#pragma once
#include "json.h"
#include "llama.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Saina Helm 2's routed head. The backbone is evaluated by llama-server; the head
// reads the final hidden state at one readout token per question.
class helm2_head {
  public:
    explicit helm2_head(const std::string &path);
    ~helm2_head();
    // hidden holds the final hidden state of every prompt token. Each field is
    // {"question": [start, end], "count": n}: the readout is the last token of
    // the span, and the head returns logits for the first n answer codes.
    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &hidden, const common_json &fields);

  private:
    struct impl;
    std::unique_ptr<impl> p;
};

// Whether the model is Saina Helm 2 (GGUF metadata <arch>.decision.type = helm2).
inline bool helm2_model(const llama_model *model) {
    char arch[128], decision[16];
    return model && llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) >= 0 &&
           llama_model_meta_val_str(model, (std::string(arch) + ".decision.type").c_str(), decision,
                                    sizeof(decision)) >= 0 &&
           std::strcmp(decision, "helm2") == 0;
}
