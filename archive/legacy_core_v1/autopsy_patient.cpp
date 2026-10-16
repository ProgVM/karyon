// autopsy_patient.cpp
// KEP Principle 23: Endoscopic Internal State Telemetry & Mechanistic Dissection ("Вскрытие Пациента")
// Standard: KEP v16.0 Sovereign Master & Principle 28 (C++20 Native Execution)
// Author: Bazilevs (ProgVM) & Autonomous Lead AI Cyberneticist

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>

struct StepProbe {
    size_t step;
    uint8_t byte_in;
    uint8_t byte_target;
    uint8_t byte_pred;
    float loss;
    bool is_correct;
    size_t turns_taken;
    float entropy;
    float beta;
    float s_norm;
    float alpha;
    float eta_delta;
    float eta_hebb;
};

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE: ENDOSCOPIC AUTOPSY & MECHANISTIC DISSECTION        \n";
    std::cout << "   Auditing Phase Space, Information Transport & SVD Singularities\n";
    std::cout << "==================================================================\n";

    karyon::NativeConfig cfg;
    cfg.dim = 258;
    cfg.num_heads = 8;
    cfg.dim_k = 32;
    cfg.dim_v = 32;
    cfg.dim_emb = 128;
    cfg.max_thinking_turns = 8;
    cfg.entropy_satisfy = 0.35f;

    karyon::SovereignNativeEngine engine(cfg);

    std::string test_data = 
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n\n"
        "#include <iostream>\n"
        "void dynamic_kernel() { /* zero-torch */ }\n\n"
        "Mind is substrate-independent continuous field dynamics.\n"
        "Ashby homeostatic ultrastability balances energy and surprise.\n";

    std::vector<uint8_t> stream;
    for (int cycle = 0; cycle < 10; ++cycle) {
        stream.insert(stream.end(), test_data.begin(), test_data.end());
    }

    std::vector<StepProbe> probes;
    probes.reserve(stream.size() - 1);

    for (size_t i = 0; i < stream.size() - 1; ++i) {
        uint8_t b_in = stream[i];
        uint8_t b_target = stream[i + 1];
        auto metrics = engine.step(b_in, b_target);

        StepProbe p;
        p.step = i;
        p.byte_in = b_in;
        p.byte_target = b_target;
        p.byte_pred = metrics.pred_byte;
        p.loss = metrics.loss;
        p.is_correct = metrics.is_correct;
        p.turns_taken = metrics.turns_taken;
        p.entropy = metrics.entropy;
        p.beta = metrics.beta;
        p.s_norm = metrics.s_norm;
        p.alpha = metrics.mean_alpha;
        p.eta_delta = metrics.mean_eta_delta;
        p.eta_hebb = metrics.mean_eta_hebb;
        probes.push_back(p);
    }

    // ------------------------------------------------------------------------
    // DISSECTION 1: DISCRIMINATION OF ERROR BY BYTE MODALITIES
    // ------------------------------------------------------------------------
    std::cout << "\n>>> [ORGAN 1: SENSORY-MOTOR ERROR BY BYTE MODALITY] <<<\n";
    // Categories: Whitespace (' ', '\n', '\t'), Alpha ('a'-'z', 'A'-'Z'), Syntax/Punctuation ('(', ')', ':', ';', '{', '}', etc.)
    float loss_ws = 0.0f, loss_alpha = 0.0f, loss_punct = 0.0f;
    size_t count_ws = 0, count_alpha = 0, count_punct = 0;
    size_t err_ws = 0, err_alpha = 0, err_punct = 0;

    // First half vs second half error distribution
    size_t half = probes.size() / 2;
    float second_half_loss_ws = 0.0f, second_half_loss_alpha = 0.0f, second_half_loss_punct = 0.0f;
    size_t sh_count_ws = 0, sh_count_alpha = 0, sh_count_punct = 0;

    for (size_t i = 0; i < probes.size(); ++i) {
        const auto& p = probes[i];
        char c = static_cast<char>(p.byte_target);
        bool is_second_half = (i >= half);

        if (c == ' ' || c == '\n' || c == '\t') {
            loss_ws += p.loss; count_ws++;
            if (!p.is_correct) err_ws++;
            if (is_second_half) { second_half_loss_ws += p.loss; sh_count_ws++; }
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
            loss_alpha += p.loss; count_alpha++;
            if (!p.is_correct) err_alpha++;
            if (is_second_half) { second_half_loss_alpha += p.loss; sh_count_alpha++; }
        } else {
            loss_punct += p.loss; count_punct++;
            if (!p.is_correct) err_punct++;
            if (is_second_half) { second_half_loss_punct += p.loss; sh_count_punct++; }
        }
    }

    std::cout << "Modality            | Count | Overall Loss | 2nd-Half Loss | Accuracy (2nd-Half)\n";
    std::cout << "--------------------+-------+--------------+---------------+--------------------\n";
    std::cout << "Whitespace (' ',\\n) | " << std::setw(5) << count_ws 
              << " | " << std::setw(12) << (loss_ws / count_ws) 
              << " | " << std::setw(13) << (second_half_loss_ws / sh_count_ws)
              << " | " << std::setw(17) << (100.0f - (err_ws * 100.0f / count_ws)) << "%\n";
    std::cout << "Alphanumeric / ID   | " << std::setw(5) << count_alpha 
              << " | " << std::setw(12) << (loss_alpha / count_alpha) 
              << " | " << std::setw(13) << (second_half_loss_alpha / sh_count_alpha)
              << " | " << std::setw(17) << (100.0f - (err_alpha * 100.0f / count_alpha)) << "%\n";
    std::cout << "Syntax / Operators  | " << std::setw(5) << count_punct 
              << " | " << std::setw(12) << (loss_punct / count_punct) 
              << " | " << std::setw(13) << (second_half_loss_punct / sh_count_punct)
              << " | " << std::setw(17) << (100.0f - (err_punct * 100.0f / count_punct)) << "%\n";

    // ------------------------------------------------------------------------
    // DISSECTION 2: RECURRENT DELIBERATIVE THINKING HISTOGRAM
    // ------------------------------------------------------------------------
    std::cout << "\n>>> [ORGAN 2: DELIBERATIVE THINKING TURNS & HALTING PROFILE] <<<\n";
    std::map<size_t, size_t> turns_histogram;
    for (const auto& p : probes) {
        turns_histogram[p.turns_taken]++;
    }

    for (const auto& [turns, cnt] : turns_histogram) {
        float pct = cnt * 100.0f / probes.size();
        std::cout << "  Turns = " << turns << " : " << std::setw(5) << cnt 
                  << " steps (" << std::fixed << std::setprecision(1) << pct << "%) [";
        int bar_len = static_cast<int>(pct / 2.0f);
        for (int b = 0; b < bar_len; ++b) std::cout << "#";
        std::cout << "]\n";
    }

    // ------------------------------------------------------------------------
    // DISSECTION 3: RESIDUAL FATAL ERRORS (TOP 10 FAILED BYTES IN 2ND HALF)
    // ------------------------------------------------------------------------
    std::cout << "\n>>> [ORGAN 3: RESIDUAL FATAL ERRORS — WHY IS LOSS NOT 0.0?] <<<\n";
    std::cout << "Step  | In  | Target | Pred | Loss (nats) | Entropy | Reason for Surprise\n";
    std::cout << "------+-----+--------+------+-------------+---------+------------------------------\n";
    
    // Sort errors in second half by loss descending
    std::vector<StepProbe> high_loss_probes;
    for (size_t i = half; i < probes.size(); ++i) {
        if (!probes[i].is_correct) {
            high_loss_probes.push_back(probes[i]);
        }
    }
    std::sort(high_loss_probes.begin(), high_loss_probes.end(), 
              [](const StepProbe& a, const StepProbe& b) { return a.loss > b.loss; });

    size_t display_count = std::min<size_t>(10, high_loss_probes.size());
    for (size_t i = 0; i < display_count; ++i) {
        const auto& p = high_loss_probes[i];
        std::string in_char = (p.byte_in == '\n') ? "\\n" : (p.byte_in == ' ') ? "' '" : std::string(1, static_cast<char>(p.byte_in));
        std::string tgt_char = (p.byte_target == '\n') ? "\\n" : (p.byte_target == ' ') ? "' '" : std::string(1, static_cast<char>(p.byte_target));
        std::string prd_char = (p.byte_pred == '\n') ? "\\n" : (p.byte_pred == ' ') ? "' '" : std::string(1, static_cast<char>(p.byte_pred));

        std::string explanation = "";
        if (p.byte_in == '\n' || p.byte_in == ' ') {
            explanation = "Branching entropy at word/line boundary";
        } else if (p.byte_target == '(' || p.byte_target == ')' || p.byte_target == '{' || p.byte_target == '}') {
            explanation = "Syntax bracket structural bifurcation";
        } else {
            explanation = "Long-range semantic token transition";
        }

        std::cout << std::setw(5) << p.step << " | "
                  << std::setw(3) << in_char << " | "
                  << std::setw(6) << tgt_char << " | "
                  << std::setw(4) << prd_char << " | "
                  << std::setw(11) << std::fixed << std::setprecision(4) << p.loss << " | "
                  << std::setw(7) << std::setprecision(3) << p.entropy << " | "
                  << explanation << "\n";
    }

    // ------------------------------------------------------------------------
    // DISSECTION 4: ASSOCIATIVE MEMORY SPECTRAL SATURATION
    // ------------------------------------------------------------------------
    std::cout << "\n>>> [ORGAN 4: ASSOCIATIVE MEMORY HEALTH & SATURATION] <<<\n";
    const auto& last_probe = probes.back();
    std::cout << "  Final Precision Beta (Beta_eff) : " << last_probe.beta << " (Max: 21.00)\n";
    std::cout << "  Final Trace Norm (||S||_2)      : " << last_probe.s_norm << " (Bounded by 30.0)\n";
    std::cout << "  Final Retention Alpha (Mean)    : " << last_probe.alpha << "\n";
    std::cout << "  Final Delta Plasticity Eta_Delta: " << last_probe.eta_delta << "\n";
    std::cout << "  Final Hebbian Trace Eta_Hebb    : " << last_probe.eta_hebb << "\n";

    std::cout << "==================================================================\n";
    std::cout << "   AUTOPSY COMPLETE: MECHANICAL DIAGNOSIS EXTRACTED              \n";
    std::cout << "==================================================================\n";

    return 0;
}
