// ============================================================================
// 8.12 AUTONOMOUS ZERO-PRIOR OPERATOR HYPER-SYNTHESIS (EXP-405 REVOLUTION)
// KEP Principle 2, Principle 27 & Rule #12:
// Zero hardcoded human classes or predetermined operator functions!
// Computation emerges from an autopoietic multilinear tensor contraction field:
//   Drift: f(x, h) = (x @ W1 * h @ W2) @ W1^T + x @ W3
//   Metric Flow: g(x, h) = Sigmoid( x @ M @ h^T )
//   Synthesized Output: y = g(x, h) * f(x, h)
// ============================================================================
class ZeroPriorSynthesizedOpImpl : public GraphOp {
public:
    int64_t dim;
    int64_t rank;
    torch::Tensor w1;
    torch::Tensor w2;
    torch::Tensor w3;
    torch::Tensor m_metric;

    ZeroPriorSynthesizedOpImpl(int64_t dim, int64_t rank = 32, std::string device_str = "cpu")
        : dim(dim), rank(rank) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        w1 = register_parameter("w1", torch::randn({dim, rank}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w2 = register_parameter("w2", torch::randn({dim, rank}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        w3 = register_parameter("w3", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (1.0f / std::sqrt((float)dim)));
        m_metric = register_parameter("m_metric", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * (0.1f / std::sqrt((float)dim)));
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) override {
        // Unary forward for standard GraphOp interface (with self-recurrent metric field)
        return forward_birecurrent(x, x);
    }

    torch::Tensor forward_birecurrent(torch::Tensor x, torch::Tensor h) {
        bool is_1d = (x.dim() == 1);
        auto x_2d = is_1d ? x.unsqueeze(0) : x;
        auto h_2d = (h.dim() == 1) ? h.unsqueeze(0) : h;

        // Multilinear tensor core: (x @ W1 * h @ W2) @ W1^T + x @ W3
        auto u1 = torch::matmul(x_2d, w1);
        auto u2 = torch::matmul(h_2d, w2);
        auto core = u1 * u2;
        auto drift = torch::matmul(core, w1.t()) + torch::matmul(x_2d, w3);

        // Dynamic Riemannian metric scalar modulation
        auto h_m = torch::matmul(h_2d, m_metric);
        auto metric_dot = torch::sum(x_2d * h_m, -1, true);
        auto metric_scale = torch::sigmoid(metric_dot);

        auto out = metric_scale * drift;
        return is_1d ? out.squeeze(0) : out;
    }
};
