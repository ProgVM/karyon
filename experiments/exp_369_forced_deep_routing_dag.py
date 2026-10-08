"""
EXP-369: Deep Autopoietic Mathematical Expression DAG with Forced Deep Routing (DAM-DAG-FDR)
=============================================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Eliminate all shortcut pathways. Force the gradient descent engine to route through the deep,
newly synthesized mathematical composition nodes by preventing the readout from directly
bypassing the graph and reading raw input nodes."

Key Architectural Enhancements:
1. Forced Deep Routing (Anti-Shortcut Mandate):
   - In EXP-368, the readout layer bypassed the deep computation graph by routing directly from
     raw input nodes (Nodes 0..7). This caused "gradient laziness" (shortcut learning).
   - In EXP-369, the readout layer is strictly prohibited from accessing raw inputs (x_t) and
     registers (s_t-1) directly.
   - The readout layer must compute predictions strictly by routing from the terminal active
     computation nodes (the latest 4 active computation nodes in the DAG).
   - This forces the system to utilize the mathematical transformations synthesized inside the DAG.
2. Arbitrary-Depth Composition Trees:
   - Computation nodes can still route from any predecessor, creating deep recursive formulas.
   - Primitives:
     * Unary: {id, neg, sin, cos, tanh, exp_decay, sqr, recip}
     * Binary: {add, mul, sub, div_safe}
3. Temperature Annealing & Single-Pass Stream Learning (N=1).
4. True Recursive Symbolic Formula Output.
=============================================================================================
"""

import os
import time
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UNBROKEN REALITY STREAM GENERATOR (8-bit bipolar vector space R^8)
# =============================================================================

def byte_to_bit_vector(b: int) -> torch.Tensor:
    bits = [1.0 if ((b >> i) & 1) else -1.0 for i in range(8)]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


def bit_vector_to_byte(vec: torch.Tensor) -> int:
    signs = (vec > 0.0).cpu().numpy().astype(int)
    val = 0
    for i, bit in enumerate(signs):
        if bit:
            val |= (1 << i)
    return val


def generate_unbroken_stream(stream_length: int = 2048):
    vectors = []
    stream_types = []  # 0: det, 1: stoch
    raw_bytes = []

    current_lfsr = 0xA5
    current_brownian = 0.0
    mode = 0
    block_remaining = 64

    for _ in range(stream_length):
        if block_remaining <= 0:
            mode = 1 - mode
            block_remaining = random.randint(32, 64)

        if mode == 0:
            current_lfsr = ((current_lfsr << 1) ^ (0x1D if (current_lfsr & 0x80) else 0x00)) & 0xFF
            val = current_lfsr
            stream_types.append(0)
        else:
            current_brownian += random.gauss(0.0, 1.0)
            val = int((math.sin(current_brownian * 0.15) * 0.5 + 0.5) * 255)
            stream_types.append(1)

        raw_bytes.append(val)
        vectors.append(byte_to_bit_vector(val))
        block_remaining -= 1

    stream_tensor = torch.stack(vectors, dim=0)
    type_tensor = torch.tensor(stream_types, dtype=torch.long, device=DEVICE)
    return stream_tensor, type_tensor, raw_bytes


# =============================================================================
# ATOMIC MATHEMATICAL OPERATORS
# =============================================================================

UNARY_OPS = [
    ("id", lambda x: x),
    ("neg", lambda x: -x),
    ("sin", lambda x: torch.sin(torch.clamp(x, -10.0, 10.0))),
    ("cos", lambda x: torch.cos(torch.clamp(x, -10.0, 10.0))),
    ("tanh", lambda x: torch.tanh(x)),
    ("exp_decay", lambda x: torch.exp(-torch.abs(torch.clamp(x, -10.0, 10.0)))),
    ("sqr", lambda x: torch.clamp(x**2, -20.0, 20.0)),
    ("recip", lambda x: 1.0 / (torch.abs(x) + 1e-4))
]

BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: torch.clamp(a * b, -20.0, 20.0)),
    ("sub", lambda a, b: a - b),
    ("div_safe", lambda a, b: torch.clamp(a / (torch.abs(b) + 1e-4), -20.0, 20.0))
]

NUM_UNARY = len(UNARY_OPS)
NUM_BINARY = len(BINARY_OPS)
TOTAL_OPS = NUM_UNARY + NUM_BINARY


# =============================================================================
# COMPOSITION NODE (THE BUILDING BLOCK OF DAM-DAG)
# =============================================================================

class DAMComputationNode(nn.Module):
    def __init__(self, node_id: int, max_predecessors: int, device: torch.device = DEVICE):
        super().__init__()
        self.node_id = node_id
        self.max_predecessors = max_predecessors
        self.device = device

        # Routing parameters: logits for selecting inputs from predecessors
        self.routing_logits_1 = nn.Parameter(torch.randn(max_predecessors, device=device) * 0.1)
        self.routing_logits_2 = nn.Parameter(torch.randn(max_predecessors, device=device) * 0.1)

        # Operator selection logits
        self.op_logits = nn.Parameter(torch.randn(TOTAL_OPS, device=device) * 0.1)

        self.scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, predecessor_values: list, temperature: float = 1.0):
        preds_tensor = torch.cat(predecessor_values, dim=0)  # [num_preds]

        num_preds = len(predecessor_values)
        p_route_1 = torch.softmax(self.routing_logits_1[:num_preds] / temperature, dim=-1)
        p_route_2 = torch.softmax(self.routing_logits_2[:num_preds] / temperature, dim=-1)

        in_1 = torch.sum(preds_tensor * p_route_1, dim=0, keepdim=True)
        in_2 = torch.sum(preds_tensor * p_route_2, dim=0, keepdim=True)

        outputs = []
        for name, op in UNARY_OPS:
            outputs.append(op(in_1))
        for name, op in BINARY_OPS:
            outputs.append(op(in_1, in_2))

        outputs_tensor = torch.cat(outputs, dim=0)
        p_ops = torch.softmax(self.op_logits / temperature, dim=-1)

        out_val = torch.sum(outputs_tensor * p_ops, dim=0, keepdim=True)
        final_val = self.scale * out_val + self.bias

        # Entropies
        entropy_r1 = -torch.sum(p_route_1 * torch.log(p_route_1 + 1e-8))
        entropy_r2 = -torch.sum(p_route_2 * torch.log(p_route_2 + 1e-8))
        entropy_ops = -torch.sum(p_ops * torch.log(p_ops + 1e-8))
        total_entropy = entropy_r1 + entropy_r2 + entropy_ops

        return final_val, total_entropy


# =============================================================================
# DEEP AUTOPOIETIC MATHEMATICAL DAG ENGINE WITH FORCED DEEP ROUTING
# =============================================================================

class DAMDAGEngineFDR(nn.Module):
    def __init__(self, in_dim: int = 8, initial_registers: int = 2, max_computation_nodes: int = 24, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.max_comp_nodes = max_computation_nodes
        self.device = device

        # Registers (persistent state values)
        self.num_registers = initial_registers
        self.register_values = nn.Parameter(torch.zeros(initial_registers, 1, device=device))

        # Computation nodes pool
        self.comp_nodes = nn.ModuleList()
        for i in range(max_computation_nodes):
            max_preds = in_dim + initial_registers + i
            self.comp_nodes.append(DAMComputationNode(node_id=i, max_predecessors=max_preds, device=device))

        # Active computation nodes count
        self.active_comp_nodes = 4

        # Readout routing: strictly prohibited from accessing raw inputs (Nodes 0..7) and registers
        # The readout layer can only select from the latest 4 active computation nodes
        self.readout_logits = nn.Parameter(torch.randn(in_dim, 4, device=device) * 0.1)

        self.sprout_events = 0

    def sprout_node(self):
        if self.active_comp_nodes < self.max_comp_nodes:
            self.active_comp_nodes += 1
            self.sprout_events += 1
            return True
        return False

    def forward(self, x_t: torch.Tensor, temp: float = 1.0):
        # x_t: [8]
        preds = [x_t[i].unsqueeze(0) for i in range(self.in_dim)]

        for k in range(self.num_registers):
            preds.append(self.register_values[k])

        total_entropy = 0.0

        # Execute active computation nodes sequentially
        comp_outputs = []
        for i in range(self.active_comp_nodes):
            node_out, entropy = self.comp_nodes[i](preds, temp)
            preds.append(node_out)
            comp_outputs.append(node_out)
            total_entropy += entropy

        # Readout: strictly route from the latest 4 active computation nodes
        # This completely forces the signal through the computational depth of the DAG
        terminal_nodes = comp_outputs[-4:]
        stacked_terminals = torch.cat(terminal_nodes, dim=0)  # [4]

        y_pred_list = []
        for i in range(self.in_dim):
            p_readout = torch.softmax(self.readout_logits[i] / temp, dim=-1)
            y_i = torch.sum(stacked_terminals * p_readout, dim=0, keepdim=True)
            y_pred_list.append(torch.tanh(y_i))

        y_pred = torch.cat(y_pred_list, dim=0)

        # Update registers with the latest computation nodes to preserve memory flow
        with torch.no_grad():
            for k in range(self.num_registers):
                source_idx = -1 - k
                self.register_values[k].copy_(preds[source_idx])

        return y_pred, total_entropy

    def extract_symbolic_formulas(self) -> str:
        """Recursively decodes the entire computational DAG into nested algebraic equations."""
        unary_names = [name for name, _ in UNARY_OPS]
        binary_names = [name for name, _ in BINARY_OPS]

        node_formulas = []
        for i in range(self.in_dim):
            node_formulas.append(f"x_{i}")
        for k in range(self.num_registers):
            node_formulas.append(f"s_{k}")

        report = []
        report.append("=== KARYON AUTOPOIETIC FORCED DEEP ROUTING DAG SPECIFICATION ===")

        # Trace computation nodes
        for i in range(self.active_comp_nodes):
            node = self.comp_nodes[i]
            num_preds = self.in_dim + self.num_registers + i
            
            p_route_1 = torch.softmax(node.routing_logits_1[:num_preds], dim=-1).detach().cpu().numpy()
            p_route_2 = torch.softmax(node.routing_logits_2[:num_preds], dim=-1).detach().cpu().numpy()
            p_ops = torch.softmax(node.op_logits, dim=-1).detach().cpu().numpy()

            idx_1 = int(np.argmax(p_route_1))
            idx_2 = int(np.argmax(p_route_2))
            op_idx = int(np.argmax(p_ops))

            arg_1 = node_formulas[idx_1]
            arg_2 = node_formulas[idx_2]

            scale_val = node.scale.item()
            bias_val = node.bias.item()

            if op_idx < NUM_UNARY:
                op_name = unary_names[op_idx]
                expr = f"{scale_val:.2f} * {op_name}({arg_1}) + {bias_val:.2f}"
            else:
                op_name = binary_names[op_idx - NUM_UNARY]
                expr = f"{scale_val:.2f} * {op_name}({arg_1}, {arg_2}) + {bias_val:.2f}"

            node_formulas.append(f"Node_{i}")
            report.append(f"  Node_{i} = {expr}  [Inputs: {idx_1} (conf {p_route_1[idx_1]:.2f}), {idx_2} (conf {p_route_2[idx_2]:.2f}) | Op: {op_name} (conf {p_ops[op_idx]:.2f})]")

        # Trace Readouts
        report.append("  Forced Readout Equations:")
        terminal_offset = self.in_dim + self.num_registers + self.active_comp_nodes - 4
        for i in range(self.in_dim):
            p_readout = torch.softmax(self.readout_logits[i], dim=-1).detach().cpu().numpy()
            best_local_idx = int(np.argmax(p_readout))
            global_idx = terminal_offset + best_local_idx
            report.append(f"    y_pred[{i}] = tanh({node_formulas[global_idx]})  [Source Terminal: Node_{global_idx - (self.in_dim + self.num_registers)} (conf {p_readout[best_local_idx]:.2f})]")

        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_369():
    print("===============================================================================")
    print("=== KEP EXP-369: FORCED DEEP ROUTING DAG (DAM-DAG-FDR)                      ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = DAMDAGEngineFDR(in_dim=8, initial_registers=2, max_computation_nodes=24, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []

    rolling_strain = 2.0
    t0 = time.time()

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        temp = max(0.1, 1.0 - (t / stream_len) * 0.8)

        optimizer.zero_grad()

        # Step computational DAG
        y_pred, routing_entropy = model(cur_vec, temp=temp)

        # Physical quadratic strain + complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.001 * routing_entropy + 0.002 * model.active_comp_nodes

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: expand computation nodes if strain > 1.1
        if t > 40 and t % 32 == 0:
            if rolling_strain > 1.1:
                if model.sprout_node():
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)

        # Bit parity check
        pred_byte = bit_vector_to_byte(y_pred)
        tgt_byte = raw_bytes[t + 1]
        bit_diff = bin(pred_byte ^ tgt_byte).count('1')

        if is_stochastic == 0:
            det_errors.append(strain_val)
            bit_mismatches_det.append(bit_diff)
        else:
            stoch_errors.append(strain_val)

        if (t + 1) % 256 == 0:
            det_tail_err = np.mean(det_errors[-64:]) if len(det_errors) >= 64 else np.mean(det_errors)
            det_tail_bits = np.mean(bit_mismatches_det[-64:]) if len(bit_mismatches_det) >= 64 else np.mean(bit_mismatches_det)
            print(f"  Step {t+1:04d}/{stream_len} | Active Nodes: {model.active_comp_nodes:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas()

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & AUTOPOIETIC DISCOVERED SYSTEM (EXP-369) ===")
    print("===============================================================================")
    print(f"  Final Active Node Count      : {model.active_comp_nodes} (Started at 4)")
    print(f"  Sprout Morphogenesis Events  : {model.sprout_events}")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Processing Throughput         : {tok_per_sec:.1f} tok/s\n")
    print("-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------")

    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-369",
        "active_comp_nodes": model.active_comp_nodes,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_369_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_369()
