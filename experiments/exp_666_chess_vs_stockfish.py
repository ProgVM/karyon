#!/usr/bin/env python3
"""
EXP-666: Karyon Holo-Manifold Active Inference vs Stockfish 14.1
Full Closed-Loop Match Orchestrator
"""

import os
import sys
import math
import json
import time
import random
import chess
import chess.engine
import numpy as np

STOCKFISH_PATH = "/usr/games/stockfish"

class KaryonHoloChessAgent:
    def __init__(self, dim=64, beta_parity=12.0):
        self.dim = dim
        self.beta_parity = beta_parity
        np.random.seed(42)
        # 1st-order linear positional matrix H1 and 2nd-order bilinear H2
        self.H1 = np.random.randn(dim, 64).astype(np.float32) / math.sqrt(64)
        self.H2 = np.random.randn(dim, 64).astype(np.float32) / math.sqrt(64)
        
        # Piece material valuation weights
        self.piece_weights = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 100.0
        }

    def encode_board_to_hypersphere(self, board: chess.Board) -> np.ndarray:
        vec = np.zeros(64, dtype=np.float32)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                val = self.piece_weights[piece.piece_type]
                vec[sq] = val if piece.color == chess.WHITE else -val
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def compute_holo_syndrome_energy(self, state_vec: np.ndarray) -> float:
        p1 = self.H1 @ state_vec
        p2 = self.H2 @ (state_vec ** 2)
        energy = float(np.sum(p1 ** 2) + 0.5 * np.sum(p2 ** 2))
        return energy

    def evaluate_board_material(self, board: chess.Board, for_color: chess.Color) -> float:
        score = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                w = self.piece_weights[piece.piece_type]
                score += w if piece.color == chess.WHITE else -w
        return score if for_color == chess.WHITE else -score

    def select_move_active_inference(self, board: chess.Board, k_steps: int = 2) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = None
        min_expected_free_energy = float("inf")

        color = board.turn
        for move in legal_moves:
            # Latent mental rollout (System 2 step)
            board.push(move)
            
            # Evaluate projected state on S^{D-1}
            state_vec = self.encode_board_to_hypersphere(board)
            e_syndrome = self.compute_holo_syndrome_energy(state_vec)
            mat_eval = self.evaluate_board_material(board, color)
            
            # Additional heuristic: check bonus / center square control
            center_bonus = 0.2 if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5] else 0.0
            
            # Expected Free Energy G_t:
            # G = -MaterialAdvantage + ParityViolation - CenterBonus
            expected_free_energy = -mat_eval + self.beta_parity * e_syndrome - center_bonus

            board.pop()

            if expected_free_energy < min_expected_free_energy:
                min_expected_free_energy = expected_free_energy
                best_move = move

        return best_move

def play_match(karyon_color=chess.WHITE, sf_time_limit=0.01, max_moves=60):
    board = chess.Board()
    agent = KaryonHoloChessAgent()
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # Configure Stockfish Skill Level for realistic benchmark baseline (Skill Level 3)
    engine.configure({"Skill Level": 3})

    move_history = []
    print(f"\n[MATCH START] Karyon ({'White' if karyon_color == chess.WHITE else 'Black'}) vs Stockfish 14.1")

    step = 0
    while not board.is_game_over() and step < max_moves:
        step += 1
        if board.turn == karyon_color:
            t0 = time.perf_counter()
            move = agent.select_move_active_inference(board)
            dt = (time.perf_counter() - t0) * 1000.0
            board.push(move)
            move_history.append((f"Karyon: {move.uci()}", f"{dt:.2f}ms"))
        else:
            result = engine.play(board, chess.engine.Limit(time=sf_time_limit))
            board.push(result.move)
            move_history.append((f"Stockfish: {result.move.uci()}", "SF"))

    engine.quit()

    result_str = board.result() if board.is_game_over() else "1/2-1/2 (Move limit reached)"
    print(f"[MATCH FINISHED] Total moves: {step} | Result: {result_str}")
    print(board)
    return {
        "result": result_str,
        "moves_played": step,
        "fen": board.fen()
    }

def run_tournament(num_games=10):
    print("=" * 80)
    print(f"=== EXP-666: KARYON-CORE HOLO-MANIFOLD VS STOCKFISH TOURNAMENT ({num_games} GAMES) ===")
    print("=" * 80)

    results = []
    karyon_score = 0.0
    sf_score = 0.0

    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        res = play_match(karyon_color=color, max_moves=50)
        results.append(res)
        if "1-0" in res["result"]:
            if color == chess.WHITE: karyon_score += 1.0
            else: sf_score += 1.0
        elif "0-1" in res["result"]:
            if color == chess.BLACK: karyon_score += 1.0
            else: sf_score += 1.0
        else:
            karyon_score += 0.5
            sf_score += 0.5

    print("\n" + "=" * 80)
    print(f"=== TOURNAMENT SCOREBOARD: KARYON {karyon_score:.1f} - {sf_score:.1f} STOCKFISH ===")
    print(f"Games: {num_games} | Win Rate: {(karyon_score / num_games) * 100:.1f}%")
    print("=" * 80)

    telemetry = {
        "exp_id": "EXP-666",
        "num_games": num_games,
        "karyon_score": karyon_score,
        "stockfish_score": sf_score,
        "win_rate": (karyon_score / num_games) * 100.0,
        "results": results
    }
    with open("experiments/exp_666_telemetry.json", "w") as f:
        json.dump(telemetry, f, indent=2)

if __name__ == "__main__":
    run_tournament(num_games=5)
