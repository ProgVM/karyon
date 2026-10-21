#!/usr/bin/env python3
"""
EXP-667: Attractor-Sharpened Continuous Latent Tree Search (AS-LTS) vs Stockfish 14.1
Evaluating Dynamic Dopaminergic Precision Sharpening (beta: 2.0 -> 24.0) inside
Continuous Hopfield Energy Landscapes for Discrete Combinatorial Games.
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

class AttractorSharpenedChessEngine:
    def __init__(self, beta_base=12.0, num_basins=64, dim=768):
        self.beta_base = beta_base
        self.dim = dim
        self.num_basins = num_basins
        
        # Piece material values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Synthesize random orthogonal discrete attractor basins on S^{767}
        np.random.seed(667)
        raw_basins = np.random.randn(num_basins, dim).astype(np.float32)
        norms = np.linalg.norm(raw_basins, axis=1, keepdims=True)
        self.basins = raw_basins / np.maximum(norms, 1e-6)

    def encode_board_to_hypersphere(self, board: chess.Board) -> np.ndarray:
        tensor = np.zeros(768, dtype=np.float32)
        piece_type_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                offset = piece_type_to_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    offset += 6
                tensor[offset * 64 + sq] = 1.0
        
        norm = np.linalg.norm(tensor)
        if norm > 1e-6:
            tensor /= norm
        return tensor

    def compute_hopfield_attractor_energy(self, state_vec: np.ndarray, dopamine_da: float = 0.5) -> float:
        # Effective temperature sharpening: beta = beta_base * (1.0 + 2.0 * DA)
        beta_eff = self.beta_base * (1.0 + 2.0 * dopamine_da)
        
        # Inner product with attractor basins (cosines on S^{767})
        cosines = self.basins @ state_vec
        
        # Continuous Log-Sum-Exp Hopfield Energy: E = -(1/beta) * log(sum(exp(beta * cos)))
        max_cos = np.max(cosines)
        lse = max_cos + (1.0 / beta_eff) * np.log(np.sum(np.exp(beta_eff * (cosines - max_cos))))
        hopfield_energy = -lse
        return float(hopfield_energy)

    def static_material_eval(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -99999.0 if board.turn else 99999.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                val = self.piece_values[piece.piece_type]
                score += val if piece.color == chess.WHITE else -val

        return score if board.turn == chess.WHITE else -score

    def latent_mcts_rollout(self, board: chess.Board, depth: int, dopamine_da: float = 0.8) -> float:
        """
        System 2 Latent Tree Search:
        Evaluates Expected Free Energy G = -Material + beta * HopfieldAttractorEnergy
        """
        if depth == 0 or board.is_game_over():
            state_vec = self.encode_board_to_hypersphere(board)
            e_attractor = self.compute_hopfield_attractor_energy(state_vec, dopamine_da)
            mat_score = self.static_material_eval(board)
            # Expected Free Energy G
            return mat_score - 50.0 * e_attractor

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.static_material_eval(board)

        best_v = -float("inf")
        for move in legal_moves:
            board.push(move)
            v = -self.latent_mcts_rollout(board, depth - 1, dopamine_da)
            board.pop()
            if v > best_v:
                best_v = v
        return best_v

    def select_best_move(self, board: chess.Board, search_depth: int = 3, dopamine_da: float = 0.8) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = None
        best_value = -float("inf")

        for move in legal_moves:
            board.push(move)
            # Active Inference System 2 Rollout
            val = -self.latent_mcts_rollout(board, search_depth - 1, dopamine_da)
            board.pop()

            if val > best_value:
                best_value = val
                best_move = move

        return best_move

def play_match(karyon_color=chess.WHITE, sf_skill=1, max_moves=40):
    board = chess.Board()
    engine_karyon = AttractorSharpenedChessEngine(beta_base=16.0)
    
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Skill Level": sf_skill})

    print(f"\n[EXP-667 MATCH] Karyon AS-LTS ({'White' if karyon_color == chess.WHITE else 'Black'}) vs Stockfish (Skill {sf_skill})")

    ply = 0
    while not board.is_game_over() and ply < max_moves:
        ply += 1
        if board.turn == karyon_color:
            t0 = time.perf_counter()
            # High Dopamine DA=1.0 sharpens attractor basins into discrete wells
            move = engine_karyon.select_best_move(board, search_depth=3, dopamine_da=1.0)
            dt = (time.perf_counter() - t0) * 1000.0
            board.push(move)
            print(f"Ply {ply:02d} | Karyon AS-LTS: {move.uci()} ({dt:.1f}ms)")
        else:
            result = sf.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
            print(f"Ply {ply:02d} | Stockfish    : {result.move.uci()}")

    sf.quit()
    res = board.result() if board.is_game_over() else "1/2-1/2 (Limit)"
    print(f"[MATCH END] Final Result: {res} after {ply} plies")
    print(board)
    return res

if __name__ == "__main__":
    play_match(karyon_color=chess.WHITE, sf_skill=1)
