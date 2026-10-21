#!/usr/bin/env python3
"""
EXP-666: Karyon Holo-Manifold Active Inference Engine vs Stockfish 14.1
Calibrated 768D Tensor Board Mapping & System 2 Latent Rollout
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

class CalibratedHoloChessEngine:
    def __init__(self, beta_parity=15.0, alpha_material=1.0, alpha_position=0.25):
        self.beta_parity = beta_parity
        self.alpha_material = alpha_material
        self.alpha_position = alpha_position
        
        # Piece Values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Simplified Piece-Square Tables (PST) for Positional Control
        self.pawn_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
           50, 50, 50, 50, 50, 50, 50, 50,
           10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 27, 27, 10,  5,  5,
            0,  0,  0, 25, 25,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-25,-25, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ], dtype=np.float32)

        self.knight_table = np.array([
          -50,-40,-30,-30,-30,-30,-40,-50,
          -40,-20,  0,  0,  0,  0,-20,-40,
          -30,  0, 10, 15, 15, 10,  0,-30,
          -30,  5, 15, 20, 20, 15,  5,-30,
          -30,  0, 15, 20, 20, 15,  0,-30,
          -30,  5, 10, 15, 15, 10,  5,-30,
          -40,-20,  0,  5,  5,  0,-20,-40,
          -50,-40,-30,-30,-30,-30,-40,-50
        ], dtype=np.float32)

        # 768D (12 piece types x 64 squares) Riemannian Projection Matrices H1, H2
        np.random.seed(666)
        self.H1 = np.random.randn(64, 768).astype(np.float32) / math.sqrt(768)
        self.H2 = np.random.randn(64, 768).astype(np.float32) / math.sqrt(768)

    def encode_768d_tensor(self, board: chess.Board) -> np.ndarray:
        # 12 piece planes: W_P, W_N, W_B, W_R, W_Q, W_K, B_P, B_N, B_B, B_R, B_Q, B_K
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
        
        # Unit sphere normalization S^{767}
        norm = np.linalg.norm(tensor)
        if norm > 1e-6:
            tensor /= norm
        return tensor

    def compute_holo_syndrome_energy(self, tensor_768d: np.ndarray) -> float:
        p1 = self.H1 @ tensor_768d
        p2 = self.H2 @ (tensor_768d ** 2)
        return float(np.sum(p1 ** 2) + 0.5 * np.sum(p2 ** 2))

    def static_evaluation(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -99999.0 if board.turn else 99999.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                val = self.piece_values[piece.piece_type]
                pst = 0.0
                if piece.piece_type == chess.PAWN:
                    pst = self.pawn_table[sq] if piece.color == chess.WHITE else self.pawn_table[chess.square_mirror(sq)]
                elif piece.piece_type == chess.KNIGHT:
                    pst = self.knight_table[sq] if piece.color == chess.WHITE else self.knight_table[chess.square_mirror(sq)]

                tot = val + pst * self.alpha_position
                score += tot if piece.color == chess.WHITE else -tot

        return score if board.turn == chess.WHITE else -score

    def alpha_beta_rollout(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            # Combine static evaluation with Riemannian Holo-Parity Syndrome Energy
            tensor_768 = self.encode_768d_tensor(board)
            e_syndrome = self.compute_holo_syndrome_energy(tensor_768)
            static_v = self.static_evaluation(board)
            # Free Energy F_t = -StaticValue + beta_parity * e_syndrome
            return static_v - self.beta_parity * e_syndrome

        legal_moves = list(board.legal_moves)
        if maximizing:
            max_eval = -float("inf")
            for move in legal_moves:
                board.push(move)
                eval_v = self.alpha_beta_rollout(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_v)
                alpha = max(alpha, eval_v)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                board.push(move)
                eval_v = self.alpha_beta_rollout(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_v)
                beta = min(beta, eval_v)
                if beta <= alpha:
                    break
            return min_eval

    def select_best_move(self, board: chess.Board, search_depth: int = 3) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = None
        best_value = -float("inf")

        for move in legal_moves:
            board.push(move)
            # System 2 Active Inference Latent Search
            val = self.alpha_beta_rollout(board, search_depth - 1, -float("inf"), float("inf"), False)
            board.pop()

            if val > best_value:
                best_value = val
                best_move = move

        return best_move

def play_calibrated_match(karyon_color=chess.WHITE, sf_skill=1, max_moves=60):
    board = chess.Board()
    engine_karyon = CalibratedHoloChessEngine(beta_parity=2.0)
    
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Skill Level": sf_skill})

    print(f"\n[CALIBRATED MATCH] Karyon ({'White' if karyon_color == chess.WHITE else 'Black'}) vs Stockfish (Skill {sf_skill})")

    ply = 0
    while not board.is_game_over() and ply < max_moves:
        ply += 1
        if board.turn == karyon_color:
            t0 = time.perf_counter()
            move = engine_karyon.select_best_move(board, search_depth=3)
            dt = (time.perf_counter() - t0) * 1000.0
            board.push(move)
            print(f"Ply {ply:02d} | Karyon: {move.uci()} ({dt:.1f}ms)")
        else:
            result = sf.play(board, chess.engine.Limit(time=0.02))
            board.push(result.move)
            print(f"Ply {ply:02d} | Stockfish: {result.move.uci()}")

    sf.quit()
    res = board.result() if board.is_game_over() else "1/2-1/2 (Limit)"
    print(f"[MATCH END] Final Result: {res} after {ply} plies")
    print(board)
    return res

if __name__ == "__main__":
    play_calibrated_match(karyon_color=chess.WHITE, sf_skill=1)
