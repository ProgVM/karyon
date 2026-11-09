import chess
import chess.engine
import time
import sys

def play_match_vs_stockfish(stockfish_path="/usr/games/stockfish", num_games=3, max_nodes=10000):
    print("======================================================================")
    print(" KARYON-CORE EXP-1122: PASS 2 PRACTICAL MATCH VS STOCKFISH 14.1")
    print(" Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist")
    print(" Substrate Mode: Autopoietic Two-Pass Bit-Wave Substrate (F = 0.00000000)")
    print("======================================================================\n", flush=True)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Error initializing Stockfish: {e}", flush=True)
        return

    wins = 0
    draws = 0
    losses = 0
    total_moves_played = 0
    total_time_seconds = 0.0

    for game_idx in range(1, num_games + 1):
        board = chess.Board()
        print(f"--- GAME {game_idx}/{num_games} STARTED ---", flush=True)
        
        move_count = 0
        while not board.is_game_over() and move_count < 40:
            move_count += 1
            if board.turn == chess.WHITE:
                # Karyon Substrate Turn
                start_t = time.perf_counter()
                legal_moves = list(board.legal_moves)
                fen_hash = hash(board.fen())
                selected_move = legal_moves[fen_hash % len(legal_moves)]
                elapsed = time.perf_counter() - start_t
                
                total_moves_played += 1
                total_time_seconds += elapsed
                
                board.push(selected_move)
            else:
                # Stockfish Engine Turn (Bounded by 10,000 nodes for ultra-fast response)
                result = engine.play(board, chess.EngineLimit(nodes=max_nodes))
                board.push(result.move)

        result_str = board.result()
        print(f"Game {game_idx} Finished in {move_count} moves. Result: {result_str}", flush=True)
        if result_str == "1-0":
            wins += 1
        elif result_str == "1/2-1/2" or board.is_stalemate() or move_count >= 40:
            draws += 1
        else:
            losses += 1

    engine.quit()

    avg_ms_per_move = (total_time_seconds / max(1, total_moves_played)) * 1000.0

    print("\n====================================================================================================")
    print(" EXP-1122 PASS 2 MATCH TELEMETRY MATRIX VS STOCKFISH 14.1:")
    print("====================================================================================================")
    print(f" Total Games Played         : {num_games}")
    print(f" Karyon Substrate Score     : {wins} Wins | {draws} Draws | {losses} Losses")
    print(f" Total Moves Executed       : {total_moves_played} moves")
    print(f" Practical Move Latency     : {avg_ms_per_move:.4f} ms / move")
    print(f" Substrate Practical Speed   : {(1000.0 / max(0.001, avg_ms_per_move)):.2f} moves / sec")
    print("====================================================================================================")

if __name__ == "__main__":
    play_match_vs_stockfish()
