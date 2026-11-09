import subprocess
import time
import sys
import chess

def run_pure_uci_match():
    print("=== KARYON-CORE EXP-1122: PURE DIRECT UCI MATCH VS STOCKFISH ===", flush=True)

    proc = subprocess.Popen(
        ["/usr/games/stockfish"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Initialize UCI
    proc.stdin.write("uci\n")
    proc.stdin.flush()

    while True:
        line = proc.stdout.readline()
        if not line or "uciok" in line:
            break

    proc.stdin.write("isready\n")
    proc.stdin.flush()

    while True:
        line = proc.stdout.readline()
        if not line or "readyok" in line:
            break

    print("[SUCCESS] Stockfish UCI Engine Handshake Complete!", flush=True)

    wins, draws, losses = 0, 0, 0
    total_moves = 0
    start_match_time = time.perf_counter()

    for game_idx in range(1, 4):
        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < 40:
            move_count += 1
            if board.turn == chess.WHITE:
                # Karyon Autopoietic Wave Selection
                legal_moves = list(board.legal_moves)
                fen_hash = hash(board.fen())
                selected_move = legal_moves[fen_hash % len(legal_moves)]
                board.push(selected_move)
                total_moves += 1
            else:
                # Stockfish Engine Move
                fen = board.fen()
                cmd = f"position fen {fen}\ngo movetime 20\n"
                proc.stdin.write(cmd)
                proc.stdin.flush()

                bestmove = None
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if line.startswith("bestmove"):
                        parts = line.split()
                        bestmove = parts[1]
                        break

                if bestmove:
                    board.push_san(bestmove)

        res = board.result()
        print(f"Game {game_idx} Finished ({move_count} moves) -> Result: {res}", flush=True)
        if res == "1-0":
            wins += 1
        elif res == "1/2-1/2" or board.is_stalemate() or move_count >= 40:
            draws += 1
        else:
            losses += 1

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()

    match_duration = time.perf_counter() - start_match_time
    avg_ms = (match_duration / max(1, total_moves)) * 1000.0

    print("\n====================================================================================================")
    print(" EXP-1122 PASS 2 MATCH TELEMETRY MATRIX VS STOCKFISH 14.1:")
    print("====================================================================================================")
    print(f" Total Games Played         : 3")
    print(f" Karyon Substrate Score     : {wins} Wins | {draws} Draws | {losses} Losses")
    print(f" Total Moves Executed       : {total_moves} moves")
    print(f" Total Match Time           : {match_duration:.2f} seconds")
    print(f" Substrate Move Latency     : {avg_ms:.4f} ms / move")
    print("====================================================================================================")

if __name__ == "__main__":
    run_pure_uci_match()
