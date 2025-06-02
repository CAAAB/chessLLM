import chess
import chess.svg
import chess.engine
import random
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, SVG, HTML # Keep for potential non-pygame use if needed
import pygame
import sys
import time # For timing analysis if needed
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting needed for pygame integration

# --- Pygame & Display Configuration ---
pygame.init()
pygame.font.init() # Initialize font module

# Screen dimensions
BOARD_SIZE = 512 # Pixel size of the board
SQ_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = 100 # Space below board for text info
PLOT_PANEL_HEIGHT = 150 # Space below info for the plot
EVAL_BAR_WIDTH_PX = 30 # Width of the eval bar in pixels
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20 # Space to the right for eval bar + padding

SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60) # Background for bar/plot
LIGHT_SQ_COLOR = (238, 238, 210) # Example light square color
DARK_SQ_COLOR = (118, 150, 86) # Example dark square color
HIGHLIGHT_COLOR = (255, 255, 0, 150) # Yellow highlight with transparency
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70) # Semi-transparent black circle
ORANGE = (255, 165, 0)

# Font
INFO_FONT_SIZE = 18
try:
    INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
except Exception as e:
    print(f"Warning: Could not load monospace font ({e}). Using default.")
    INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE+2) # Default fallback

# --- Chess Configuration ---
# STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Adjust as needed
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Example path
ANALYSIS_TIME_LIMIT = 0.5
EVAL_BAR_HEIGHT_FRAC = BOARD_SIZE # Eval bar matches board height
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = 15000 # Same as before

# --- Asset Loading ---
PIECE_IMAGE_PATH = "pieces" # FOLDER containing piece images (e.g., wP.png, bN.png)
PIECE_IMAGES = {}

def load_piece_images(path=PIECE_IMAGE_PATH, sq_size=SQ_SIZE):
    """Loads piece images from a folder and scales them."""
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    loaded_images = {}
    if not os.path.isdir(path):
        print(f"Error: Piece image directory not found: '{path}'")
        print("Please create this directory and add piece images (e.g., wP.png, bN.png...).")
        return None # Indicate failure

    for piece in pieces:
        file_path = os.path.join(path, f"{piece}.png")
        try:
            img = pygame.image.load(file_path).convert_alpha() # Load with transparency
            img = pygame.transform.smoothscale(img, (sq_size, sq_size)) # Scale smoothly
            loaded_images[piece] = img
        except pygame.error as e:
            print(f"Error loading piece image '{file_path}': {e}")
            print("Ensure all 12 piece PNG files exist in the specified directory.")
            return None # Indicate failure
    print(f"Loaded {len(loaded_images)} piece images.")
    return loaded_images

# --- Helper Functions (Chess Logic - Mostly Unchanged) ---
def get_castling_rights_str(board):
    rights = []
    if board.has_kingside_castling_rights(chess.WHITE): rights.append("K")
    if board.has_queenside_castling_rights(chess.WHITE): rights.append("Q")
    if board.has_kingside_castling_rights(chess.BLACK): rights.append("k")
    if board.has_queenside_castling_rights(chess.BLACK): rights.append("q")
    return "".join(rights) if rights else "-"

def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    if score is None: return None
    pov_score = score.white()
    if pov_score.is_mate():
        mate_val = pov_score.mate()
        return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE)
    if cp is None: return None
    clamped_cp = max(-clamp_limit, min(clamp_limit, cp))
    normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
    return normalized * 100.0

def get_engine_analysis(board, engine, time_limit):
    if not engine:
        return None, "Analysis unavailable (Engine not loaded).", None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_move = info.get("pv", [None])[0]
        score = info.get("score")
        if best_move and score is not None:
            pov_score = score.pov(board.turn)
            score_str = ""
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                score_str = f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate (N/A)"
            else:
                cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
                score_str = f"Score: {cp / 100.0:+.2f}" if cp is not None else "Score: N/A"
            try: best_move_san = board.san(best_move)
            except Exception: best_move_san = best_move.uci()
            return best_move_san, score_str, score
        else:
            return None, "Engine could not provide suggestion.", None
    except chess.engine.EngineTerminatedError:
         print("Engine terminated unexpectedly.")
         return None, "Engine terminated.", None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None, f"Analysis error: {e}", None

# --- Drawing Functions (Pygame) ---

def draw_board(surface):
    """Draws the checkerboard squares."""
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(surface, board, piece_images, dragging_piece_info):
    """Draws the pieces on the board, handling the dragged piece."""
    if piece_images is None: return # Don't draw if images failed to load

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Don't draw the piece being dragged at its original square
            if dragging_piece_info and square == dragging_piece_info['square']:
                continue

            piece_symbol = piece.symbol()
            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
            img = piece_images.get(piece_key)
            if img:
                # Pygame coords: (0,0) is top-left. chess squares: a1=0, h8=63.
                # Rank 0..7 maps to y 0..7. File 0..7 maps to x 0..7.
                # chess.square_rank(sq) gives 0 (rank 1) to 7 (rank 8)
                # chess.square_file(sq) gives 0 (file a) to 7 (file h)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                # Need to invert rank for y-coordinate: y = (7 - rank) * SQ_SIZE
                screen_x = file * SQ_SIZE
                screen_y = (7 - rank) * SQ_SIZE
                surface.blit(img, (screen_x, screen_y))

    # Draw the dragging piece last, at the mouse cursor position
    if dragging_piece_info and dragging_piece_info['img']:
        # Center the image on the cursor
        img_rect = dragging_piece_info['img'].get_rect(center=dragging_piece_info['pos'])
        surface.blit(dragging_piece_info['img'], img_rect)

def draw_highlights(surface, board, selected_square, legal_moves_for_selected):
    """Highlights the selected square and possible move destinations."""
    # Highlight selected square
    if selected_square is not None:
        rank = chess.square_rank(selected_square)
        file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        # Draw semi-transparent highlight
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT_COLOR)
        surface.blit(s, highlight_rect.topleft)

    # Highlight legal moves for the selected piece
    if legal_moves_for_selected:
        for move in legal_moves_for_selected:
            dest_sq = move.to_square
            rank = chess.square_rank(dest_sq)
            file = chess.square_file(dest_sq)
            center_x = file * SQ_SIZE + SQ_SIZE // 2
            center_y = (7 - rank) * SQ_SIZE + SQ_SIZE // 2
            # Draw semi-transparent circle
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA) # Surface for transparency
            pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6)
            surface.blit(s, (file * SQ_SIZE, (7 - rank) * SQ_SIZE))


def draw_eval_bar(surface, white_percentage):
    """Draws the vertical evaluation bar."""
    bar_x = BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2 # Center in side panel
    bar_y = 0
    bar_height = BOARD_SIZE # Match board height

    if white_percentage is None:
        white_percentage = 50.0 # Default to 50/50

    white_height = int(bar_height * (white_percentage / 100.0))
    black_height = bar_height - white_height

    # Draw black part (top)
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    # Draw white part (bottom)
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    # Draw border
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

def create_eval_plot_surface(eval_history_percent, plot_width_px, plot_height_px):
    """Generates the eval plot PNG and returns it as a Pygame surface."""
    if not eval_history_percent or len(eval_history_percent) < 2:
        return None # Not enough data to plot

    plies = [item[0] for item in eval_history_percent]
    percentages = [item[1] if item[1] is not None else 50.0 for item in eval_history_percent]

    # --- Convert Pygame colors to Matplotlib format (0.0-1.0) ---
    # Divide each component by 255.0
    dark_grey_mpl = tuple(c / 255.0 for c in DARK_GREY)
    orange_mpl = tuple(c / 255.0 for c in ORANGE)
    grey_mpl = tuple(c / 255.0 for c in GREY)
    light_grey_mpl = (211/255.0, 211/255.0, 211/255.0) # Example for lightgrey name
    white_mpl = (1.0, 1.0, 1.0)
    # You can also use hex strings directly which Matplotlib understands:
    # dark_grey_mpl = '#3C3C3C' # Hex for (60, 60, 60)
    # orange_mpl = '#FFA500'
    # etc. Using normalized tuples is fine too.
    # -----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(plot_width_px / 80, plot_height_px / 80), dpi=80)

    # --- Use the converted Matplotlib colors ---
    fig.patch.set_facecolor(dark_grey_mpl) # Use normalized tuple
    ax.set_facecolor(dark_grey_mpl)        # Use normalized tuple

    # Matplotlib understands common color names directly, or use converted ones
    ax.fill_between(plies, percentages, color=white_mpl, alpha=0.9) # 'white' is fine, or white_mpl
    ax.axhline(50, color=orange_mpl, linestyle='-', linewidth=1.5) # Use orange_mpl

    ax.set_xlabel("Ply", color=light_grey_mpl) # Use converted or hex/name
    ax.set_ylabel("White Win %", color=light_grey_mpl)
    ax.set_title("Win Probability History", color=white_mpl, fontsize=10)
    ax.set_xlim(0, max(plies) if plies else 1)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', colors=light_grey_mpl, labelsize=8)
    ax.tick_params(axis='y', colors=light_grey_mpl, labelsize=8)
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl) # Use grey_mpl
    ax.spines['top'].set_color(grey_mpl)
    ax.spines['bottom'].set_color(grey_mpl)
    ax.spines['left'].set_color(grey_mpl)
    ax.spines['right'].set_color(grey_mpl)
    # ------------------------------------------

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    # Save with the correct background color (get_facecolor returns what was set)
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    try:
        # Load into Pygame surface
        plot_surface = pygame.image.load(buf, 'png').convert()
        # Optional: Set colorkey for transparency if plot PNG has it
        # plot_surface.set_colorkey(THE_TRANSPARENT_COLOR_IF_ANY)
    except pygame.error as e:
        print(f"Error loading plot image into Pygame surface: {e}")
        plot_surface = None # Ensure it's None on error
    finally:
         buf.close()

    return plot_surface


def draw_game_info(surface, board, analysis_text, message, font):
    """Draws text information below the board."""
    start_y = BOARD_SIZE + 5 # Starting y-coordinate for text
    line_height = font.get_height() + 2

    lines = []
    turn = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"Turn: {turn}")
    if board.is_check(): lines.append("!! CHECK !!")
    try:
        last_move_san = board.san(board.peek()) if board.move_stack else "N/A"
    except:
        last_move_san = board.peek().uci() if board.move_stack else "N/A"
    lines.append(f"Last Move: {last_move_san}")
    lines.append(f"Ply: {len(board.move_stack)} | Move: {board.fullmove_number}")
    lines.append(f"Castling: {get_castling_rights_str(board)}")

    if analysis_text:
        lines.append("-" * 10)
        lines.append(analysis_text)
    if message:
        lines.append("-" * 10)
        lines.append(f"Msg: {message}")
    if board.is_game_over(claim_draw=True):
         lines.append("-" * 10)
         res = board.outcome(claim_draw=True)
         term = res.termination.name.replace('_',' ').title() if res else "N/A"
         lines.append(f"GAME OVER: {res.result() if res else board.result(claim_draw=True)} ({term})")

    # Render lines
    info_rect = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, INFO_PANEL_HEIGHT)
    surface.fill(DARK_GREY, info_rect) # Background for info panel

    for i, line in enumerate(lines):
        text_surface = font.render(line, True, WHITE) # White text
        surface.blit(text_surface, (5, start_y + i * line_height))

# --- Coordinate Conversion ---

def screen_to_square(pos):
    """Converts pixel coordinates (x, y) to a chess square index (0-63)."""
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None # Click outside board
    file = x // SQ_SIZE
    rank = 7 - (y // SQ_SIZE) # Invert rank for chess indexing
    return chess.square(file, rank)

# --- Main Game Function ---

def play_chess960_pygame():
    global PIECE_IMAGES # Allow modification
    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None:
        print("Exiting due to missing piece images.")
        return # Exit if images aren't loaded

    board = chess.Board(chess960=True)
    engine = None
    message = None
    start_pos_num = -1
    eval_history_percent = [] # Stores (ply, white_percentage)
    last_raw_score = None
    one_off_analysis_text = None
    meter_visible = True # Default to visible in GUI
    plot_surface = None # To store the rendered plot
    needs_plot_update = True # Flag to regenerate plot

    # --- Get Starting Position (Simple console input before Pygame window) ---
    while True:
        # clear_output() # Not applicable here
        print("\n--- Chess960 Setup ---")
        pos_choice = input("Enter Chess960 position number (0-959), 'random', or leave blank for random: ").strip().lower()
        if not pos_choice or pos_choice == 'random':
            start_pos_num = random.randint(0, 959)
            break
        else:
            try:
                start_pos_num = int(pos_choice)
                if 0 <= start_pos_num <= 959:
                    break
                else:
                    print("Position number must be between 0 and 959.")
            except ValueError:
                print("Invalid input.")

    board.set_chess960_pos(start_pos_num)
    start_fen = board.fen()
    print(f"Starting Chess960 Position {start_pos_num} (FEN: {start_fen})")
    print("Initializing Pygame window...")
    time.sleep(1) # Brief pause

    # --- Initialize Engine ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try:
                engine.configure({"UCI_Chess960": True})
                print("Engine: Set UCI_Chess960 option.")
            except Exception:
                 print("Engine: UCI_Chess960 option not supported/failed (usually ok).")
            print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
            # Get initial analysis
            _, _, last_raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT)
            if last_raw_score:
                 percent = score_to_white_percentage(last_raw_score, EVAL_CLAMP_LIMIT)
                 eval_history_percent.append((0, percent if percent is not None else 50.0))
                 needs_plot_update = True

        else:
            print(f"Warning: Stockfish not found ('{STOCKFISH_PATH}'). Analysis disabled.")
            message = "Engine unavailable"
    except Exception as e:
        print(f"Error initializing Stockfish: {e}. Analysis disabled.")
        engine = None
        message = "Engine init failed"

    # --- Pygame Loop Variables ---
    running = True
    clock = pygame.time.Clock()
    selected_square = None
    dragging_piece_info = None # Store {'square': sq, 'piece': piece, 'img': img, 'pos': (x,y)}
    legal_moves_for_selected = [] # Store list of legal moves from selected_square

    # --- Main Game Loop ---
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    pos = pygame.mouse.get_pos()
                    sq = screen_to_square(pos)

                    if sq is not None and not board.is_game_over():
                        piece = board.piece_at(sq)
                        # Check if clicking on a piece of the current turn's color
                        if piece and piece.color == board.turn:
                            selected_square = sq
                            piece_symbol = piece.symbol()
                            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                            dragging_piece_info = {
                                'square': sq,
                                'piece': piece,
                                'img': PIECE_IMAGES.get(piece_key),
                                'pos': pos
                            }
                            # Get legal moves for highlighting
                            legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                        else:
                            # Clicking on empty square or opponent's piece clears selection
                            selected_square = None
                            dragging_piece_info = None
                            legal_moves_for_selected = []

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging_piece_info: # Left release while dragging
                    pos = pygame.mouse.get_pos()
                    to_sq = screen_to_square(pos)
                    from_sq = dragging_piece_info['square']
                    move_made = False

                    if to_sq is not None and from_sq != to_sq:
                        # Check for promotion
                        promotion = None
                        piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN:
                            target_rank = 7 if piece.color == chess.WHITE else 0
                            if chess.square_rank(to_sq) == target_rank:
                                promotion = chess.QUEEN # Default to Queen for simplicity

                        # Create the move object
                        move = chess.Move(from_sq, to_sq, promotion=promotion)

                        # Check if legal and make move
                        if move in board.legal_moves:
                            board.push(move)
                            move_made = True
                            message = None # Clear previous messages
                            one_off_analysis_text = None
                            # Get analysis for the new position
                            if engine:
                                _, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT)
                                if raw_score is not None:
                                    last_raw_score = raw_score
                                    percent = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                                    current_ply = len(board.move_stack)
                                    eval_history_percent.append((current_ply, percent if percent is not None else 50.0))
                                    needs_plot_update = True
                                else:
                                    # Handle case where analysis fails after move
                                    last_raw_score = None
                                    current_ply = len(board.move_stack)
                                    # Append 50% or keep last value? Append 50 is safer.
                                    eval_history_percent.append((current_ply, 50.0))
                                    needs_plot_update = True
                                    message = "Analysis failed post-move."
                                one_off_analysis_text = f"Analysis: {score_str}" # Show brief analysis
                            else:
                                # No engine, still need to add a placeholder if meter is on
                                current_ply = len(board.move_stack)
                                eval_history_percent.append((current_ply, 50.0))
                                needs_plot_update = True


                        else:
                            message = f"Illegal move: {chess.square_name(from_sq)}{chess.square_name(to_sq)}"
                            # Piece snaps back automatically as dragging_piece_info is cleared

                    # Reset dragging state regardless of legality
                    dragging_piece_info = None
                    selected_square = None
                    legal_moves_for_selected = []


            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info:
                    dragging_piece_info['pos'] = event.pos # Update position for drawing

            elif event.type == pygame.KEYDOWN:
                 # --- Keyboard Commands ---
                if event.key == pygame.K_u: # Undo
                    if board.move_stack:
                        try:
                            board.pop()
                            message = "Last move undone."
                            if eval_history_percent:
                                eval_history_percent.pop()
                                needs_plot_update = True
                            # Re-analyze previous position
                            if engine:
                                _, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT)
                                last_raw_score = raw_score
                                one_off_analysis_text = f"Analysis: {score_str}"
                            else:
                                last_raw_score = None
                                one_off_analysis_text = None

                        except IndexError: message = "Cannot undo."
                    else: message = "No moves to undo."
                    dragging_piece_info = None # Clear selection after undo
                    selected_square = None
                    legal_moves_for_selected = []

                elif event.key == pygame.K_a: # Analyze
                     if engine:
                         message = "Analyzing..."
                         # Redraw immediately to show "Analyzing..."
                         screen.fill(DARK_GREY)
                         draw_board(screen)
                         draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
                         draw_highlights(screen, board, selected_square, legal_moves_for_selected)
                         draw_eval_bar(screen, score_to_white_percentage(last_raw_score, EVAL_CLAMP_LIMIT))
                         if meter_visible and plot_surface:
                            plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                            screen.blit(plot_surface, plot_rect)
                         draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT)
                         pygame.display.flip()

                         # Run analysis (longer time)
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 3) # Longer analysis
                         if best_move_san:
                             one_off_analysis_text = f"Engine Suggests: {best_move_san} ({score_str})"
                             last_raw_score = raw_score # Update score bar
                             # Update plot history for current ply if meter on
                             if meter_visible and raw_score is not None:
                                current_ply = len(board.move_stack)
                                percent = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                                if percent is not None:
                                     if eval_history_percent and eval_history_percent[-1][0] == current_ply:
                                         if eval_history_percent[-1][1] != percent: # Only update if changed
                                             eval_history_percent[-1] = (current_ply, percent)
                                             needs_plot_update = True
                                     elif not eval_history_percent or eval_history_percent[-1][0] != current_ply :
                                         eval_history_percent.append((current_ply, percent))
                                         needs_plot_update = True
                         else:
                             one_off_analysis_text = score_str # Show error
                         message = None # Clear "Analyzing..." message
                     else:
                         message = "Engine not available for analysis."

                elif event.key == pygame.K_m: # Toggle Meter/Plot
                    meter_visible = not meter_visible
                    message = f"Eval Meter/Plot {'ON' if meter_visible else 'OFF'}"
                    if meter_visible and needs_plot_update: # Regenerate plot if turning on and needed
                         plot_surface = create_eval_plot_surface(eval_history_percent, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                         needs_plot_update = False # Plot is now up-to-date


                elif event.key == pygame.K_ESCAPE: # Quit
                    running = False


        # --- Update ---
        # Regenerate plot if needed and visible
        if meter_visible and needs_plot_update:
            plot_surface = create_eval_plot_surface(eval_history_percent, BOARD_SIZE, PLOT_PANEL_HEIGHT) # Generate surface
            needs_plot_update = False # Reset flag

        # --- Drawing ---
        screen.fill(DARK_GREY) # Background for whole window

        # Draw Board Area
        draw_board(screen)
        draw_highlights(screen, board, selected_square, legal_moves_for_selected)
        draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)

        # Draw Side Panel (Eval Bar)
        draw_eval_bar(screen, score_to_white_percentage(last_raw_score, EVAL_CLAMP_LIMIT))

        # Draw Info Panel
        draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT)

        # Draw Plot Panel (if visible and surface exists)
        if meter_visible and plot_surface:
            # Position plot below the info panel
            plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
            screen.blit(plot_surface, plot_rect)

        # Update the display
        pygame.display.flip()

        # Limit frame rate
        clock.tick(30) # FPS limit

    # --- Cleanup ---
    print("\nExiting Pygame...")
    if engine:
        try: engine.quit(); print("Stockfish engine closed.")
        except Exception: print("Error closing engine or already closed.")
    plt.close('all') # Close any lingering matplotlib plots
    pygame.quit()
    sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()