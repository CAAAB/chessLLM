import chess
import chess.svg
import chess.engine
import random
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display, clear_output, SVG, HTML # No longer needed for GUI
import pygame
import sys
import time
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting needed for pygame integration

# --- Pygame & Display Configuration ---
pygame.init()
pygame.font.init()

# Screen dimensions
BOARD_SIZE = 512
SQ_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = 80 # Reduced height for info text
PLOT_PANEL_HEIGHT = 120 # Slightly reduced plot height
TREE_PANEL_HEIGHT = 150 # New panel for the game tree vis
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20

SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT # Added Tree panel

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board with History Tree")

# Colors (add some for tree)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60)
LIGHT_SQ_COLOR = (238, 238, 210)
DARK_SQ_COLOR = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
ORANGE = (255, 165, 0)
TREE_NODE_COLOR = (100, 100, 200) # Color for tree nodes
TREE_CURRENT_NODE_COLOR = (255, 100, 100) # Color for current node
TREE_LINE_COLOR = (200, 200, 200)

# Font
INFO_FONT_SIZE = 16
TREE_FONT_SIZE = 14
try:
    INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE) # Use sans-serif for tree
except Exception as e:
    print(f"Warning: Could not load fonts ({e}). Using default.")
    INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE + 2)
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)

# --- Chess Configuration ---
# STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Adjust as needed
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Example path
ANALYSIS_TIME_LIMIT = 0.4 # Slightly faster for responsiveness
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = 15000

# --- Asset Loading ---
PIECE_IMAGE_PATH = "pieces"
PIECE_IMAGES = {}
# load_piece_images function (same as before)
def load_piece_images(path=PIECE_IMAGE_PATH, sq_size=SQ_SIZE):
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    loaded_images = {}
    if not os.path.isdir(path):
        print(f"Error: Piece image directory not found: '{path}'")
        return None
    all_loaded = True
    for piece in pieces:
        file_path = os.path.join(path, f"{piece}.png")
        try:
            img = pygame.image.load(file_path).convert_alpha()
            img = pygame.transform.smoothscale(img, (sq_size, sq_size))
            loaded_images[piece] = img
        except pygame.error as e:
            print(f"Error loading piece image '{file_path}': {e}")
            all_loaded = False
    if not all_loaded:
         print("Please ensure all 12 piece PNG files exist in the specified directory.")
         return None
    print(f"Loaded {len(loaded_images)} piece images.")
    return loaded_images


# --- Game History Tree Node ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen
        self.move = move # chess.Move object that *led* to this node (None for root)
        self.parent = parent
        self.children = [] # List of child GameNode objects
        self.raw_score = raw_score # Store eval with the state
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT) if raw_score else None # Calculate and store (handle None score)
        # Add SAN cache
        self._san_cache = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_ply(self):
        count = 0
        node = self
        while node.parent:
            count += 1
            node = node.parent
        return count

    def get_san(self, board_at_parent):
        """Gets the SAN notation for the move leading to this node. Requires parent's board state."""
        if self._san_cache is not None:
            return self._san_cache
        if not self.move or not self.parent:
            return "" # Root or no move

        try:
            # Temporarily set board to parent state to get SAN
            # board_at_parent = chess.Board(self.parent.fen, chess960=True) # Use chess960 flag
            san = board_at_parent.san(self.move)
            self._san_cache = san # Cache it
            return san
        except Exception as e:
            # Fallback to UCI if SAN fails
            # print(f"SAN Error for move {self.move.uci()} from FEN {self.parent.fen}: {e}")
            return self.move.uci()


# --- Helper Functions (Chess Logic - score_to_white_percentage, get_engine_analysis - modified slightly) ---
def get_castling_rights_str(board):
    rights = []
    if board.has_kingside_castling_rights(chess.WHITE): rights.append("K")
    if board.has_queenside_castling_rights(chess.WHITE): rights.append("Q")
    if board.has_kingside_castling_rights(chess.BLACK): rights.append("k")
    if board.has_queenside_castling_rights(chess.BLACK): rights.append("q")
    return "".join(rights) if rights else "-"
    
def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    # (Same as before - make sure it handles score=None)
    if score is None: return None # IMPORTANT: Return None if score is None
    pov_score = score.white()
    if pov_score.is_mate():
        mate_val = pov_score.mate()
        return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE)
    if cp is None: return None # Handle case where score exists but cp is None
    clamped_cp = max(-clamp_limit, min(clamp_limit, cp))
    normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
    return normalized * 100.0

def get_engine_analysis(board, engine, time_limit):
    # (Same as before, ensure it handles engine errors gracefully)
    if not engine:
        return None, "Analysis unavailable (Engine not loaded).", None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_move = info.get("pv", [None])[0]
        score = info.get("score")
        if score is None and not info.get("pv"): # Check if analysis completely failed
             return None, "Engine analysis failed (no score/pv).", None
        if best_move and score is not None: # Proceed if we have score and best move
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
        elif score is not None: # Have score but no best move (e.g., mate position?)
             pov_score = score.pov(board.turn)
             score_str = f"Score: {pov_score.score() / 100.0:+.2f}" if not pov_score.is_mate() else f"Mate {pov_score.mate()}"
             return None, f"Position Score: {score_str} (no pv)", score # Return score even without PV
        else:
            return None, "Engine could not provide suggestion.", None
    except chess.engine.EngineTerminatedError:
         print("Engine terminated unexpectedly.")
         return None, "Engine terminated.", None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None, f"Analysis error: {e}", None


# --- Helper Function to get path for plot ---
def get_path_to_node(node):
    """Returns the list of nodes from root to the given node."""
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    return path[::-1] # Reverse to get root -> node order


# --- Drawing Functions (Pygame - Existing ones updated, new draw_game_tree) ---

def draw_board(surface):
    # (Same as before)
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(surface, board, piece_images, dragging_piece_info):
    # (Same as before)
    if piece_images is None: return
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if dragging_piece_info and square == dragging_piece_info['square']:
                continue
            piece_symbol = piece.symbol()
            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
            img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                screen_x = file * SQ_SIZE
                screen_y = (7 - rank) * SQ_SIZE
                surface.blit(img, (screen_x, screen_y))
    if dragging_piece_info and dragging_piece_info['img']:
        img_rect = dragging_piece_info['img'].get_rect(center=dragging_piece_info['pos'])
        surface.blit(dragging_piece_info['img'], img_rect)

def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move):
    # --- Highlight last move ---
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        for sq in [from_sq, to_sq]:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((186, 202, 68, 180)) # Last move highlight color (different from selection)
            surface.blit(s, highlight_rect.topleft)
    # -------------------------

    # Highlight selected square
    if selected_square is not None:
        rank = chess.square_rank(selected_square)
        file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
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
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6)
            surface.blit(s, (file * SQ_SIZE, (7 - rank) * SQ_SIZE))


def draw_eval_bar(surface, white_percentage):
    # (Same as before, ensure white_percentage can be None)
    bar_x = BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = 0
    bar_height = BOARD_SIZE

    if white_percentage is None:
        white_percentage = 50.0 # Default if no eval data

    white_height = int(bar_height * (white_percentage / 100.0))
    black_height = bar_height - white_height

    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

# create_eval_plot_surface (modified to take path and handle None percentages)
def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    if not node_path or len(node_path) < 2: # Need at least root and one move
        return None

    plies = [node.get_ply() for node in node_path]
    # Handle None percentages gracefully - use 50 or previous value? Using 50 is simpler.
    percentages = [node.white_percentage if node.white_percentage is not None else 50.0 for node in node_path]

    # Convert Pygame colors to Matplotlib format (0.0-1.0)
    dark_grey_mpl = tuple(c / 255.0 for c in DARK_GREY)
    orange_mpl = tuple(c / 255.0 for c in ORANGE)
    grey_mpl = tuple(c / 255.0 for c in GREY)
    light_grey_mpl = (211/255.0, 211/255.0, 211/255.0)
    white_mpl = (1.0, 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(plot_width_px / 80, plot_height_px / 80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl)
    ax.set_facecolor(dark_grey_mpl)

    # Plot the data
    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=0.9, step='post') # Use step='post' maybe?
        ax.plot(plies, percentages, color=white_mpl, marker='.', markersize=3, linestyle='-') # Add line too
    elif len(plies) == 1: # Single point (root)
         ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=5)


    ax.axhline(50, color=orange_mpl, linestyle='-', linewidth=1.5)

    ax.set_xlabel("Ply", color=light_grey_mpl)
    ax.set_ylabel("White Win %", color=light_grey_mpl)
    ax.set_title("Win Probability History (Current Line)", color=white_mpl, fontsize=10)
    ax.set_xlim(0, max(plies) if plies else 1)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', colors=light_grey_mpl, labelsize=8)
    ax.tick_params(axis='y', colors=light_grey_mpl, labelsize=8)
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl)
    ax.spines['top'].set_color(grey_mpl)
    ax.spines['bottom'].set_color(grey_mpl)
    ax.spines['left'].set_color(grey_mpl)
    ax.spines['right'].set_color(grey_mpl)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    try:
        plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e:
        print(f"Error loading plot image into Pygame surface: {e}")
        plot_surface = None
    finally:
         buf.close()

    return plot_surface


def draw_game_info(surface, board, analysis_text, message, font, current_node):
    # (Minor changes to get move/ply from node)
    start_y = BOARD_SIZE + 5
    line_height = font.get_height() + 2

    # Get info from the current_node and the board synced to it
    ply = current_node.get_ply()
    move_num = (ply + 1) // 2 + (ply % 2) # Calculate full move number

    lines = []
    turn = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"Turn: {turn}")
    if board.is_check(): lines.append("!! CHECK !!")

    # Get last move SAN from the node itself
    last_move_san = "Start"
    if current_node.move and current_node.parent:
        # Need parent board state to generate SAN correctly
        parent_board = chess.Board(current_node.parent.fen, chess960=True) # chess960 flag important
        last_move_san = current_node.get_san(parent_board)
    lines.append(f"Last Move: {last_move_san}")

    lines.append(f"Ply: {ply} | Move: {move_num}")
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

    info_rect = pygame.Rect(0, BOARD_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT) # Use SCREEN_WIDTH
    surface.fill(DARK_GREY, info_rect)

    for i, line in enumerate(lines):
        if i * line_height > INFO_PANEL_HEIGHT - line_height: break # Avoid drawing outside panel
        text_surface = font.render(line, True, WHITE)
        surface.blit(text_surface, (5, start_y + i * line_height))

# --- NEW: Game Tree Drawing ---
NODE_RADIUS = 6
NODE_DIAMETER = NODE_RADIUS * 2
HORIZ_SPACING = 45 # Increased spacing
VERT_SPACING = 25

# Store node positions for click detection (global or passed around)
drawn_tree_nodes = {} # Maps GameNode object to its screen Rect

# Temporary board for SAN generation within drawing
temp_san_board = chess.Board(chess960=True)

def draw_node_recursive(surface, node, x, y, level_width, font, current_node, parent_board_state=None):
    """Recursively draws a node and its children, returning the total width used by this branch."""
    global drawn_tree_nodes

    # Draw the node
    node_rect = pygame.Rect(x - NODE_RADIUS, y - NODE_RADIUS, NODE_DIAMETER, NODE_DIAMETER)
    color = TREE_CURRENT_NODE_COLOR if node == current_node else TREE_NODE_COLOR
    pygame.draw.circle(surface, color, (x, y), NODE_RADIUS)
    drawn_tree_nodes[node] = node_rect # Store rect for click detection

    # Draw move text (SAN) next to the node (if not root)
    move_text = ""
    current_board_state = chess.Board(node.fen, chess960=True) # Board state for *this* node
    if node.move and parent_board_state:
        move_text = node.get_san(parent_board_state) # Use cached SAN if possible
        text_surf = font.render(move_text, True, WHITE)
        text_rect = text_surf.get_rect(midleft=(x + NODE_RADIUS + 3, y))
        surface.blit(text_surf, text_rect)

    # Draw children and lines
    child_start_x = x - level_width / 2 # Distribute children somewhat evenly
    total_child_width = 0
    child_x_offset = 0

    # Pre-calculate widths needed for children to center properly
    child_widths = []
    for child in node.children:
        # Estimate width needed for child subtree (this is tricky without full layout pass)
        # For now, assume a base width per child branch
        child_widths.append(HORIZ_SPACING) # Simplified width estimate

    total_estimated_child_width = sum(child_widths)
    current_x_pos = x - total_estimated_child_width / 2 + child_widths[0]/2 if child_widths else x


    for i, child in enumerate(node.children):
        child_y = y + VERT_SPACING
        # Simple horizontal placement for now
        # child_x = child_start_x + i * HORIZ_SPACING + HORIZ_SPACING / 2
        child_x = current_x_pos

        # Draw connecting line
        pygame.draw.line(surface, TREE_LINE_COLOR, (x, y + NODE_RADIUS), (child_x, child_y - NODE_RADIUS), 1)

        # Recurse
        # Pass the board state of the *current* node as the parent state for the child
        branch_width = draw_node_recursive(surface, child, child_x, child_y, child_widths[i], font, current_node, current_board_state)
        total_child_width += branch_width + 5 # Add some padding between branches
        current_x_pos += child_widths[i] + 5 # Move to next position


    # Return width used by this node/branch
    return max(NODE_DIAMETER + (len(move_text) * font.size(move_text)[0] / len(move_text) if move_text else 0), total_child_width)


def draw_game_tree(surface, root_node, current_node, font):
    """Draws the game history tree visualization."""
    global drawn_tree_nodes
    drawn_tree_nodes.clear() # Clear previous positions

    tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
    surface.fill(DARK_GREY, tree_panel_rect) # Background

    if not root_node: return

    # Start drawing from top-center of the panel
    start_x = tree_panel_rect.width // 2
    start_y = tree_panel_rect.top + VERT_SPACING // 2

    # Initial call to recursive drawing function
    # Need the initial board state (root FEN) for the first level SAN calculation
    root_board_state = chess.Board(root_node.fen, chess960=True)
    draw_node_recursive(surface, root_node, start_x, start_y, tree_panel_rect.width, font, current_node, None)


# --- Coordinate Conversion ---
def screen_to_square(pos):
    # (Same as before)
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    file = x // SQ_SIZE
    rank = 7 - (y // SQ_SIZE)
    return chess.square(file, rank)

# --- Main Game Function ---
def play_chess960_pygame():
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board # Allow modification
    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None:
        print("Exiting due to missing piece images.")
        return

    board = chess.Board(chess960=True) # Main board object synced with current_node
    engine = None
    message = None
    start_pos_num = -1
    game_root = None # Root of the game tree
    current_node = None # Currently viewed node in the tree
    last_raw_score = None # Store raw score for current node
    one_off_analysis_text = None
    meter_visible = True
    plot_surface = None
    needs_redraw = True # Flag to trigger full redraw

    # --- Get Starting Position ---
    # (Same console input as before)
    while True:
        print("\n--- Chess960 Setup ---")
        pos_choice = input("Enter Chess960 position number (0-959), 'random', or leave blank for random: ").strip().lower()
        if not pos_choice or pos_choice == 'random':
            start_pos_num = random.randint(0, 959)
            break
        else:
            try:
                start_pos_num = int(pos_choice)
                if 0 <= start_pos_num <= 959: break
                else: print("Position number must be between 0 and 959.")
            except ValueError: print("Invalid input.")

    board.set_chess960_pos(start_pos_num)
    start_fen = board.fen()
    print(f"Starting Chess960 Position {start_pos_num} (FEN: {start_fen})")
    print("Initializing Pygame window...")
    time.sleep(0.5)

    # --- Initialize Engine and Game Tree Root ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try: engine.configure({"UCI_Chess960": True})
            except Exception: pass # Ignore if not supported
            print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
            # Get initial analysis for the root node
            _, _, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5) # Slightly longer for start
            game_root = GameNode(fen=start_fen, raw_score=initial_score)
            current_node = game_root
            last_raw_score = initial_score # Update score for display
            print(f"Initial analysis done (Score: {initial_score.white().score() if initial_score and not initial_score.white().is_mate() else 'Mate'})")

        else:
            print(f"Warning: Stockfish not found ('{STOCKFISH_PATH}'). Analysis disabled.")
            message = "Engine unavailable"
            game_root = GameNode(fen=start_fen) # Create root even without engine
            current_node = game_root
    except Exception as e:
        print(f"Error initializing Stockfish: {e}. Analysis disabled.")
        engine = None
        message = "Engine init failed"
        game_root = GameNode(fen=start_fen) # Create root even without engine/analysis
        current_node = game_root

    # --- Pygame Loop Variables ---
    running = True
    clock = pygame.time.Clock()
    selected_square = None
    dragging_piece_info = None
    legal_moves_for_selected = []
    last_move_displayed = None # Keep track of the move leading to current_node

    # --- Main Game Loop ---
    while running:
        # --- Event Handling ---
        current_mouse_pos = pygame.mouse.get_pos() # Get mouse pos once per frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos # Use event position for clicks
                # Check Tree Panel Click First
                tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
                if tree_panel_rect.collidepoint(pos):
                    clicked_node = None
                    for node, rect in drawn_tree_nodes.items():
                        if rect.collidepoint(pos):
                            clicked_node = node
                            break
                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node
                        board.set_fen(current_node.fen) # Sync board
                        last_raw_score = current_node.raw_score # Update score display
                        message = f"Jumped to ply {current_node.get_ply()}"
                        one_off_analysis_text = None # Clear previous analysis
                        selected_square = None # Clear board selection
                        dragging_piece_info = None
                        legal_moves_for_selected = []
                        needs_redraw = True
                # Check Board Click
                elif pos[0] < BOARD_SIZE and pos[1] < BOARD_SIZE: # Click is on the board
                    if event.button == 1: # Left click
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over(): # Ensure board is synced before checking game over
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn:
                                selected_square = sq
                                piece_symbol = piece.symbol()
                                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                                dragging_piece_info = {
                                    'square': sq, 'piece': piece,
                                    'img': PIECE_IMAGES.get(piece_key), 'pos': pos
                                }
                                legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                needs_redraw = True
                            else:
                                selected_square = None
                                dragging_piece_info = None
                                legal_moves_for_selected = []
                                needs_redraw = True # Redraw to clear highlights

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 and dragging_piece_info: # Left release while dragging
                    pos = event.pos
                    to_sq = screen_to_square(pos)
                    from_sq = dragging_piece_info['square']

                    if to_sq is not None and from_sq != to_sq:
                        promotion = None
                        piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN:
                            target_rank = 7 if piece.color == chess.WHITE else 0
                            if chess.square_rank(to_sq) == target_rank:
                                promotion = chess.QUEEN

                        move = chess.Move(from_sq, to_sq, promotion=promotion)

                        if move in board.legal_moves:
                            # --- Branching Logic ---
                            # Check if this move already exists as a child of current_node
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move:
                                    existing_child = child
                                    break

                            if existing_child:
                                # Move already exists, just navigate to it
                                current_node = existing_child
                                board.set_fen(current_node.fen)
                                last_raw_score = current_node.raw_score
                                message = "Navigated to existing move."
                            else:
                                # New move - create new node and add as child
                                board.push(move) # Make move on temp board to get FEN
                                new_fen = board.fen()
                                board.pop() # Revert board back to current_node's state

                                # Analyze the new position *before* creating the node
                                temp_board_for_analysis = chess.Board(new_fen, chess960=True)
                                new_raw_score = None
                                analysis_failed = False
                                if engine:
                                    _, _, new_raw_score = get_engine_analysis(temp_board_for_analysis, engine, ANALYSIS_TIME_LIMIT)
                                    if new_raw_score is None:
                                         analysis_failed = True
                                         message = "Analysis failed for new move."

                                # Create the new node
                                new_node = GameNode(fen=new_fen, move=move, parent=current_node, raw_score=new_raw_score)
                                current_node.add_child(new_node)
                                current_node = new_node # Advance to the new node
                                board.set_fen(current_node.fen) # Sync main board
                                last_raw_score = current_node.raw_score # Update score display

                                if analysis_failed: message = "Move made (analysis failed)."
                                else: message = None # Clear message if successful


                            # --- End Branching Logic ---
                            one_off_analysis_text = None # Clear previous analysis
                            needs_redraw = True

                        else:
                            message = f"Illegal move: {chess.square_name(from_sq)}{chess.square_name(to_sq)}"
                            needs_redraw = True # Redraw to snap piece back

                    # Reset dragging state
                    dragging_piece_info = None
                    selected_square = None
                    legal_moves_for_selected = []
                    needs_redraw = True # Redraw after dropping


            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info:
                    dragging_piece_info['pos'] = event.pos
                    needs_redraw = True # Need redraw while dragging

            elif event.type == pygame.KEYDOWN:
                 # --- Keyboard Commands ---
                if event.key == pygame.K_LEFT: # Go back (parent)
                    if current_node.parent:
                        current_node = current_node.parent
                        board.set_fen(current_node.fen)
                        last_raw_score = current_node.raw_score
                        message = f"Back to ply {current_node.get_ply()}"
                        one_off_analysis_text = None
                        selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True
                    else: message = "At start of game."

                elif event.key == pygame.K_RIGHT: # Go forward (first child)
                    if current_node.children:
                        # Move to the first child (main line assumption for now)
                        current_node = current_node.children[0]
                        board.set_fen(current_node.fen)
                        last_raw_score = current_node.raw_score
                        message = f"Forward to ply {current_node.get_ply()}"
                        one_off_analysis_text = None
                        selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True
                    else: message = "At end of current line."

                elif event.key == pygame.K_a: # Analyze current position
                     if engine:
                         message = "Analyzing..."
                         needs_redraw = True # Show message immediately

                         # Force redraw before blocking for analysis
                         # (Copy drawing code block here - slightly redundant but needed for responsiveness)
                         screen.fill(DARK_GREY)
                         draw_board(screen)
                         last_move_displayed = current_node.move if current_node.parent else None
                         draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed)
                         draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
                         draw_eval_bar(screen, current_node.white_percentage) # Use node's percentage
                         path = get_path_to_node(current_node)
                         if meter_visible:
                            plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                            if plot_surface:
                                plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                                screen.blit(plot_surface, plot_rect)
                         draw_game_tree(screen, game_root, current_node, TREE_FONT)
                         draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node)
                         pygame.display.flip()
                         # ------------------------------------------

                         # Run analysis
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 3)

                         # Update the *current node's* score data
                         current_node.raw_score = raw_score
                         new_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                         # Only update plot if percentage actually changes (or becomes available)
                         if new_percentage is not None and new_percentage != current_node.white_percentage:
                              current_node.white_percentage = new_percentage
                              # Plot will be updated on next full redraw cycle
                         last_raw_score = raw_score # Update live bar score immediately

                         if best_move_san:
                             one_off_analysis_text = f"Suggests: {best_move_san} ({score_str})"
                         else:
                             one_off_analysis_text = score_str if score_str else "Analysis complete (no suggestion)."
                         message = None # Clear "Analyzing..."
                         needs_redraw = True
                     else:
                         message = "Engine not available."

                elif event.key == pygame.K_m: # Toggle Meter/Plot
                    meter_visible = not meter_visible
                    message = f"Eval Meter/Plot {'ON' if meter_visible else 'OFF'}"
                    needs_redraw = True # Need redraw to show/hide plot

                elif event.key == pygame.K_ESCAPE:
                    running = False

        # --- Update based on state ---
        last_move_displayed = current_node.move if current_node.parent else None
        board.set_fen(current_node.fen) # Ensure board is always synced before drawing

        # --- Drawing (only if needed) ---
        if needs_redraw:
            screen.fill(DARK_GREY)

            # Board Area
            draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)

            # Side Panel (Eval Bar)
            # Use score from current_node for consistency with history
            draw_eval_bar(screen, current_node.white_percentage)

            # Info Panel
            draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node)

            # Plot Panel
            if meter_visible:
                path = get_path_to_node(current_node) # Get path for current line
                plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT) # Generate plot for this path
                if plot_surface:
                    plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                    screen.blit(plot_surface, plot_rect)

            # Tree Panel
            draw_game_tree(screen, game_root, current_node, TREE_FONT)


            pygame.display.flip()
            needs_redraw = False # Reset redraw flag

        # Limit frame rate
        clock.tick(30)

    # --- Cleanup ---
    # (Same as before)
    print("\nExiting Pygame...")
    if engine:
        try: engine.quit(); print("Stockfish engine closed.")
        except Exception: print("Error closing engine or already closed.")
    plt.close('all')
    pygame.quit()
    sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()