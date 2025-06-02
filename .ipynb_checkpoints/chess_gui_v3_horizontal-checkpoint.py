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
INFO_PANEL_HEIGHT = 80
PLOT_PANEL_HEIGHT = 120
TREE_PANEL_HEIGHT = 180 # Slightly increased tree panel height
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20

SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board with Horizontal History Tree")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60)
LIGHT_SQ_COLOR = (238, 238, 210)
DARK_SQ_COLOR = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
ORANGE = (255, 165, 0)
# Tree Colors
TREE_BG_COLOR = DARK_GREY
TREE_LINE_COLOR = (150, 150, 150)
TREE_NODE_WHITE_MOVE = (230, 230, 230) # Node for a state reached by a White move
TREE_NODE_BLACK_MOVE = (50, 50, 50)   # Node for a state reached by a Black move
TREE_NODE_ROOT = (100, 100, 150) # Special color for root
TREE_NODE_CURRENT_OUTLINE = (255, 100, 100) # Outline for current node
TREE_TEXT_COLOR = (200, 200, 200)


# Font
INFO_FONT_SIZE = 16
TREE_FONT_SIZE = 12 # Slightly smaller for potentially more text
try:
    INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
except Exception as e:
    print(f"Warning: Could not load fonts ({e}). Using default.")
    INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE + 2)
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)

# --- Chess Configuration ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Adjust as needed
ANALYSIS_TIME_LIMIT = 0.4
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
        self.raw_score = raw_score
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT) if raw_score else None
        self._san_cache = None
        # Layout attributes (will be calculated during drawing)
        self.x = 0
        self.y = 0
        self.screen_rect = None # Store the drawn rectangle here

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
        if self._san_cache is not None:
            return self._san_cache
        if not self.move or not self.parent:
            return "root" # Root node representation

        try:
            san = board_at_parent.san(self.move)
            self._san_cache = san
            return san
        except Exception as e:
            return self.move.uci()

    def get_player_color_who_moved(self):
        """Determines the color of the player who made the move TO this node."""
        if not self.parent:
            return None # Root node has no preceding move

        # The FEN stores the player whose turn it is NOW.
        # So, the player who moved TO this state is the OPPOSITE color.
        board_state = chess.Board(self.fen, chess960=True)
        return not board_state.turn # Return chess.WHITE if black moved, chess.BLACK if white moved


# --- Helper Functions (Chess Logic) ---
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
        if score is None and not info.get("pv"):
             return None, "Engine analysis failed (no score/pv).", None
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
        elif score is not None:
             pov_score = score.pov(board.turn)
             score_str = f"Score: {pov_score.score() / 100.0:+.2f}" if not pov_score.is_mate() else f"Mate {pov_score.mate()}"
             return None, f"Position Score: {score_str} (no pv)", score
        else:
            return None, "Engine could not provide suggestion.", None
    except chess.engine.EngineTerminatedError:
         print("Engine terminated unexpectedly.")
         return None, "Engine terminated.", None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None, f"Analysis error: {e}", None

def get_path_to_node(node):
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    return path[::-1]

# --- Drawing Functions ---

def draw_board(surface):
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(surface, board, piece_images, dragging_piece_info):
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
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        for sq in [from_sq, to_sq]:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((186, 202, 68, 180))
            surface.blit(s, highlight_rect.topleft)

    if selected_square is not None:
        rank = chess.square_rank(selected_square)
        file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT_COLOR)
        surface.blit(s, highlight_rect.topleft)

    if legal_moves_for_selected:
        for move in legal_moves_for_selected:
            dest_sq = move.to_square
            rank = chess.square_rank(dest_sq)
            file = chess.square_file(dest_sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6)
            surface.blit(s, (file * SQ_SIZE, (7 - rank) * SQ_SIZE))

def draw_eval_bar(surface, white_percentage):
    bar_x = BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = 0
    bar_height = BOARD_SIZE

    if white_percentage is None:
        white_percentage = 50.0

    white_height = int(bar_height * (white_percentage / 100.0))
    black_height = bar_height - white_height

    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    if not node_path or len(node_path) < 2:
        return None

    plies = [node.get_ply() for node in node_path]
    percentages = [node.white_percentage if node.white_percentage is not None else 50.0 for node in node_path]

    dark_grey_mpl = tuple(c / 255.0 for c in DARK_GREY)
    orange_mpl = tuple(c / 255.0 for c in ORANGE)
    grey_mpl = tuple(c / 255.0 for c in GREY)
    light_grey_mpl = (211/255.0, 211/255.0, 211/255.0)
    white_mpl = (1.0, 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(plot_width_px / 80, plot_height_px / 80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl)
    ax.set_facecolor(dark_grey_mpl)

    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=0.9, step='post')
        ax.plot(plies, percentages, color=white_mpl, marker='.', markersize=3, linestyle='-')
    elif len(plies) == 1:
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
    start_y = BOARD_SIZE + 5
    line_height = font.get_height() + 2
    ply = current_node.get_ply()
    move_num = (ply + 1) // 2 + (ply % 2)

    lines = []
    turn = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"Turn: {turn}")
    if board.is_check(): lines.append("!! CHECK !!")

    last_move_san = "Start"
    if current_node.move and current_node.parent:
        parent_board = chess.Board(current_node.parent.fen, chess960=True)
        last_move_san = current_node.get_san(parent_board)
    lines.append(f"Last Move: {last_move_san}")

    lines.append(f"Ply: {ply} | Move: {move_num}")
    lines.append(f"Castling: {get_castling_rights_str(board)}")

    if analysis_text:
        lines.append("-" * 10); lines.append(analysis_text)
    if message:
        lines.append("-" * 10); lines.append(f"Msg: {message}")
    if board.is_game_over(claim_draw=True):
         lines.append("-" * 10)
         res = board.outcome(claim_draw=True)
         term = res.termination.name.replace('_',' ').title() if res else "N/A"
         lines.append(f"GAME OVER: {res.result() if res else board.result(claim_draw=True)} ({term})")

    info_rect = pygame.Rect(0, BOARD_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    surface.fill(DARK_GREY, info_rect)

    for i, line in enumerate(lines):
        if i * line_height > INFO_PANEL_HEIGHT - line_height: break
        text_surface = font.render(line, True, WHITE)
        surface.blit(text_surface, (5, start_y + i * line_height))


# --- NEW: Horizontal Game Tree Drawing ---
NODE_RADIUS = 5
NODE_DIAMETER = NODE_RADIUS * 2
HORIZ_SPACING = 55 # Horizontal distance between plies
VERT_SPACING = 25 # Minimum vertical distance between siblings
TEXT_OFFSET_X = 8 # Distance from node center to start of text
TEXT_OFFSET_Y = -TREE_FONT_SIZE // 2 # Vertical offset for text centering

# Store node objects mapped to their calculated screen Rects for click detection
drawn_tree_nodes = {} # Cleared each frame draw_game_tree is called

# Temporary board for SAN generation within drawing
temp_san_board = chess.Board(chess960=True)

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node, panel_rect):
    """
    Calculates layout positions and draws the tree recursively.
    Uses a simple vertical distribution for siblings.
    Returns the actual y position used and the total vertical height of the drawn subtree.
    """
    global drawn_tree_nodes

    node.x = x # Store calculated position

    # --- 1. Calculate positions of children first (to determine vertical spread) ---
    child_y_positions = []
    child_heights = []
    total_child_height_estimate = 0
    child_x = x + HORIZ_SPACING

    if node.children:
        # Estimate height needed for children based on VERT_SPACING
        # This is a simplification and doesn't account for nested subtree heights perfectly
        total_child_height_estimate = (len(node.children) - 1) * VERT_SPACING

        # Calculate starting y for the first child, aiming to center the block of children around the parent's y_center
        current_child_y = y_center - total_child_height_estimate / 2

        for child in node.children:
            # Recursively layout children - this only gets an *estimated* position for now
            # A true layout algorithm (like Reingold-Tilford) is needed for perfect non-overlapping placement,
            # but we'll use this simpler approach.
            child_actual_y, child_subtree_height = layout_and_draw_tree_recursive(
                surface, child, child_x, current_child_y, level + 1, font, current_node, panel_rect
            )
            child_y_positions.append(child_actual_y)
            child_heights.append(child_subtree_height)

            # Update y for the next child, ensuring minimum spacing
            # Add half the current child's height and half the *estimated* next child's height (using VERT_SPACING as proxy)
            current_child_y += max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2) # Use max to ensure min spacing


    # --- 2. Determine this node's final y position ---
    # For simplicity, we keep the passed y_center. In a better layout, this might be adjusted based on children.
    node.y = y_center

    # --- 3. Draw this node (if within panel bounds) ---
    if panel_rect.collidepoint(node.x, node.y):
        # Determine node color based on who moved *to* this state
        player_color = node.get_player_color_who_moved()
        if player_color == chess.WHITE:
            node_color = TREE_NODE_WHITE_MOVE
        elif player_color == chess.BLACK:
            node_color = TREE_NODE_BLACK_MOVE
        else: # Root node
            node_color = TREE_NODE_ROOT

        # Draw the node circle
        pygame.draw.circle(surface, node_color, (int(node.x), int(node.y)), NODE_RADIUS)

        # Outline the current node
        if node == current_node:
            pygame.draw.circle(surface, TREE_NODE_CURRENT_OUTLINE, (int(node.x), int(node.y)), NODE_RADIUS + 1, 1) # Outline

        # Draw the move text (SAN)
        move_text = ""
        if node.parent:
            # Need parent's board state for SAN
            parent_board = chess.Board(node.parent.fen, chess960=True)
            move_text = node.get_san(parent_board)
        else:
            move_text = "Start" # Indicate root

        if move_text:
            text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
            text_rect = text_surf.get_rect(midleft=(node.x + TEXT_OFFSET_X, node.y + TEXT_OFFSET_Y + NODE_RADIUS//2)) # Adjust Y slightly
            # Clip text drawing to panel
            clipped_text_rect = text_rect.clip(panel_rect)
            if clipped_text_rect.width > 0 and clipped_text_rect.height > 0:
                 # Blit requires the source surface, dest top-left, and optionally an area from the source
                 # We need to calculate the correct area from the *original* text_surf
                 area_to_blit = pygame.Rect(
                     clipped_text_rect.left - text_rect.left,
                     clipped_text_rect.top - text_rect.top,
                     clipped_text_rect.width,
                     clipped_text_rect.height
                 )
                 surface.blit(text_surf, clipped_text_rect.topleft, area=area_to_blit)


        # Store the clickable area
        node.screen_rect = pygame.Rect(node.x - NODE_RADIUS, node.y - NODE_RADIUS, NODE_DIAMETER, NODE_DIAMETER)
        drawn_tree_nodes[node] = node.screen_rect

    # --- 4. Draw lines to children ---
    for i, child in enumerate(node.children):
        # Ensure both parent and child have valid coordinates (might not if off-panel)
        if hasattr(child, 'x') and hasattr(child, 'y') and panel_rect.collidepoint(node.x, node.y) and panel_rect.collidepoint(child.x, child.y):
           # Draw line from right edge of parent to left edge of child
           start_pos = (int(node.x + NODE_RADIUS), int(node.y))
           end_pos = (int(child.x - NODE_RADIUS), int(child.y))
           pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)


    # --- 5. Return actual Y position and estimated height ---
    # Calculate the height based on the spread of children
    my_height = NODE_DIAMETER # Minimum height is the node itself
    if child_y_positions:
        min_child_y = min(child_y_positions) - max(child_heights)/2 # Approximate top
        max_child_y = max(child_y_positions) + max(child_heights)/2 # Approximate bottom
        my_height = max(my_height, max_child_y - min_child_y)

    return node.y, my_height


def draw_game_tree(surface, root_node, current_node, font):
    """Draws the game history tree visualization horizontally."""
    global drawn_tree_nodes
    drawn_tree_nodes.clear() # Clear previous positions

    tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
    surface.fill(TREE_BG_COLOR, tree_panel_rect) # Background

    if not root_node: return

    # Start drawing from near the left-center of the panel
    # We might want to add panning later if the tree gets too wide
    start_x = 15 # Small left padding
    start_y = tree_panel_rect.centery # Start vertically centered

    # Create a subsurface for clipping (optional but good practice)
    tree_surface = surface.subsurface(tree_panel_rect)

    # Initial call to recursive layout and drawing function
    # Pass the panel_rect for clipping checks
    layout_and_draw_tree_recursive(tree_surface, root_node, start_x, start_y - tree_panel_rect.top, 0, font, current_node, tree_surface.get_rect())


# --- Coordinate Conversion ---
def screen_to_square(pos):
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    file = x // SQ_SIZE
    rank = 7 - (y // SQ_SIZE)
    return chess.square(file, rank)

# --- Main Game Function ---
def play_chess960_pygame():
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board
    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None:
        print("Exiting due to missing piece images.")
        return

    board = chess.Board(chess960=True)
    engine = None
    message = None
    start_pos_num = -1
    game_root = None
    current_node = None
    last_raw_score = None
    one_off_analysis_text = None
    meter_visible = True
    plot_surface = None
    needs_redraw = True

    # --- Get Starting Position ---
    while True:
        print("\n--- Chess960 Setup ---")
        pos_choice = input("Enter Chess960 position number (0-959), 'random', or leave blank for random: ").strip().lower()
        if not pos_choice or pos_choice == 'random':
            start_pos_num = random.randint(0, 959); break
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
            except Exception: pass
            print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
            _, _, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5)
            game_root = GameNode(fen=start_fen, raw_score=initial_score)
            current_node = game_root
            last_raw_score = initial_score
            print(f"Initial analysis done (Score: {initial_score.white().score() if initial_score and not initial_score.white().is_mate() else 'Mate'})")
        else:
            print(f"Warning: Stockfish not found ('{STOCKFISH_PATH}'). Analysis disabled.")
            message = "Engine unavailable"
            game_root = GameNode(fen=start_fen)
            current_node = game_root
    except Exception as e:
        print(f"Error initializing Stockfish: {e}. Analysis disabled.")
        engine = None; message = "Engine init failed"
        game_root = GameNode(fen=start_fen)
        current_node = game_root

    # --- Pygame Loop Variables ---
    running = True
    clock = pygame.time.Clock()
    selected_square = None
    dragging_piece_info = None
    legal_moves_for_selected = []
    last_move_displayed = None

    # --- Main Game Loop ---
    while running:
        current_mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

                # --- Check Tree Panel Click ---
                if tree_panel_rect.collidepoint(pos):
                     # Adjust click position relative to the tree panel's top-left
                    relative_click_pos = (pos[0] - tree_panel_rect.left, pos[1] - tree_panel_rect.top)
                    clicked_node = None
                    # Iterate through the stored node rectangles
                    for node, rect in drawn_tree_nodes.items():
                        # Check collision using the relative click position against the node's rect (which should also be relative to the tree panel)
                        if rect and rect.collidepoint(relative_click_pos): # Ensure rect exists
                            clicked_node = node
                            break # Found the node

                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node
                        board.set_fen(current_node.fen)
                        last_raw_score = current_node.raw_score
                        message = f"Jumped to ply {current_node.get_ply()}"
                        one_off_analysis_text = None
                        selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True

                # --- Check Board Click ---
                elif pos[0] < BOARD_SIZE and pos[1] < BOARD_SIZE: # Click is on the board
                    if event.button == 1: # Left click
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
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
                                needs_redraw = True

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
                                # --- Simple Promotion (Always Queen for now) ---
                                promotion = chess.QUEEN
                                # TODO: Implement promotion choice UI if desired

                        move = chess.Move(from_sq, to_sq, promotion=promotion)

                        if move in board.legal_moves:
                            # Branching Logic
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move:
                                    existing_child = child; break

                            if existing_child:
                                current_node = existing_child
                                board.set_fen(current_node.fen)
                                last_raw_score = current_node.raw_score
                                message = "Navigated to existing move."
                            else:
                                board.push(move); new_fen = board.fen(); board.pop() # Get FEN without changing main board yet

                                temp_board_for_analysis = chess.Board(new_fen, chess960=True)
                                new_raw_score = None; analysis_failed = False
                                if engine:
                                    _, _, new_raw_score = get_engine_analysis(temp_board_for_analysis, engine, ANALYSIS_TIME_LIMIT)
                                    if new_raw_score is None:
                                         analysis_failed = True; message = "Analysis failed for new move."

                                new_node = GameNode(fen=new_fen, move=move, parent=current_node, raw_score=new_raw_score)
                                current_node.add_child(new_node)
                                current_node = new_node # Advance
                                board.set_fen(current_node.fen) # Sync main board
                                last_raw_score = current_node.raw_score

                                if not analysis_failed: message = None # Clear message if successful

                            one_off_analysis_text = None
                            needs_redraw = True

                        else: # Illegal move attempt
                            message = f"Illegal move: {chess.square_name(from_sq)}{chess.square_name(to_sq)}"
                            needs_redraw = True

                    # Reset dragging state regardless of outcome
                    dragging_piece_info = None
                    selected_square = None
                    legal_moves_for_selected = []
                    needs_redraw = True


            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info:
                    dragging_piece_info['pos'] = event.pos
                    needs_redraw = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: # Go back (parent)
                    if current_node.parent:
                        current_node = current_node.parent
                        board.set_fen(current_node.fen)
                        last_raw_score = current_node.raw_score
                        message = f"Back to ply {current_node.get_ply()}"
                        one_off_analysis_text = None; selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True
                    else: message = "At start of game."

                elif event.key == pygame.K_RIGHT: # Go forward (first child)
                    if current_node.children:
                        current_node = current_node.children[0] # Simple: always follow first child
                        board.set_fen(current_node.fen)
                        last_raw_score = current_node.raw_score
                        message = f"Forward to ply {current_node.get_ply()}"
                        one_off_analysis_text = None; selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True
                    else: message = "At end of current line."

                elif event.key == pygame.K_a: # Analyze current position
                     if engine:
                         message = "Analyzing..."; needs_redraw = True
                         # Force immediate redraw to show "Analyzing..."
                         # (Simplified redraw - assumes state before analysis is what's needed)
                         screen.fill(DARK_GREY)
                         draw_board(screen); draw_highlights(screen, board, selected_square, legal_moves_for_selected, current_node.move if current_node.parent else None)
                         draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info); draw_eval_bar(screen, current_node.white_percentage)
                         if meter_visible:
                             path = get_path_to_node(current_node)
                             plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                             if plot_surface: screen.blit(plot_surface, (0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                         draw_game_tree(screen, game_root, current_node, TREE_FONT) # Draw tree before info
                         draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node)
                         pygame.display.flip()

                         # Run analysis
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 3)

                         # Update current node's score data
                         current_node.raw_score = raw_score
                         new_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                         if new_percentage is not None and new_percentage != current_node.white_percentage:
                              current_node.white_percentage = new_percentage
                         last_raw_score = raw_score # Update live bar score

                         if best_move_san: one_off_analysis_text = f"Suggests: {best_move_san} ({score_str})"
                         else: one_off_analysis_text = score_str if score_str else "Analysis complete."
                         message = None # Clear "Analyzing..."
                         needs_redraw = True
                     else: message = "Engine not available."

                elif event.key == pygame.K_m: # Toggle Meter/Plot
                    meter_visible = not meter_visible; message = f"Eval Meter/Plot {'ON' if meter_visible else 'OFF'}"; needs_redraw = True

                elif event.key == pygame.K_ESCAPE: running = False

        # --- Update based on state ---
        last_move_displayed = current_node.move if current_node.parent else None
        board.set_fen(current_node.fen) # Ensure board is synced before drawing

        # --- Drawing (only if needed) ---
        if needs_redraw:
            screen.fill(DARK_GREY)

            # Board Area
            draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)

            # Side Panel (Eval Bar)
            draw_eval_bar(screen, current_node.white_percentage) # Use node's stored percentage

            # Info Panel
            draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node)

            # Plot Panel
            if meter_visible:
                path = get_path_to_node(current_node)
                plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                if plot_surface:
                    plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                    screen.blit(plot_surface, plot_rect)

            # Tree Panel - Drawn last to overlay other panels if needed (though it shouldn't)
            draw_game_tree(screen, game_root, current_node, TREE_FONT)

            pygame.display.flip()
            needs_redraw = False # Reset redraw flag

        # Limit frame rate
        clock.tick(30)

    # --- Cleanup ---
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