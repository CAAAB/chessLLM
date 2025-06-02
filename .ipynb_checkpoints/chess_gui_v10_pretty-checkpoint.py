import chess
import chess.svg
import chess.engine
import random
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import time
import math

import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Pygame & Display Configuration ---
pygame.init()
pygame.font.init()

# --- NEW: Setup Panel ---
SETUP_PANEL_HEIGHT = 40
COORD_PADDING = 20 # Space around the board for coordinates

BOARD_SIZE = 512 # The actual board drawing area size
SQ_SIZE = BOARD_SIZE // 8

# --- Adjusted Panel Heights & Sizes ---
# INFO_PANEL_HEIGHT = 100 # REMOVED
PLOT_PANEL_HEIGHT = 100 # Adjusted height maybe
TREE_PANEL_HEIGHT = 150 # Adjusted height maybe
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20 # Eval bar area

# --- Screen Dimensions ---
SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH + 2 * COORD_PADDING
SCREEN_HEIGHT = SETUP_PANEL_HEIGHT + BOARD_SIZE + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT + 2 * COORD_PADDING
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board")

# --- Board Position on Screen ---
BOARD_X = COORD_PADDING
BOARD_Y = SETUP_PANEL_HEIGHT + COORD_PADDING

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60)
LIGHT_SQ_COLOR = (238, 238, 210)
DARK_SQ_COLOR = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
LAST_MOVE_HIGHLIGHT_COLOR = (186, 202, 68, 180)
ENGINE_MOVE_HIGHLIGHT_COLOR = (0, 150, 255, 150)
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
ORANGE = (255, 165, 0)
TREE_BG_COLOR = DARK_GREY
TREE_LINE_COLOR = (150, 150, 150)
TREE_NODE_ROOT_COLOR = (100, 100, 150)
TREE_NODE_CURRENT_OUTLINE = (255, 100, 100)
TREE_TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (80, 80, 100)
BUTTON_TEXT_COLOR = WHITE
INPUT_BG_COLOR = (200, 200, 200)
INPUT_TEXT_COLOR = BLACK
INPUT_BORDER_INACTIVE_COLOR = GREY
INPUT_BORDER_ACTIVE_COLOR = BLACK
COORD_COLOR = (200, 200, 200) # Color for a-h, 1-8

# --- Badge Colors & Properties ---
TREE_BADGE_RADIUS = 4 # Tree badges remain small
TREE_BADGE_BEST_COLOR = (0, 180, 0)
TREE_BADGE_EXCELLENT_COLOR = (50, 205, 50)
TREE_BADGE_GOOD_COLOR = (0, 100, 0)
TREE_BADGE_INACCURACY_COLOR = (240, 230, 140)
TREE_BADGE_MISTAKE_COLOR = (255, 140, 0)
TREE_BADGE_BLUNDER_COLOR = (200, 0, 0)
TREE_MOVE_QUALITY_COLORS = { "Best": TREE_BADGE_BEST_COLOR, "Excellent": TREE_BADGE_EXCELLENT_COLOR, "Good": TREE_BADGE_GOOD_COLOR, "Inaccuracy": TREE_BADGE_INACCURACY_COLOR, "Mistake": TREE_BADGE_MISTAKE_COLOR, "Blunder": TREE_BADGE_BLUNDER_COLOR }

# --- Board Badges (Larger) ---
BOARD_BADGE_RADIUS = 14 # <<< INCREASED Radius of the background circle
BOARD_BADGE_OUTLINE_COLOR = DARK_GREY
BOARD_BADGE_IMAGE_SIZE = (22, 22) # <<< INCREASED Target size for the icon
# Adjust offset based on new radius
BOARD_BADGE_OFFSET_X = SQ_SIZE - BOARD_BADGE_RADIUS - 2 # Offset for the *center* from top-right
BOARD_BADGE_OFFSET_Y = BOARD_BADGE_RADIUS + 2

# Helper function to load, invert, and resize badge images (Unchanged)
def load_and_process_badge_image(path, target_size, target_color=WHITE):
    try:
        img = pygame.image.load(path).convert_alpha()
        for x in range(img.get_width()):
            for y in range(img.get_height()):
                color = img.get_at((x, y))
                if color[3] > 0 and color[0] < 100 and color[1] < 100 and color[2] < 100:
                    img.set_at((x, y), (*target_color, color[3]))
        img = pygame.transform.smoothscale(img, target_size)
        return img
    except pygame.error as e: print(f"Error processing badge image '{path}': {e}"); return None
    except Exception as e: print(f"Unexpected error processing badge image '{path}': {e}"); return None

# Load badge images (Unchanged)
BADGE_TYPES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
BADGE_IMAGES = {}
all_badges_loaded = True
print("Loading and processing badge images...")
for quality in BADGE_TYPES:
    file_path = os.path.join("badges", f"{quality.lower()}.png")
    processed_img = load_and_process_badge_image(file_path, BOARD_BADGE_IMAGE_SIZE, WHITE)
    if processed_img: BADGE_IMAGES[quality] = processed_img; print(f"  - Loaded: {quality}")
    else: print(f"  - FAILED: {quality}"); all_badges_loaded = False
if not all_badges_loaded: print("Warning: Some badge images failed.")

# --- Font ---
# INFO_FONT_SIZE = 16 # REMOVED
TREE_FONT_SIZE = 12
BUTTON_FONT_SIZE = 14
INPUT_FONT_SIZE = 16
COORD_FONT_SIZE = 12
try:
    # INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE) # REMOVED
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
    BUTTON_FONT = pygame.font.SysFont("sans", BUTTON_FONT_SIZE)
    INPUT_FONT = pygame.font.SysFont("monospace", INPUT_FONT_SIZE) # Font for input box
    COORD_FONT = pygame.font.SysFont("sans", COORD_FONT_SIZE)     # Font for a-h, 1-8
except Exception as e:
    print(f"Warning: Could not load fonts ({e}). Using default.")
    # INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE + 2) # REMOVED
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)
    BUTTON_FONT = pygame.font.Font(None, BUTTON_FONT_SIZE + 2)
    INPUT_FONT = pygame.font.Font(None, INPUT_FONT_SIZE + 2)
    COORD_FONT = pygame.font.Font(None, COORD_FONT_SIZE + 2)


# --- Chess Configuration ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Make sure this path is correct
ANALYSIS_TIME_LIMIT = 0.4
BEST_MOVE_ANALYSIS_TIME = 0.8
NUM_BEST_MOVES_TO_SHOW = 5
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = 15000 # Used for plotting mate scores

# --- Asset Loading ---
PIECE_IMAGE_PATH = "pieces"
PIECE_IMAGES = {}
def load_piece_images(path=PIECE_IMAGE_PATH, sq_size=SQ_SIZE): # (Unchanged logic)
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    loaded_images = {}
    if not os.path.isdir(path): print(f"Error: Piece image directory not found: '{path}'"); return None
    all_loaded = True
    for piece in pieces:
        file_path = os.path.join(path, f"{piece}.png")
        try:
            img = pygame.image.load(file_path).convert_alpha()
            img = pygame.transform.smoothscale(img, (sq_size, sq_size))
            loaded_images[piece] = img
        except pygame.error as e: print(f"Error loading piece image '{file_path}': {e}"); all_loaded = False
    if not all_loaded: print("Please ensure all 12 piece PNG files exist..."); return None
    print(f"Loaded {len(loaded_images)} piece images.")
    return loaded_images

# --- Move Quality Classification (Unchanged) ---
def classify_move_quality(points_lost):
    if points_lost <= 0.00: return "Best"
    elif points_lost <= 0.02: return "Excellent"
    elif points_lost <= 0.05: return "Good"
    elif points_lost <= 0.10: return "Inaccuracy"
    elif points_lost <= 0.20: return "Mistake"
    else: return "Blunder"

# --- Game History Tree Node (Unchanged) ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen; self.move = move; self.parent = parent; self.children = []
        self.raw_score = raw_score; self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
        self._san_cache = None; self.x = 0; self.y = 0; self.screen_rect = None; self.move_quality = None
    def add_child(self, child_node): 
        self.children.append(child_node)
    def get_ply(self): 
        count = 0; node = self; 
        while node.parent: 
            count += 1; 
            node = node.parent; 
        return count
    def get_san(self, board_at_parent):
        if self._san_cache is not None: return self._san_cache
        if not self.move or not self.parent: return "root"
        try: san = board_at_parent.san(self.move); self._san_cache = san; return san
        except: return self.move.uci() # Fallback
    def calculate_and_set_move_quality(self):
        if not self.parent or self.parent.raw_score is None or self.raw_score is None: self.move_quality = None; return
        wp_before = self.parent.white_percentage; wp_after = self.white_percentage
        wp_before = 50.0 if wp_before is None else wp_before; wp_after = 50.0 if wp_after is None else wp_after
        try: parent_board = chess.Board(self.parent.fen, chess960=True); turn_before_move = parent_board.turn
        except: self.move_quality = None; return
        eval_before_pov = (wp_before / 100.0) if turn_before_move == chess.WHITE else ((100.0 - wp_before) / 100.0)
        eval_after_pov = (wp_after / 100.0) if turn_before_move == chess.WHITE else ((100.0 - wp_after) / 100.0)
        points_lost = max(0.0, eval_before_pov - eval_after_pov); self.move_quality = classify_move_quality(points_lost)

# --- Helper Functions (Chess Logic - Unchanged) ---

def format_score(score, turn):
    if score is None: return "N/A"
    pov_score = score.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate (N/A)"
    else:
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        return f"{cp / 100.0:+.2f}" if cp is not None else "N/A"
        
def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT): # (Unchanged)
    if score is None: return None
    pov_score = score.white()
    if pov_score.is_mate(): mate_val = pov_score.mate(); return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    cp_val = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE);
    if cp_val is None: return None
    clamped_cp = max(-clamp_limit, min(clamp_limit, cp_val)); normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
    return normalized * 100.0

def get_engine_analysis(board, engine, time_limit): # (Unchanged)
    if not engine: return None, "Engine unavailable.", None
    try: info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    except Exception as e: return None, f"Engine analysis error: {e}", None # Catch engine errors early
    best_move = info.get("pv", [None])[0]; score = info.get("score")
    if score is None and not info.get("pv"): return None, "Analysis failed (no score/pv).", None
    score_str = format_score(score, board.turn)
    if best_move:
        try: best_move_san = board.san(best_move)
        except Exception: best_move_san = best_move.uci()
        return best_move_san, f"Score: {score_str}", score
    elif score is not None: return None, f"Pos Score: {score_str}", score
    else: return None, "Engine analysis failed.", None

def get_top_engine_moves(board, engine, time_limit, num_moves): # (Unchanged)
    if not engine: return [], "Engine not available."
    if board.is_game_over(): return [], "Game is over."
    moves_info = []
    try:
        with engine.analysis(board, chess.engine.Limit(time=time_limit), multipv=num_moves) as analysis:
            for info in analysis:
                if "pv" in info and info["pv"]:
                    move = info["pv"][0]; score = info.get("score"); score_str = format_score(score, board.turn)
                    try: move_san = board.san(move)
                    except Exception: move_san = move.uci()
                    moves_info.append({"move": move, "san": move_san, "score_str": score_str, "score_obj": score})
                    if len(moves_info) >= num_moves: break
        return moves_info, None
    except Exception as e: print(f"MultiPV Analysis error: {e}"); return [], f"Analysis error: {e}"

def get_path_to_node(node): # (Unchanged)
    path = []; current = node; 
    while current: 
        path.append(current); 
        current = current.parent; 
    return path[::-1]

# --- Drawing Functions ---

def draw_board(surface): # Needs coordinates
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (BOARD_X + c * SQ_SIZE, BOARD_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(surface, board, piece_images, dragging_piece_info): # Needs coordinates
    if piece_images is None: return
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if dragging_piece_info and square == dragging_piece_info['square']: continue
            piece_symbol = piece.symbol(); piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper(); img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square); file = chess.square_file(square)
                screen_x = BOARD_X + file * SQ_SIZE
                screen_y = BOARD_Y + (7 - rank) * SQ_SIZE
                surface.blit(img, (screen_x, screen_y))
    # Draw dragging piece last (on top)
    if dragging_piece_info and dragging_piece_info['img']:
        img_rect = dragging_piece_info['img'].get_rect(center=dragging_piece_info['pos'])
        surface.blit(dragging_piece_info['img'], img_rect)

def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight): # Needs coordinates
    # Engine move highlight
    if engine_move_to_highlight:
        for sq in [engine_move_to_highlight.from_square, engine_move_to_highlight.to_square]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill(ENGINE_MOVE_HIGHLIGHT_COLOR); surface.blit(s, highlight_rect.topleft)
    # Last move highlight
    if last_move:
        for sq in [last_move.from_square, last_move.to_square]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill(LAST_MOVE_HIGHLIGHT_COLOR); surface.blit(s, highlight_rect.topleft)
    # Selected square highlight
    if selected_square is not None:
        rank = chess.square_rank(selected_square); file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill(HIGHLIGHT_COLOR); surface.blit(s, highlight_rect.topleft)
    # Possible moves highlight
    if legal_moves_for_selected:
        for move in legal_moves_for_selected:
            dest_sq = move.to_square; rank = chess.square_rank(dest_sq); file = chess.square_file(dest_sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); is_capture = board.is_capture(move)
            center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2; radius = SQ_SIZE // 6
            if is_capture: pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius + 3, 3) # Thicker outline for capture
            else: pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius)
            surface.blit(s, (BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE))

def draw_board_badge(surface, square, quality): # Needs coordinates, uses larger badge constants
    badge_image = BADGE_IMAGES.get(quality)
    badge_color = TREE_MOVE_QUALITY_COLORS.get(quality)
    if quality is None or badge_image is None or badge_color is None: return

    rank = chess.square_rank(square); file = chess.square_file(square)
    square_base_x = BOARD_X + file * SQ_SIZE
    square_base_y = BOARD_Y + (7 - rank) * SQ_SIZE
    center_x = square_base_x + BOARD_BADGE_OFFSET_X
    center_y = square_base_y + BOARD_BADGE_OFFSET_Y

    pygame.draw.circle(surface, badge_color, (center_x, center_y), BOARD_BADGE_RADIUS)
    pygame.draw.circle(surface, BOARD_BADGE_OUTLINE_COLOR, (center_x, center_y), BOARD_BADGE_RADIUS, 1)
    badge_rect = badge_image.get_rect(center=(center_x, center_y))
    surface.blit(badge_image, badge_rect.topleft)

def draw_eval_bar(surface, white_percentage): # Needs positioning update
    bar_height = BOARD_SIZE # Eval bar matches board height
    bar_x = BOARD_X + BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = BOARD_Y # Align with board top
    white_percentage = 50.0 if white_percentage is None else white_percentage
    white_height = int(bar_height * (white_percentage / 100.0)); black_height = bar_height - white_height
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1) # Outline

def create_eval_plot_surface(node_path, plot_width_px, plot_height_px): # <<< SIMPLIFIED
    """Creates a surface with the evaluation plot (line + filled area, no text)."""
    if not node_path or len(node_path) < 2: return None

    plies = [node.get_ply() for node in node_path]
    percentages = [(node.white_percentage if node.white_percentage is not None else 50.0) for node in node_path]

    dark_grey_mpl = tuple(c/255.0 for c in DARK_GREY)
    orange_mpl = tuple(c/255.0 for c in ORANGE)
    grey_mpl = tuple(c/255.0 for c in GREY)
    white_mpl = tuple(c/255.0 for c in WHITE)

    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl)
    ax.set_facecolor(dark_grey_mpl)

    # Plotting
    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=0.7)
        ax.plot(plies, percentages, color=white_mpl, marker=None, linestyle='-', linewidth=1.5) # No markers
    elif len(plies) == 1:
         ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=3) # Small marker for single point

    # Styling - Minimal
    ax.axhline(50, color=orange_mpl, linestyle='--', linewidth=1) # 50% line still useful
    ax.set_xlim(0, max(max(plies), 1) if plies else 1)
    ax.set_ylim(0, 100)

    # --- REMOVE ALL TEXT ---
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([]) # Remove x-axis ticks
    ax.set_yticks([]) # Remove y-axis ticks
    # ax.grid(False) # Optionally remove grid too
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl, alpha=0.4) # Keep faint grid?

    # Keep spines faint
    ax.spines['top'].set_color(grey_mpl)
    ax.spines['bottom'].set_color(grey_mpl)
    ax.spines['left'].set_color(grey_mpl)
    ax.spines['right'].set_color(grey_mpl)

    # Save plot to buffer
    plt.tight_layout(pad=0.1) # Minimal padding
    buf = io.BytesIO()
    try: plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, bbox_inches='tight', pad_inches=0.05) # Tight bounding box
    except Exception as e: print(f"Error saving plot: {e}"); plt.close(fig); buf.close(); return None
    finally: plt.close(fig)
    buf.seek(0)

    # Load buffer into Pygame surface
    plot_surface = None
    try: plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e: print(f"Error loading plot image: {e}")
    finally: buf.close()
    return plot_surface

# REMOVED: def draw_game_info(...)

# --- NEW: Draw Coordinates ---
def draw_coordinates(surface, font):
    # Files (a-h) - Below board
    for i in range(8):
        text = chr(ord('a') + i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y + BOARD_SIZE + COORD_PADDING // 2))
        surface.blit(text_surf, text_rect)
        # Optional: Draw above board too
        # text_rect_top = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y - COORD_PADDING // 2))
        # surface.blit(text_surf, text_rect_top)

    # Ranks (1-8) - Left of board
    for i in range(8):
        text = str(8 - i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X - COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2))
        surface.blit(text_surf, text_rect)
        # Optional: Draw right of board too
        # text_rect_right = text_surf.get_rect(center=(BOARD_X + BOARD_SIZE + COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2))
        # surface.blit(text_surf, text_rect_right)


# --- Horizontal Game Tree Drawing (Needs coordinate updates) ---
TREE_PIECE_SIZE = 20
NODE_DIAMETER = TREE_PIECE_SIZE
HORIZ_SPACING = 45 + TREE_PIECE_SIZE
VERT_SPACING = 5 + TREE_PIECE_SIZE
TEXT_OFFSET_X = 3
INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 2 # Keep large initial size
INITIAL_TREE_SURFACE_HEIGHT = TREE_PANEL_HEIGHT * 4 # Maybe relate to panel height
drawn_tree_nodes = {}
tree_scroll_x = 0
tree_scroll_y = 0
max_drawn_tree_x = 0
max_drawn_tree_y = 0
tree_render_surface = None
temp_san_board = chess.Board(chess960=True)
scaled_tree_piece_images = {}

def get_scaled_tree_image(piece_key, target_size): # (Unchanged)
    global PIECE_IMAGES, scaled_tree_piece_images
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images: return scaled_tree_piece_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key)
    if original_img:
        try: scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size)); scaled_tree_piece_images[cache_key] = scaled_img; return scaled_img
        except: return None
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node): # (Unchanged logic)
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board
    node.x = x; node.y = y_center
    piece_img = None; is_root = not node.parent
    if not is_root:
        try:
            parent_board = chess.Board(node.parent.fen, chess960=True); moved_piece = parent_board.piece_at(node.move.from_square)
            if moved_piece: piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper(); piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE)
        except: pass # Ignore errors finding piece

    child_y_positions = []; child_heights = []; total_child_height_estimate = 0; child_x = x + HORIZ_SPACING
    if node.children:
        total_child_height_estimate = (len(node.children) - 1) * VERT_SPACING # Initial guess
        current_child_y = y_center - total_child_height_estimate / 2
        for i, child in enumerate(node.children):
            child_actual_y, child_subtree_height = layout_and_draw_tree_recursive(surface, child, child_x, current_child_y, level + 1, font, current_node)
            child_y_positions.append(child_actual_y); child_heights.append(child_subtree_height)
            spacing = max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2 if child_subtree_height > 0 else VERT_SPACING)
            current_child_y += spacing

    node_rect = None
    if piece_img: img_rect = piece_img.get_rect(center=(int(node.x), int(node.y))); surface.blit(piece_img, img_rect.topleft); node_rect = img_rect
    elif is_root: radius = TREE_PIECE_SIZE // 2; pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius); node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    else: node_rect = pygame.Rect(node.x-2, node.y-2, 4, 4) # Fallback dot

    if node_rect: max_drawn_tree_x = max(max_drawn_tree_x, node_rect.right)
    if node == current_node and node_rect: pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(4, 4), 1) # Thicker outline

    if node_rect and node.move_quality and node.move_quality in TREE_MOVE_QUALITY_COLORS:
        badge_color = TREE_MOVE_QUALITY_COLORS[node.move_quality]
        badge_center_x = node_rect.centerx + TREE_PIECE_SIZE // 2 - TREE_BADGE_RADIUS - 1
        badge_center_y = node_rect.centery + TREE_PIECE_SIZE // 2 - TREE_BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS)
        pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS, 1)

    move_text = ""; text_rect = None
    if node.parent: move_text = node.get_san(chess.Board(node.parent.fen, chess960=True)) # Assumes board valid
    if move_text and node_rect:
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR); text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery)); surface.blit(text_surf, text_rect)
        max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)

    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect: clickable_rect.width = max(clickable_rect.width, text_rect.right - clickable_rect.left) # Expand clickable area
    node.screen_rect = clickable_rect; drawn_tree_nodes[node] = node.screen_rect

    if node_rect: # Draw lines after node is placed
        for i, child in enumerate(node.children):
            if hasattr(child, 'x') and hasattr(child, 'y'):
               child_visual_rect = child.screen_rect if child.screen_rect else pygame.Rect(child.x-1, child.y-1,2,2)
               start_pos = (node_rect.right, node_rect.centery)
               end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)

    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE
    subtree_total_height = 0
    if child_y_positions:
        min_child_y = min(child_y_positions) - (max(child_heights)/2 if child_heights else 0)
        max_child_y = max(child_y_positions) + (max(child_heights)/2 if child_heights else 0)
        subtree_total_height = max(0, max_child_y - min_child_y)
    max_drawn_tree_y = max(max_drawn_tree_y, node.y + max(my_height, subtree_total_height) / 2)
    return node.y, max(my_height, subtree_total_height)


def draw_game_tree(surface, root_node, current_node, font): # Needs coordinate update for panel position
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface
    drawn_tree_nodes.clear(); max_drawn_tree_x = 0; max_drawn_tree_y = 0

    # --- Calculate Tree Panel Rect based on other panels ---
    plot_panel_y = BOARD_Y + BOARD_SIZE
    tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
    tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

    # Surface creation/resizing (ensure large enough)
    required_width = max(INITIAL_TREE_SURFACE_WIDTH, int(max_drawn_tree_x + 2 * HORIZ_SPACING)) # Estimate needed width
    required_height = max(INITIAL_TREE_SURFACE_HEIGHT, int(max_drawn_tree_y + 2 * VERT_SPACING)) # Estimate needed height

    if tree_render_surface is None or tree_render_surface.get_width() < required_width or tree_render_surface.get_height() < required_height:
         try:
             tree_render_surface = pygame.Surface((required_width, required_height))
             # print(f"Resized tree surface to {required_width}x{required_height}") # Debug
         except pygame.error as e: print(f"Error resizing tree surface: {e}"); return

    tree_render_surface.fill(TREE_BG_COLOR)
    if not root_node: surface.blit(tree_render_surface, tree_panel_rect.topleft); return # Draw empty panel

    start_x = 15 + TREE_PIECE_SIZE // 2
    start_y = tree_render_surface.get_height() // 2 # Initial root Y pos

    # Layout and draw recursively
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node)

    # Calculate scroll limits based on actual drawn content
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING
    total_tree_height = max_drawn_tree_y + VERT_SPACING # Max Y reached + padding

    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width)
    max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)

    # Auto-scroll logic (Unchanged)
    scroll_margin_x = HORIZ_SPACING; scroll_margin_y = VERT_SPACING * 2
    if hasattr(current_node, 'x') and hasattr(current_node, 'y'): # Ensure node has coords
        if current_node.x > tree_scroll_x + tree_panel_rect.width - scroll_margin_x: tree_scroll_x = current_node.x - tree_panel_rect.width + scroll_margin_x
        if current_node.x < tree_scroll_x + scroll_margin_x: tree_scroll_x = current_node.x - scroll_margin_x
        if current_node.y > tree_scroll_y + tree_panel_rect.height - scroll_margin_y: tree_scroll_y = current_node.y - tree_panel_rect.height + scroll_margin_y
        if current_node.y < tree_scroll_y + scroll_margin_y: tree_scroll_y = current_node.y - scroll_margin_y

    # Clamp scroll
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x))
    tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))

    # Blit visible portion
    source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, tree_panel_rect.width, tree_panel_rect.height)
    try: surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
    except pygame.error as e: print(f"Error blitting tree: {e}")

    # Draw scrollbars (Unchanged logic, position based on tree_panel_rect)
    scrollbar_thickness = 7
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width; scrollbar_width = max(15, tree_panel_rect.width * ratio_visible); scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio; scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - scrollbar_thickness - 1, scrollbar_width, scrollbar_thickness)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height; scrollbar_height = max(15, tree_panel_rect.height * ratio_visible); scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0
        scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio; scrollbar_rect = pygame.Rect(tree_panel_rect.right - scrollbar_thickness - 1, scrollbar_y, scrollbar_thickness, scrollbar_height)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)


# --- Coordinate Conversion --- Needs BOARD_X, BOARD_Y offset
def screen_to_square(pos):
    x, y = pos
    # Check if click is within the board bounds
    if x < BOARD_X or x >= BOARD_X + BOARD_SIZE or y < BOARD_Y or y >= BOARD_Y + BOARD_SIZE:
        return None
    # Convert screen coords relative to board origin
    file = (x - BOARD_X) // SQ_SIZE
    rank = 7 - ((y - BOARD_Y) // SQ_SIZE)
    return chess.square(file, rank)

# --- NEW: Setup Panel Drawing ---
def draw_setup_panel(surface, input_rect, button_rect, input_text, input_active, font):
    # Panel background
    panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SETUP_PANEL_HEIGHT)
    pygame.draw.rect(surface, DARK_GREY, panel_rect)

    # Input field background
    pygame.draw.rect(surface, INPUT_BG_COLOR, input_rect, border_radius=3)

    # Input field border (thicker if active)
    border_color = INPUT_BORDER_ACTIVE_COLOR if input_active else INPUT_BORDER_INACTIVE_COLOR
    pygame.draw.rect(surface, border_color, input_rect, 2, border_radius=3)

    # Input field text
    text_surface = font.render(input_text, True, INPUT_TEXT_COLOR)
    # Position text slightly indented within the input box
    text_rect = text_surface.get_rect(midleft=(input_rect.left + 5, input_rect.centery))
    # Clip text if too long? For now, let it overflow or handle in input logic
    surface.blit(text_surface, text_rect)

    # Blinking cursor (optional enhancement)
    if input_active and int(time.time() * 2) % 2 == 0: # Blink approx every 0.5s
         cursor_x = text_rect.right + 1
         cursor_y_start = input_rect.top + 4
         cursor_y_end = input_rect.bottom - 4
         pygame.draw.line(surface, INPUT_TEXT_COLOR, (cursor_x, cursor_y_start), (cursor_x, cursor_y_end), 1)


    # Button background and border
    pygame.draw.rect(surface, BUTTON_COLOR, button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, button_rect, 1, border_radius=3)

    # Button text
    btn_text_surf = BUTTON_FONT.render("Set/Random", True, BUTTON_TEXT_COLOR)
    btn_text_rect = btn_text_surf.get_rect(center=button_rect.center)
    surface.blit(btn_text_surf, btn_text_rect)


# --- Main Game Function ---
def play_chess960_pygame():
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y, BADGE_IMAGES, all_badges_loaded
    # --- Game State Variables ---
    board = chess.Board(chess960=True) # Initialize board object
    engine = None
    message = "" # Persistent status message area (optional)
    game_root = None
    current_node = None
    last_raw_score = None
    one_off_analysis_text = None # For temporary analysis results
    meter_visible = True # Eval plot visibility
    plot_surface = None
    needs_redraw = True
    selected_square = None
    dragging_piece_info = None
    legal_moves_for_selected = []
    last_move_displayed = None
    best_moves_cache = {}
    current_best_move_index = -1
    highlighted_engine_move = None

    # --- Setup Panel State ---
    input_field_rect = pygame.Rect(COORD_PADDING + 50, 5, 100, SETUP_PANEL_HEIGHT - 10)
    setup_button_rect = pygame.Rect(input_field_rect.right + 10, 5, 120, SETUP_PANEL_HEIGHT - 10)
    input_text = ""
    input_active = False


    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: print("Exiting: Missing piece images."); return
    if not all_badges_loaded: print("Warning: Running without all badge images.")

    # --- Engine Initialization ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try: engine.configure({"UCI_Chess960": True})
            except Exception as e: print(f"Warning: Could not set UCI_Chess960 for engine: {e}") # Informative warning
            print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
        else: print(f"Warning: Stockfish not found at '{STOCKFISH_PATH}'. Analysis disabled.")
    except Exception as e: print(f"Error initializing Stockfish: {e}. Analysis disabled."); engine = None

    # --- NEW: Reset Game Function ---
    def reset_game(start_pos_num):
        nonlocal board, game_root, current_node, last_raw_score, message
        nonlocal best_moves_cache, current_best_move_index, highlighted_engine_move
        nonlocal one_off_analysis_text, selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal needs_redraw
        try:
            board.set_chess960_pos(start_pos_num)
            start_fen = board.fen()
            print(f"Resetting to Chess960 Position {start_pos_num} (FEN: {start_fen})")

            # Clear game tree and related state
            game_root = None
            current_node = None
            last_raw_score = None
            best_moves_cache = {}
            drawn_tree_nodes.clear() # Clear clickable areas cache
            tree_scroll_x = 0 # Reset tree scroll
            tree_scroll_y = 0

            # Clear transient UI state
            one_off_analysis_text = None
            selected_square = None
            dragging_piece_info = None
            legal_moves_for_selected = []
            current_best_move_index = -1
            highlighted_engine_move = None

            # Get initial analysis for the root node if engine exists
            initial_score = None
            analysis_msg = ""
            if engine:
                _, analysis_msg, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5)
                if initial_score is None: print(f"Warning: Initial analysis failed. {analysis_msg}")

            # Create the root node
            game_root = GameNode(fen=start_fen, raw_score=initial_score)
            current_node = game_root
            last_raw_score = initial_score
            message = f"Position {start_pos_num} set." # Update message

            needs_redraw = True

        except Exception as e:
            print(f"Error resetting game to position {start_pos_num}: {e}")
            message = f"Error setting pos {start_pos_num}."
            # Should perhaps revert to a default state? For now, just show error.
            needs_redraw = True


    # --- Initial Game Setup ---
    initial_random_pos = random.randint(0, 959)
    input_text = str(initial_random_pos) # Show initial random pos in box
    reset_game(initial_random_pos)


    running = True; clock = pygame.time.Clock(); selected_square = None; dragging_piece_info = None
    legal_moves_for_selected = []; last_move_displayed = None; tree_scroll_speed = 30

    # --- Main Game Loop ---
    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        # Calculate panel Y positions dynamically based on constants
        plot_panel_y = BOARD_Y + BOARD_SIZE
        tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
        tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT) # Used for tree scroll events

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            # --- Mouse Wheel (Tree Scroll) ---
            elif event.type == pygame.MOUSEWHEEL:
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    tree_scroll_y -= event.y * tree_scroll_speed
                    tree_scroll_y = max(0, tree_scroll_y) # Clamp bottom
                    # Max clamping happens in draw_game_tree
                    needs_redraw = True

            # --- Mouse Click ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # Activate Input Field
                if input_field_rect.collidepoint(pos):
                    input_active = True
                else:
                    input_active = False

                # --- Setup Button Click ---
                if setup_button_rect.collidepoint(pos):
                    chosen_pos = -1
                    try:
                        num = int(input_text)
                        if 0 <= num <= 959:
                            chosen_pos = num
                        else: # Invalid number range
                            print(f"Invalid input: '{input_text}'. Using random.")
                            chosen_pos = random.randint(0, 959)
                            input_text = str(chosen_pos) # Update input field
                    except ValueError: # Not an integer
                         print(f"Invalid input: '{input_text}'. Using random.")
                         chosen_pos = random.randint(0, 959)
                         input_text = str(chosen_pos) # Update input field

                    if chosen_pos != -1:
                        reset_game(chosen_pos)
                    input_active = False # Deactivate input after button press

                # --- Tree Click ---
                elif tree_panel_rect.collidepoint(pos):
                    if input_active: input_active = False # Deactivate input if clicking tree
                    relative_panel_x = pos[0]-tree_panel_rect.left; relative_panel_y = pos[1]-tree_panel_rect.top
                    absolute_tree_click_x = relative_panel_x + tree_scroll_x; absolute_tree_click_y = relative_panel_y + tree_scroll_y
                    clicked_node = None
                    nodes_to_check = list(drawn_tree_nodes.items())
                    for node, rect in nodes_to_check:
                        if rect and rect.collidepoint(absolute_tree_click_x, absolute_tree_click_y): clicked_node = node; break
                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                        message = f"Jumped to ply {current_node.get_ply()}" # Update message
                        reset_transient_state()
                        needs_redraw = True

                # --- Board Click (Piece Selection/Drag Start) ---
                elif BOARD_X <= pos[0] < BOARD_X + BOARD_SIZE and BOARD_Y <= pos[1] < BOARD_Y + BOARD_SIZE:
                    if input_active: input_active = False # Deactivate input if clicking board
                    if event.button == 1: # Left Click
                        highlighted_engine_move = None; current_best_move_index = -1 # Clear engine highlight
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn: # Select own piece
                                selected_square = sq; piece_symbol = piece.symbol(); piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                                img = PIECE_IMAGES.get(piece_key)
                                if img: dragging_piece_info = {'square': sq, 'piece': piece, 'img': img, 'pos': pos}; legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                else: reset_transient_state() # Should not happen
                                needs_redraw = True
                            else: # Clicked empty/opponent
                                reset_transient_state()
                        else: # Clicked off board / game over
                            reset_transient_state()
                    elif event.button == 3: # Right Click on board
                         reset_transient_state()

                # Click elsewhere (deactivate input) - Handled by individual checks above

            # --- Mouse Button Release (Drop Piece) ---
            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 and dragging_piece_info:
                    pos = event.pos; to_sq = screen_to_square(pos); from_sq = dragging_piece_info['square']
                    move_made = False
                    # Check drop validity and move legality
                    if to_sq is not None and from_sq != to_sq:
                        promotion = None; piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]: promotion = chess.QUEEN # Auto-queen
                        move = chess.Move(from_sq, to_sq, promotion=promotion)
                        if move in board.legal_moves:
                            move_made = True; existing_child = None
                            for child in current_node.children:
                                if child.move == move: existing_child = child; break
                            if existing_child: # Navigate to existing node
                                current_node = existing_child; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score; message = f"Played {existing_child.get_san(chess.Board(existing_child.parent.fen, chess960=True))}"
                            else: # Create new node
                                parent_node = current_node
                                board.push(move); new_fen = board.fen(); board.pop() # Get FEN
                                temp_board_for_analysis = chess.Board(new_fen, chess960=True)
                                new_raw_score = None; analysis_failed = False
                                if engine: _, _, new_raw_score = get_engine_analysis(temp_board_for_analysis, engine, ANALYSIS_TIME_LIMIT); analysis_failed = new_raw_score is None
                                new_node = GameNode(fen=new_fen, move=move, parent=parent_node, raw_score=new_raw_score)
                                new_node.calculate_and_set_move_quality()
                                parent_node.add_child(new_node); current_node = new_node # Set new node as current
                                board.set_fen(current_node.fen); last_raw_score = current_node.raw_score # Update main board
                                san = new_node.get_san(chess.Board(parent_node.fen, chess960=True))
                                quality_msg = f" ({new_node.move_quality})" if new_node.move_quality else ""
                                message = f"Played {san}{quality_msg}"
                                if analysis_failed: message += " (Analysis Failed)"

                            reset_transient_state(clear_message=False) # Keep move message
                            needs_redraw = True
                        else: message = f"Illegal move"; needs_redraw = True # Keep message, clear drag below

                    # Always clear dragging info after mouse up
                    dragging_piece_info = None
                    if not move_made: # If move failed or dropped off board, clear selection
                        selected_square = None; legal_moves_for_selected = []
                        needs_redraw = True


            # --- Mouse Motion (Drag Piece) ---
            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info: dragging_piece_info['pos'] = event.pos; needs_redraw = True

            # --- Key Presses ---
            elif event.type == pygame.KEYDOWN:
                 # Handle Input Field Typing
                if input_active:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                         # Simulate button click on Enter
                        chosen_pos = -1
                        try: 
                            num = int(input_text);
                            if 0 <= num <= 959: 
                                chosen_pos = num
                            else: 
                                chosen_pos = random.randint(0, 959); 
                                input_text = str(chosen_pos)
                        except ValueError: chosen_pos = random.randint(0, 959); input_text = str(chosen_pos)
                        if chosen_pos != -1: 
                            reset_game(chosen_pos)
                        input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isdigit(): # Only allow digits
                        if len(input_text) < 3: # Limit length
                            input_text += event.unicode
                    needs_redraw = True # Redraw input field as text changes
                else: # Handle game navigation/commands if input not active
                    node_changed = False
                    if event.key == pygame.K_LEFT: # Navigate back
                        if current_node.parent: current_node = current_node.parent; node_changed = True; message = f"Back (Ply {current_node.get_ply()})"
                        else: message = "At start"
                    elif event.key == pygame.K_RIGHT: # Navigate forward (main line)
                        if current_node.children: current_node = current_node.children[0]; node_changed = True; message = f"Forward (Ply {current_node.get_ply()})"
                        else: message = "End of line"

                    if node_changed:
                        board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                        reset_transient_state(clear_message=False) # Keep navigation message
                        needs_redraw = True

                    # --- Other commands (Analyze, Toggle Plot, etc.) ---
                    elif event.key == pygame.K_a: # Analyze Current Position
                         if engine:
                             message = "Analyzing current position..."; one_off_analysis_text = "Analyzing..." # Show immediate feedback
                             needs_redraw = True; pygame.display.flip() # Quick update display

                             best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 2) # Slightly longer explicit analysis
                             if raw_score is not None:
                                 current_node.raw_score = raw_score; current_node.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                                 last_raw_score = raw_score
                                 current_node.calculate_and_set_move_quality() # Re-calc quality of move *leading* here
                             one_off_analysis_text = f"Suggests: {best_move_san} ({score_str})" if best_move_san else score_str if score_str else "Analysis complete."
                             message = ""; needs_redraw = True # Clear persistent message
                             highlighted_engine_move = None; current_best_move_index = -1 # Clear any best move highlight
                         else: message = "No engine available."

                    elif event.key == pygame.K_m: # Toggle Eval Plot
                        meter_visible = not meter_visible; message = f"Eval Plot {'ON' if meter_visible else 'OFF'}"; needs_redraw = True
                    elif event.key == pygame.K_ESCAPE: running = False

        # --- Reset transient state helper (modified) ---
        def reset_transient_state(clear_message=True):
            nonlocal selected_square, dragging_piece_info, legal_moves_for_selected
            nonlocal one_off_analysis_text, needs_redraw, message
            nonlocal current_best_move_index, highlighted_engine_move
            selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
            one_off_analysis_text = None
            if clear_message: message = "" # Optionally clear persistent message
            current_best_move_index = -1; highlighted_engine_move = None
            # needs_redraw should be set by the caller context


        # --- Update state before drawing ---
        last_move_displayed = current_node.move if current_node.parent else None
        # Ensure board object matches current node FEN (critical after navigation/reset)
        if board.fen() != current_node.fen:
             try: board.set_fen(current_node.fen)
             except ValueError: print(f"CRITICAL ERROR: Invalid FEN in node! FEN: {current_node.fen}"); running=False

        # --- Drawing ---
        if needs_redraw:
            screen.fill(DARK_GREY) # Background

            # --- Draw UI Elements ---
            draw_setup_panel(screen, input_field_rect, setup_button_rect, input_text, input_active, INPUT_FONT)
            draw_coordinates(screen, COORD_FONT)
            draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)

            # Draw board badge (larger)
            if current_node.move and current_node.parent and current_node.move_quality:
                draw_board_badge(screen, current_node.move.to_square, current_node.move_quality)

            draw_eval_bar(screen, current_node.white_percentage)

            # Button text logic (Unchanged)
            button_text = "Show Best Move"; current_fen = current_node.fen; cached_moves = best_moves_cache.get(current_fen)
            if highlighted_engine_move and cached_moves:
                num_cached = len(cached_moves)
                button_text = f"Showing {current_best_move_index + 1}/{num_cached}"
            elif engine and not board.is_game_over(): button_text = "Show Best (1)"
            elif current_fen in best_moves_cache and not cached_moves: button_text = "No moves found"

            # Draw Eval Plot (simplified)
            if meter_visible:
                path = get_path_to_node(current_node)
                # Generate plot only if path has changed or surface doesn't exist? Optimization possible.
                plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT) # Regenerate each time for now
                if plot_surface:
                    plot_rect = plot_surface.get_rect(topleft=(BOARD_X, plot_panel_y)) # Align plot with board X
                    screen.blit(plot_surface, plot_rect)
                else: # Draw placeholder if plot fails
                    plot_rect = pygame.Rect(BOARD_X, plot_panel_y, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                    pygame.draw.rect(screen, DARK_GREY, plot_rect); pygame.draw.rect(screen, GREY, plot_rect, 1)

            # Draw Game Tree
            draw_game_tree(screen, game_root, current_node, TREE_FONT)

            # Draw Status Message (Optional - Can place it somewhere like bottom right)
            if message:
                 status_surf = TREE_FONT.render(message, True, WHITE)
                 status_rect = status_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10))
                 screen.blit(status_surf, status_rect)
            if one_off_analysis_text:
                 analysis_surf = TREE_FONT.render(one_off_analysis_text, True, ORANGE) # Use different color?
                 # Place analysis text e.g., above status message
                 analysis_rect = analysis_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, SCREEN_HEIGHT - 25 if message else SCREEN_HEIGHT - 10))
                 screen.blit(analysis_surf, analysis_rect)


            pygame.display.flip()
            needs_redraw = False # Reset redraw flag

        clock.tick(60)

    # --- Cleanup ---
    print("\nExiting Pygame...")
    if engine:
        try: engine.quit(); print("Stockfish engine closed.")
        except Exception as e: print(f"Error closing engine: {e}")
    plt.close('all'); pygame.quit(); sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()