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
matplotlib.use('Agg')

# --- Pygame & Display Configuration ---
pygame.init()
pygame.font.init()
BOARD_SIZE = 512
SQ_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = 100 # Increased slightly for button
PLOT_PANEL_HEIGHT = 120
TREE_PANEL_HEIGHT = 180
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20
SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board with Piece History Tree")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60)
LIGHT_SQ_COLOR = (238, 238, 210)
DARK_SQ_COLOR = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150) # Yellowish for selected square
LAST_MOVE_HIGHLIGHT_COLOR = (186, 202, 68, 180) # Greenish for last move
ENGINE_MOVE_HIGHLIGHT_COLOR = (0, 150, 255, 150) # Bluish for engine move suggestion
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
ORANGE = (255, 165, 0)
TREE_BG_COLOR = DARK_GREY
TREE_LINE_COLOR = (150, 150, 150)
TREE_NODE_ROOT_COLOR = (100, 100, 150)
TREE_NODE_CURRENT_OUTLINE = (255, 100, 100)
TREE_TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (80, 80, 100)
BUTTON_TEXT_COLOR = WHITE

# --- Badge Colors ---
BADGE_BEST_COLOR = (255, 215, 0) # Gold
BADGE_EXCELLENT_COLOR = (34, 139, 34) # Forest Green
BADGE_GOOD_COLOR = (135, 206, 250) # Light Sky Blue
BADGE_INACCURACY_COLOR = (240, 230, 140) # Khaki Yellow
BADGE_MISTAKE_COLOR = (255, 140, 0) # Dark Orange
BADGE_BLUNDER_COLOR = (200, 0, 0) # Red

MOVE_QUALITY_COLORS = {
    "Best": BADGE_BEST_COLOR,
    "Excellent": BADGE_EXCELLENT_COLOR,
    "Good": BADGE_GOOD_COLOR,
    "Inaccuracy": BADGE_INACCURACY_COLOR,
    "Mistake": BADGE_MISTAKE_COLOR,
    "Blunder": BADGE_BLUNDER_COLOR,
}
BADGE_RADIUS = 4 # Size of the badge circle
# Offset calculation will happen inside the drawing function based on node_rect

# --- Font ---
INFO_FONT_SIZE = 16
TREE_FONT_SIZE = 12
BUTTON_FONT_SIZE = 14
try:
    INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
    BUTTON_FONT = pygame.font.SysFont("sans", BUTTON_FONT_SIZE)
except Exception as e:
    print(f"Warning: Could not load fonts ({e}). Using default.")
    INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE + 2)
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)
    BUTTON_FONT = pygame.font.Font(None, BUTTON_FONT_SIZE + 2)


# --- Chess Configuration ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"
ANALYSIS_TIME_LIMIT = 0.4 # For background analysis when creating nodes
BEST_MOVE_ANALYSIS_TIME = 0.8 # Longer analysis for button press
NUM_BEST_MOVES_TO_SHOW = 5 # How many moves to cycle through
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = 15000

# --- Asset Loading ---
PIECE_IMAGE_PATH = "pieces"
PIECE_IMAGES = {} # This holds the SQ_SIZE x SQ_SIZE images
def load_piece_images(path=PIECE_IMAGE_PATH, sq_size=SQ_SIZE):
    # (Code unchanged)
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

# --- Move Quality Classification ---
def classify_move_quality(points_lost):
    """Classifies a move based on the expected points lost."""
    if points_lost <= 0.00: # Allows for eval gains or no loss
        return "Best"
    elif points_lost <= 0.02:
        return "Excellent"
    elif points_lost <= 0.05:
        return "Good"
    elif points_lost <= 0.10:
        return "Inaccuracy"
    elif points_lost <= 0.20:
        return "Mistake"
    else: # points_lost > 0.20
        return "Blunder"

# --- Game History Tree Node ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen
        self.move = move
        self.parent = parent
        self.children = []
        self.raw_score = raw_score
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT) # Calculate immediately
        self._san_cache = None
        self.x = 0
        self.y = 0
        self.screen_rect = None
        self.move_quality = None # Will be calculated after node creation if possible

    def add_child(self, child_node): self.children.append(child_node)
    def get_ply(self):
        count = 0; node = self
        while node.parent: count += 1; node = node.parent
        return count
    def get_san(self, board_at_parent):
        # (Code unchanged)
        if self._san_cache is not None: return self._san_cache
        if not self.move or not self.parent: return "root"
        try:

            san = board_at_parent.san(self.move)
            self._san_cache = san; return san
        except Exception as e: return self.move.uci()

    def calculate_and_set_move_quality(self):
        """Calculates and sets the move_quality based on parent's and own evaluation."""
        if not self.parent or self.parent.raw_score is None or self.raw_score is None:
            self.move_quality = None # Cannot determine quality without both scores
            return

        # Use the already calculated percentages if available, otherwise calculate them
        wp_before = self.parent.white_percentage
        wp_after = self.white_percentage

        # Handle potential None values if calculation failed earlier (treat as 50%)
        wp_before = 50.0 if wp_before is None else wp_before
        wp_after = 50.0 if wp_after is None else wp_after

        # Determine whose turn it was *before* this node's move was made
        # This requires the FEN of the parent node
        try:

            parent_board = chess.Board(self.parent.fen, chess960=True)
            turn_before_move = parent_board.turn
        except:

             # If parent FEN is invalid, cannot determine quality
             self.move_quality = None
             return

        eval_before_pov = 0.0
        eval_after_pov = 0.0

        # Calculate evaluation from the perspective of the player who moved
        if turn_before_move == chess.WHITE: # White made the move leading to this node
            eval_before_pov = wp_before / 100.0
            eval_after_pov = wp_after / 100.0
        else: # Black made the move leading to this node
            eval_before_pov = (100.0 - wp_before) / 100.0
            eval_after_pov = (100.0 - wp_after) / 100.0

        # Points lost is the drop in evaluation for the player who moved
        # Ensure it's non-negative. A gain results in 0 points lost.
        points_lost = max(0.0, eval_before_pov - eval_after_pov)

        self.move_quality = classify_move_quality(points_lost)


# --- Helper Functions (Chess Logic) ---
def get_castling_rights_str(board):
    # (Code unchanged)
    rights = []
    if board.has_kingside_castling_rights(chess.WHITE): rights.append("K")
    if board.has_queenside_castling_rights(chess.WHITE): rights.append("Q")
    if board.has_kingside_castling_rights(chess.BLACK): rights.append("k")
    if board.has_queenside_castling_rights(chess.BLACK): rights.append("q")
    return "".join(rights) if rights else "-"

def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    # (Code unchanged)
    if score is None: return None
    pov_score = score.white()
    if pov_score.is_mate():
        mate_val = pov_score.mate(); return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    # Ensure score returns a value, handle None case
    cp_val = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE)
    if cp_val is None: return None # Return None if score is None
    clamped_cp = max(-clamp_limit, min(clamp_limit, cp_val)); normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
    return normalized * 100.0

def format_score(score, turn):
    # (Code unchanged)
    if score is None: return "N/A"
    pov_score = score.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate (N/A)"
    else:
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        return f"{cp / 100.0:+.2f}" if cp is not None else "N/A"

def get_engine_analysis(board, engine, time_limit):
    # (Code unchanged)
    if not engine: return None, "Analysis unavailable (Engine not loaded).", None
    try:

        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_move = info.get("pv", [None])[0]
        score = info.get("score")

        if score is None and not info.get("pv"):
            return None, "Engine analysis failed (no score/pv).", None

        score_str = format_score(score, board.turn)

        if best_move:
            try:
                best_move_san = board.san(best_move)
            except Exception:
                best_move_san = best_move.uci()
            return best_move_san, f"Score: {score_str}", score
        elif score is not None:
             return None, f"Position Score: {score_str} (no pv)", score
        else:
            return None, "Engine could not provide suggestion.", None

    except chess.engine.EngineTerminatedError: print("Engine terminated unexpectedly."); return None, "Engine terminated.", None
    except Exception as e: print(f"Analysis error: {e}"); return None, f"Analysis error: {e}", None


def get_top_engine_moves(board, engine, time_limit, num_moves):
    # (Code unchanged)
    if not engine:
        return [], "Engine not available."
    if board.is_game_over():
         return [], "Game is over."

    moves_info = []
    try:

        # Use the analysis context manager to handle MultiPV results
        with engine.analysis(board, chess.engine.Limit(time=time_limit), multipv=num_moves) as analysis:
            for info in analysis:
                if "pv" in info and info["pv"]:
                    move = info["pv"][0]
                    score = info.get("score")
                    score_str = format_score(score, board.turn)
                    try:

                        move_san = board.san(move)
                    except Exception:

                        move_san = move.uci()
                    moves_info.append({
                        "move": move,
                        "san": move_san,
                        "score_str": score_str,
                        "score_obj": score
                    })
                    if len(moves_info) >= num_moves:
                        break
                elif "score" in info and not moves_info:
                    pass # Ignore score-only lines for now

        if not moves_info:
            return [], "Engine returned no moves."
        return moves_info, None

    except chess.engine.EngineTerminatedError:

        print("Engine terminated unexpectedly.")
        return [], "Engine terminated."
    except Exception as e:

        print(f"MultiPV Analysis error: {e}")
        return [], f"Analysis error: {e}"


def get_path_to_node(node):
    # (Code unchanged)
    path = []; current = node
    while current: path.append(current); current = current.parent
    return path[::-1]

# --- Drawing Functions ---
def draw_board(surface): # (Code unchanged)
    for r in range(8):
        for c in range(8): color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR; pygame.draw.rect(surface, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(surface, board, piece_images, dragging_piece_info): # (Code unchanged)
    if piece_images is None: return
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if dragging_piece_info and square == dragging_piece_info['square']: continue
            piece_symbol = piece.symbol(); piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper(); img = piece_images.get(piece_key)
            if img: rank = chess.square_rank(square); file = chess.square_file(square); screen_x = file * SQ_SIZE; screen_y = (7 - rank) * SQ_SIZE; surface.blit(img, (screen_x, screen_y))
    if dragging_piece_info and dragging_piece_info['img']: img_rect = dragging_piece_info['img'].get_rect(center=dragging_piece_info['pos']); surface.blit(dragging_piece_info['img'], img_rect)

def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight): # (Code unchanged)
    """Draws all types of highlights: last move, selected, legal, and engine suggestion."""
    if engine_move_to_highlight:
        from_sq = engine_move_to_highlight.from_square
        to_sq = engine_move_to_highlight.to_square
        for sq in [from_sq, to_sq]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(ENGINE_MOVE_HIGHLIGHT_COLOR)
            surface.blit(s, highlight_rect.topleft)
    if last_move:
        from_sq = last_move.from_square; to_sq = last_move.to_square
        for sq in [from_sq, to_sq]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(LAST_MOVE_HIGHLIGHT_COLOR)
            surface.blit(s, highlight_rect.topleft)
    if selected_square is not None:
        rank = chess.square_rank(selected_square); file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT_COLOR)
        surface.blit(s, highlight_rect.topleft)
    if legal_moves_for_selected:
        for move in legal_moves_for_selected:
            dest_sq = move.to_square; rank = chess.square_rank(dest_sq); file = chess.square_file(dest_sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            is_capture = board.is_capture(move)
            center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2
            radius = SQ_SIZE // 6
            if is_capture:
                 pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius + 2, 2)
            else:
                 pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius)
            surface.blit(s, (file * SQ_SIZE, (7 - rank) * SQ_SIZE))

def draw_eval_bar(surface, white_percentage): # (Code unchanged)
    bar_x = BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2; bar_y = 0; bar_height = BOARD_SIZE; white_percentage = 50.0 if white_percentage is None else white_percentage
    white_height = int(bar_height * (white_percentage / 100.0)); black_height = bar_height - white_height
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height)); pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height)); pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

def create_eval_plot_surface(node_path, plot_width_px, plot_height_px): # (Code unchanged)
    if not node_path or len(node_path) < 2: return None
    plies = [node.get_ply() for node in node_path]; percentages = [node.white_percentage if node.white_percentage is not None else 50.0 for node in node_path]
    dark_grey_mpl=tuple(c/255.0 for c in DARK_GREY); orange_mpl=tuple(c/255.0 for c in ORANGE); grey_mpl=tuple(c/255.0 for c in GREY); light_grey_mpl=(211/255.,)*3; white_mpl=(1.,)*3
    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80); fig.patch.set_facecolor(dark_grey_mpl); ax.set_facecolor(dark_grey_mpl)
    if len(plies) > 1: ax.fill_between(plies, percentages, color=white_mpl, alpha=0.9, step='post'); ax.plot(plies, percentages, color=white_mpl, marker='.', markersize=3, linestyle='-')
    elif len(plies)==1: ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=5)
    ax.axhline(50, color=orange_mpl, linestyle='-', linewidth=1.5); ax.set_xlabel("Ply", color=light_grey_mpl); ax.set_ylabel("White Win %", color=light_grey_mpl)
    ax.set_title("Win Probability History (Current Line)", color=white_mpl, fontsize=10); ax.set_xlim(0, max(plies) if plies else 1); ax.set_ylim(0, 100)
    ax.tick_params(axis='x', colors=light_grey_mpl, labelsize=8); ax.tick_params(axis='y', colors=light_grey_mpl, labelsize=8); ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl)
    ax.spines['top'].set_color(grey_mpl); ax.spines['bottom'].set_color(grey_mpl); ax.spines['left'].set_color(grey_mpl); ax.spines['right'].set_color(grey_mpl)
    plt.tight_layout(pad=0.5); buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none'); plt.close(fig); buf.seek(0)
    try:
        plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e: print(f"Error loading plot image: {e}"); plot_surface = None
    finally: buf.close()
    return plot_surface

def draw_game_info(surface, board, analysis_text, message, font, current_node, button_rect, button_text): # (Code unchanged)
    info_rect = pygame.Rect(0, BOARD_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    surface.fill(DARK_GREY, info_rect)
    start_y = BOARD_SIZE + 5
    line_height = font.get_height() + 2
    max_lines_before_button = (button_rect.top - start_y) // line_height -1
    lines = []
    turn = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"Turn: {turn}")
    if board.is_check(): lines.append("!! CHECK !!")
    last_move_san = "Start"
    if current_node.move and current_node.parent:
        parent_board = chess.Board(current_node.parent.fen, chess960=True)
        last_move_san = current_node.get_san(parent_board)
    lines.append(f"Last Move: {last_move_san}")
    ply = current_node.get_ply()
    move_num = (ply + 1) // 2 + (ply % 2)
    lines.append(f"Ply: {ply} | Move: {move_num}")
    lines.append(f"Castling: {get_castling_rights_str(board)}")
    if analysis_text: lines.append(analysis_text)
    if message: lines.append(f"Msg: {message}")
    if board.is_game_over(claim_draw=True):
        lines.append("-" * 10)
        res = board.outcome(claim_draw=True)
        term = res.termination.name.replace('_',' ').title() if res else "N/A"
        lines.append(f"GAME OVER: {res.result() if res else board.result(claim_draw=True)} ({term})")
    for i, line in enumerate(lines):
        if i >= max_lines_before_button: break
        text_surface = font.render(line, True, WHITE)
        surface.blit(text_surface, (5, start_y + i * line_height))
    pygame.draw.rect(surface, BUTTON_COLOR, button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, button_rect, 1, border_radius=3)
    btn_text_surf = BUTTON_FONT.render(button_text, True, BUTTON_TEXT_COLOR)
    btn_text_rect = btn_text_surf.get_rect(center=button_rect.center)
    surface.blit(btn_text_surf, btn_text_rect)

# --- Horizontal Game Tree Drawing ---
TREE_PIECE_SIZE = 20
NODE_DIAMETER = TREE_PIECE_SIZE
HORIZ_SPACING = 45 + TREE_PIECE_SIZE
VERT_SPACING = 5 + TREE_PIECE_SIZE
TEXT_OFFSET_X = 3
INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 2
INITIAL_TREE_SURFACE_HEIGHT = SCREEN_HEIGHT * 2
drawn_tree_nodes = {}
tree_scroll_x = 0
tree_scroll_y = 0
max_drawn_tree_x = 0
max_drawn_tree_y = 0
tree_render_surface = None
temp_san_board = chess.Board(chess960=True)
scaled_tree_piece_images = {}

def get_scaled_tree_image(piece_key, target_size): # (Code unchanged)
    """Gets or creates a scaled version of a piece image for the tree."""
    global PIECE_IMAGES, scaled_tree_piece_images
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images:
        return scaled_tree_piece_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key)
    if original_img:
        try:

            scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size))
            scaled_tree_piece_images[cache_key] = scaled_img
            return scaled_img
        except Exception as e: return None
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node):
    """
    Calculates layout, draws piece images, move quality badges, and connects nodes.
    Updates max_drawn_tree_x and max_drawn_tree_y.
    """
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board

    node.x = x
    node.y = y_center # Simple vertical placement for now

    # --- Determine what to draw (Piece or Root Circle) ---
    piece_img = None
    piece_key = None
    is_root = not node.parent

    if not is_root:
        # Determine the piece that moved TO this node
        try:

            parent_board = chess.Board(node.parent.fen, chess960=True) # Board state *before* the move
            moved_piece = parent_board.piece_at(node.move.from_square)
            if moved_piece:
                piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
                piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE) # Use cached scaler
        except:

            # Handle case where parent FEN might be invalid
            moved_piece = None


    # --- Calculate positions of children first --- (Code unchanged)
    child_y_positions = []; child_heights = []; total_child_height_estimate = 0; child_x = x + HORIZ_SPACING
    if node.children:
        total_child_height_estimate = (len(node.children) - 1) * VERT_SPACING
        current_child_y = y_center - total_child_height_estimate / 2
        for child in node.children:
            child_actual_y, child_subtree_height = layout_and_draw_tree_recursive(surface, child, child_x, current_child_y, level + 1, font, current_node)
            child_y_positions.append(child_actual_y); child_heights.append(child_subtree_height)
            # Adjust next child's Y based on the actual height used by the previous, ensuring minimum spacing
            current_child_y += max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2) # Might need refinement

    # --- Draw the Node (Piece or Root Circle) ---
    node_rect = None
    if piece_img:
        img_rect = piece_img.get_rect(center=(int(node.x), int(node.y)))
        surface.blit(piece_img, img_rect.topleft)
        node_rect = img_rect # Use image rect for highlighting and text placement
        max_drawn_tree_x = max(max_drawn_tree_x, img_rect.right)
    elif is_root:
        radius = TREE_PIECE_SIZE // 2
        pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
        max_drawn_tree_x = max(max_drawn_tree_x, node.x + radius)
    else:
        # Fallback if piece image failed
        node_rect = pygame.Rect(node.x-2, node.y-2, 4, 4) # Minimal rect
        max_drawn_tree_x = max(max_drawn_tree_x, node.x + 2)

    # --- Highlight Current Node --- (Code unchanged)
    if node == current_node and node_rect:
        pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(3, 3), 1)

    # --- Draw the move quality badge --- ADDED ---
    if node_rect and node.move_quality and node.move_quality in MOVE_QUALITY_COLORS:
        badge_color = MOVE_QUALITY_COLORS[node.move_quality]
        # Position badge relative to the node_rect (e.g., bottom-right)
        # Use node_rect center as reference to be consistent with piece image placement
        badge_center_x = node_rect.centerx + TREE_PIECE_SIZE // 2 - BADGE_RADIUS - 1
        badge_center_y = node_rect.centery + TREE_PIECE_SIZE // 2 - BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), BADGE_RADIUS)
        pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), BADGE_RADIUS, 1) # Outline

    # --- Draw the move text (SAN) --- (Code unchanged)
    move_text = ""; text_rect = None
    if node.parent:
        try:

            # Re-create parent board state for SAN generation
            parent_board = chess.Board(node.parent.fen, chess960=True)
            move_text = node.get_san(parent_board) # Use cached SAN
        except:

             move_text = node.move.uci() if node.move else "?" # Fallback

    if move_text and node_rect:
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
        text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery))
        surface.blit(text_surf, text_rect)
        max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)

    # --- Store the clickable area --- (Code unchanged)
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect:
        clickable_rect.width = text_rect.right - clickable_rect.left
    node.screen_rect = clickable_rect
    drawn_tree_nodes[node] = node.screen_rect

    # --- Draw lines to children --- (Code unchanged)
    if node_rect:
        for i, child in enumerate(node.children):
            if child in drawn_tree_nodes and hasattr(child, 'x') and hasattr(child, 'y'):
               child_rect = drawn_tree_nodes[child]
               start_pos = (node_rect.right, node_rect.centery)
               child_visual_rect = child.screen_rect if child.screen_rect else pygame.Rect(child.x-1, child.y-1,2,2)
               end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)

    # --- Return Y position and height --- (Code unchanged)
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE
    if child_y_positions:
        approx_child_height = max(child_heights) if child_heights else VERT_SPACING
        min_child_y = min(child_y_positions) - approx_child_height / 2
        max_child_y = max(child_y_positions) + approx_child_height / 2
        my_height = max(my_height, max_child_y - min_child_y)

    max_drawn_tree_y = max(max_drawn_tree_y, node.y + my_height // 2)

    return node.y, my_height

def draw_game_tree(surface, root_node, current_node, font): # (Code unchanged)
    """Draws the scrollable game history tree visualization."""
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface
    drawn_tree_nodes.clear(); max_drawn_tree_x = 0; max_drawn_tree_y = 0
    tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
    required_width = INITIAL_TREE_SURFACE_WIDTH; required_height = INITIAL_TREE_SURFACE_HEIGHT
    if tree_render_surface is None or tree_render_surface.get_width() < required_width or tree_render_surface.get_height() < required_height: tree_render_surface = pygame.Surface((required_width, required_height))
    tree_render_surface.fill(TREE_BG_COLOR)
    if not root_node: surface.blit(tree_render_surface, tree_panel_rect.topleft); return
    start_x = 15 + TREE_PIECE_SIZE // 2; start_y = tree_render_surface.get_height() // 2
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node)
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING; total_tree_height = max_drawn_tree_y + VERT_SPACING
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width); max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x)); tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))

    # Adjust autoscroll logic if needed to center current node, or just ensure visibility
    if current_node.x > tree_scroll_x + tree_panel_rect.width - HORIZ_SPACING: # Scroll right if node near edge
        tree_scroll_x = current_node.x - tree_panel_rect.width + HORIZ_SPACING
    if current_node.x < tree_scroll_x + HORIZ_SPACING: # Scroll left if node near edge
         tree_scroll_x = current_node.x - HORIZ_SPACING

    if current_node.y > tree_scroll_y + tree_panel_rect.height - VERT_SPACING: # Scroll down
        tree_scroll_y = current_node.y - tree_panel_rect.height + VERT_SPACING
    if current_node.y < tree_scroll_y + VERT_SPACING: # Scroll up
         tree_scroll_y = current_node.y - VERT_SPACING

    # Clamp again after autoscroll adjustments
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x))
    tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))

    source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, tree_panel_rect.width, tree_panel_rect.height)
    surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width; scrollbar_width = tree_panel_rect.width * ratio_visible; scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio; scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - 5, scrollbar_width, 5)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=2)
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height; scrollbar_height = tree_panel_rect.height * ratio_visible; scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0
        scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio; scrollbar_rect = pygame.Rect(tree_panel_rect.right - 5, scrollbar_y, 5, scrollbar_height)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=2)

# --- Coordinate Conversion --- (Same)
def screen_to_square(pos):
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE: return None
    file = x // SQ_SIZE; rank = 7 - (y // SQ_SIZE)
    return chess.square(file, rank)

# --- Main Game Function ---
def play_chess960_pygame():
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y

    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: print("Exiting due to missing piece images."); return

    # --- Button Setup (Identical) ---
    button_width = 140
    button_height = 25
    button_margin = 5
    best_move_button_rect = pygame.Rect(
        SCREEN_WIDTH - button_width - button_margin - SIDE_PANEL_WIDTH,
        BOARD_SIZE + INFO_PANEL_HEIGHT - button_height - button_margin,
        button_width,
        button_height
    )
    best_moves_cache = {}
    current_best_move_index = -1
    highlighted_engine_move = None

    # --- (Rest of the setup is mostly identical) ---
    board = chess.Board(chess960=True); engine = None; message = None; start_pos_num = -1
    game_root = None; current_node = None; last_raw_score = None
    one_off_analysis_text = None; meter_visible = True; plot_surface = None
    needs_redraw = True

    # --- Setup Loop (Identical) ---
    while True: # Get Start Pos
        print("\n--- Chess960 Setup ---")
        pos_choice = input("Enter Chess960 pos (0-959), 'random', or blank: ").strip().lower()
        if not pos_choice or pos_choice == 'random': start_pos_num = random.randint(0, 959); break
        else:
            try:
                start_pos_num = int(pos_choice);
                if 0 <= start_pos_num <= 959: break
                else: print("Number must be 0-959.")
            except ValueError: print("Invalid input.")
    board.set_chess960_pos(start_pos_num); start_fen = board.fen()
    print(f"Starting Chess960 Position {start_pos_num} (FEN: {start_fen})"); print("Initializing Pygame window..."); time.sleep(0.5)
    try: # Init Engine/Root
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH); print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
            try:
                engine.configure({"UCI_Chess960": True})
            except Exception: pass # Ignore if option not supported
            # Get initial score for the root node
            _, _, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5)
            game_root = GameNode(fen=start_fen, raw_score=initial_score)
            current_node = game_root
            last_raw_score = initial_score # Store the raw score object
            print("Initial analysis done.")
        else: print(f"Warning: Stockfish not found. Analysis disabled."); message="Engine unavailable"; game_root=GameNode(fen=start_fen); current_node=game_root
    except Exception as e: print(f"Error initializing Stockfish: {e}."); engine=None; message="Engine init failed"; game_root=GameNode(fen=start_fen); current_node=game_root

    running = True; clock = pygame.time.Clock(); selected_square = None; dragging_piece_info = None
    legal_moves_for_selected = []; last_move_displayed = None; tree_scroll_speed = 30

    # --- Main Game Loop ---
    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

        def reset_transient_state(): # (Identical)
            nonlocal selected_square, dragging_piece_info, legal_moves_for_selected
            nonlocal one_off_analysis_text, needs_redraw
            nonlocal current_best_move_index, highlighted_engine_move
            selected_square = None
            dragging_piece_info = None
            legal_moves_for_selected = []
            one_off_analysis_text = None
            needs_redraw = True
            current_best_move_index = -1
            highlighted_engine_move = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL: # Scroll Tree (Identical)
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    tree_scroll_y -= event.y * tree_scroll_speed
                    tree_scroll_x = max(0, tree_scroll_x) # Clamp horizontal just in case
                    tree_scroll_y = max(0, tree_scroll_y)
                    needs_redraw = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # --- Best Move Button Click (Identical) ---
                if event.button == 1 and best_move_button_rect.collidepoint(pos) and engine and not board.is_game_over():
                    current_fen = current_node.fen
                    moves_list = []
                    if current_fen in best_moves_cache:
                        moves_list = best_moves_cache[current_fen]
                        message = "Using cached analysis."
                    else:
                        message = "Analyzing top moves..."
                        needs_redraw = True
                        # Quick redraw to show message
                        screen.fill(DARK_GREY); draw_board(screen)
                        draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
                        draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
                        draw_eval_bar(screen, current_node.white_percentage)
                        btn_text = "Analyzing..."
                        draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node, best_move_button_rect, btn_text)
                        if meter_visible:
                             path=get_path_to_node(current_node); plot_surface=create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                             if plot_surface: screen.blit(plot_surface, (0,BOARD_SIZE+INFO_PANEL_HEIGHT))
                        draw_game_tree(screen,game_root,current_node,TREE_FONT)
                        pygame.display.flip()
                        # Perform analysis
                        moves_list, error_msg = get_top_engine_moves(board, engine, BEST_MOVE_ANALYSIS_TIME, NUM_BEST_MOVES_TO_SHOW)
                        if error_msg: message = error_msg; moves_list = []
                        else: message = f"Found {len(moves_list)} move(s)."; best_moves_cache[current_fen] = moves_list
                        current_best_move_index = -1 # Reset index after new analysis

                    if moves_list:
                        current_best_move_index = (current_best_move_index + 1) % len(moves_list)
                        highlighted_engine_move = moves_list[current_best_move_index]["move"]
                        move_info = moves_list[current_best_move_index]
                        one_off_analysis_text = f"Best {current_best_move_index+1}/{len(moves_list)}: {move_info['san']} ({move_info['score_str']})"
                    else:
                        current_best_move_index = -1; highlighted_engine_move = None
                        if not message: message = "No engine moves found."
                    needs_redraw = True

                # --- Tree Click (Identical) ---
                elif tree_panel_rect.collidepoint(pos):
                    relative_panel_x = pos[0]-tree_panel_rect.left; relative_panel_y = pos[1]-tree_panel_rect.top
                    absolute_tree_click_x = relative_panel_x + tree_scroll_x; absolute_tree_click_y = relative_panel_y + tree_scroll_y
                    clicked_node = None
                    # Check drawn nodes (values are Rects)
                    for node, rect in drawn_tree_nodes.items():
                        # Ensure rect is valid before checking collision
                        if rect and rect.collidepoint(absolute_tree_click_x, absolute_tree_click_y):
                            clicked_node = node
                            break # Found the node
                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                        message = f"Jumped to ply {current_node.get_ply()}"
                        reset_transient_state()

                # --- Board Click (Identical) ---
                elif pos[0] < BOARD_SIZE and pos[1] < BOARD_SIZE:
                    if event.button == 1:
                        highlighted_engine_move = None; current_best_move_index = -1
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn:
                                selected_square = sq; piece_symbol = piece.symbol(); piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                                dragging_piece_info = {'square': sq, 'piece': piece, 'img': PIECE_IMAGES.get(piece_key), 'pos': pos}
                                legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]; needs_redraw = True
                            else: selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []; needs_redraw = True
                        else:
                             selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []; needs_redraw = True


            elif event.type == pygame.MOUSEBUTTONUP: # Drop Piece
                 if event.button == 1 and dragging_piece_info:
                    pos = event.pos; to_sq = screen_to_square(pos); from_sq = dragging_piece_info['square']
                    move_made = False
                    if to_sq is not None and from_sq != to_sq:
                        promotion = None; piece = dragging_piece_info['piece']
                        # Auto-promote to Queen for simplicity in this example
                        if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                            promotion = chess.QUEEN
                        move = chess.Move(from_sq, to_sq, promotion=promotion)

                        if move in board.legal_moves:
                            move_made = True
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move: existing_child = child; break

                            if existing_child: # Navigate to existing move
                                current_node = existing_child; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score; message = "Navigated to existing move."
                            else: # --- Create new node ---
                                parent_node = current_node # Keep reference to parent
                                board.push(move); new_fen = board.fen(); board.pop() # Get fen after move, revert board state

                                # Analyze the position *after* the move
                                temp_board_for_analysis = chess.Board(new_fen, chess960=True)
                                new_raw_score = None
                                analysis_failed = False
                                if engine:
                                    # Get score for the new position
                                    _, _, new_raw_score = get_engine_analysis(temp_board_for_analysis, engine, ANALYSIS_TIME_LIMIT)
                                    analysis_failed = new_raw_score is None

                                # Create the new node with FEN, move, parent, and score
                                new_node = GameNode(fen=new_fen, move=move, parent=parent_node, raw_score=new_raw_score)

                                # --- Calculate move quality ---
                                new_node.calculate_and_set_move_quality() # Uses parent and own score

                                # Add child and update current node
                                parent_node.add_child(new_node)
                                current_node = new_node
                                board.set_fen(current_node.fen) # Set board to the new position
                                last_raw_score = current_node.raw_score # Update last raw score

                                if analysis_failed:
                                    message = "Analysis failed for new move."
                                elif new_node.move_quality:
                                     message = f"Move quality: {new_node.move_quality}" # Display quality
                                else:
                                     message = None # No message if analysis ok but quality couldn't be calculated


                            reset_transient_state() # Resets selection, highlights etc.

                        else: message = f"Illegal move"; needs_redraw = True # Keep msg short

                    # Reset drag state regardless of move legality/completion
                    dragging_piece_info = None; selected_square = None; legal_moves_for_selected = []
                    if not move_made: # If no move was made (e.g., dropped back on same square)
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True # Need redraw to remove dragging piece image


            elif event.type == pygame.MOUSEMOTION: # Drag Piece (Identical)
                if dragging_piece_info: dragging_piece_info['pos'] = event.pos; needs_redraw = True

            elif event.type == pygame.KEYDOWN: # Keyboard Commands (Identical logic for navigation)
                node_changed = False
                if event.key == pygame.K_LEFT:
                    if current_node.parent: current_node = current_node.parent; node_changed = True; message = f"Back"
                    else: message = "At start"
                elif event.key == pygame.K_RIGHT:
                    if current_node.children: current_node = current_node.children[0]; node_changed = True; message = f"Forward (main)"
                    else: message = "End of line"

                if node_changed:
                    board.set_fen(current_node.fen)
                    last_raw_score = current_node.raw_score
                    reset_transient_state() # Reset highlights, best move index etc.

                # Other key presses (Analyze, Meter toggle, Escape) - (Identical)
                elif event.key == pygame.K_a:
                     if engine:
                         message = "Analyzing..."; needs_redraw = True;
                         # Quick redraw
                         screen.fill(DARK_GREY); draw_board(screen);
                         draw_highlights(screen,board,selected_square,legal_moves_for_selected,current_node.move if current_node.parent else None, highlighted_engine_move)
                         draw_pieces(screen,board,PIECE_IMAGES,dragging_piece_info); draw_eval_bar(screen,current_node.white_percentage);
                         btn_text = "Analyze Best" if highlighted_engine_move is None else f"Best {current_best_move_index+1}/{len(best_moves_cache.get(current_node.fen,[]))}"
                         draw_game_info(screen,board,"Analyzing...",message,INFO_FONT,current_node, best_move_button_rect, btn_text);
                         if meter_visible: path=get_path_to_node(current_node); plot_surface=create_eval_plot_surface(path,BOARD_SIZE,PLOT_PANEL_HEIGHT);
                         if plot_surface: screen.blit(plot_surface, (0,BOARD_SIZE+INFO_PANEL_HEIGHT))
                         draw_game_tree(screen,game_root,current_node,TREE_FONT);
                         pygame.display.flip()
                         # Run longer analysis
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 3)
                         # Update current node's score and percentage if analysis is successful
                         if raw_score:
                             current_node.raw_score = raw_score
                             current_node.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                             last_raw_score = raw_score
                             # Re-calculate quality for THIS node based on updated parent score (if applicable)
                             # Note: This recalculates the quality of the move *leading* to the current node
                             # It does NOT update the quality of moves *originating* from this node.
                             current_node.calculate_and_set_move_quality()

                         one_off_analysis_text = f"Suggests: {best_move_san} ({score_str})" if best_move_san else score_str if score_str else "Analysis complete."
                         message = None; needs_redraw = True
                         highlighted_engine_move = None; current_best_move_index = -1 # Clear best move display after 'a'
                     else: message = "No engine"
                elif event.key == pygame.K_m: meter_visible = not meter_visible; message = f"Meter {'ON' if meter_visible else 'OFF'}"; needs_redraw = True
                elif event.key == pygame.K_ESCAPE: running = False

        # --- Update state / Drawing --- (Mostly Identical)
        last_move_displayed = current_node.move if current_node.parent else None
        if board.fen() != current_node.fen:
             board.set_fen(current_node.fen) # Ensure board object is synced

        if needs_redraw:
            screen.fill(DARK_GREY)
            draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
            draw_eval_bar(screen, current_node.white_percentage)

            # Determine button text dynamically (Identical)
            button_text = "Show Best Move"
            current_fen = current_node.fen
            cached_moves = best_moves_cache.get(current_fen)
            if highlighted_engine_move and cached_moves:
                button_text = f"Show Next ({current_best_move_index + 2}/{len(cached_moves)})"
                if current_best_move_index == len(cached_moves) - 1:
                    button_text = f"Hide ({len(cached_moves)}/{len(cached_moves)})"
            elif engine and not board.is_game_over():
                 button_text = "Show Best (1)"

            draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node, best_move_button_rect, button_text)

            if meter_visible:
                path = get_path_to_node(current_node)
                plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                if plot_surface:
                    plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT))
                    screen.blit(plot_surface, plot_rect)

            draw_game_tree(screen, game_root, current_node, TREE_FONT) # Draws tree with badges now
            pygame.display.flip()
            needs_redraw = False

        clock.tick(30)

    # --- Cleanup --- (Identical)
    print("\nExiting Pygame...")
    if engine:
        try:
            engine.quit()
            print("Stockfish engine closed.")
        except Exception as e: print(f"Error closing engine: {e}")
    plt.close('all'); pygame.quit(); sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()