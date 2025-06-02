# ... (Keep all imports and setup the same) ...
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

# --- Pygame & Display Configuration --- (Same)
pygame.init()
pygame.font.init()
BOARD_SIZE = 512
SQ_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = 80
PLOT_PANEL_HEIGHT = 120
TREE_PANEL_HEIGHT = 180
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20
SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960 Analysis Board with Piece History Tree")

# --- Colors --- (Same)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
DARK_GREY = (60, 60, 60)
LIGHT_SQ_COLOR = (238, 238, 210)
DARK_SQ_COLOR = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
ORANGE = (255, 165, 0)
TREE_BG_COLOR = DARK_GREY
TREE_LINE_COLOR = (150, 150, 150)
# TREE_NODE_WHITE_MOVE = (230, 230, 230) # No longer needed directly for node fill
# TREE_NODE_BLACK_MOVE = (50, 50, 50)   # No longer needed directly for node fill
TREE_NODE_ROOT_COLOR = (100, 100, 150) # Keep color for root circle
TREE_NODE_CURRENT_OUTLINE = (255, 100, 100)
TREE_TEXT_COLOR = (200, 200, 200)


# --- Font --- (Same)
INFO_FONT_SIZE = 16
TREE_FONT_SIZE = 12
try:
    INFO_FONT = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
except Exception as e:
    print(f"Warning: Could not load fonts ({e}). Using default.")
    INFO_FONT = pygame.font.Font(None, INFO_FONT_SIZE + 2)
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)


# --- Chess Configuration --- (Same)
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"
ANALYSIS_TIME_LIMIT = 0.4
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = 15000


# --- Asset Loading --- (Same)
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


# --- Game History Tree Node --- (Same)
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen
        self.move = move
        self.parent = parent
        self.children = []
        self.raw_score = raw_score
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT) if raw_score else None
        self._san_cache = None
        self.x = 0
        self.y = 0
        self.screen_rect = None

    def add_child(self, child_node): self.children.append(child_node)
    def get_ply(self):
        count = 0; node = self
        while node.parent: count += 1; node = node.parent
        return count
    def get_san(self, board_at_parent):
        # (Code unchanged) - Caching is important here
        if self._san_cache is not None: return self._san_cache
        if not self.move or not self.parent: return "root"
        try:
            # Use the provided board object directly
            san = board_at_parent.san(self.move)
            self._san_cache = san; return san
        except Exception as e: return self.move.uci() # Fallback
    # get_player_color_who_moved not strictly needed anymore for node color


# --- Helper Functions (Chess Logic) --- (Same)
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
    cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE)
    if cp is None: return None
    clamped_cp = max(-clamp_limit, min(clamp_limit, cp)); normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
    return normalized * 100.0
def get_engine_analysis(board, engine, time_limit):
    # (Code unchanged)
    if not engine: return None, "Analysis unavailable (Engine not loaded).", None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_move = info.get("pv", [None])[0]; score = info.get("score")
        if score is None and not info.get("pv"): return None, "Engine analysis failed (no score/pv).", None
        if best_move and score is not None:
            pov_score = score.pov(board.turn); score_str = ""
            if pov_score.is_mate(): mate_in = pov_score.mate(); score_str = f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate (N/A)"
            else: cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2); score_str = f"Score: {cp / 100.0:+.2f}" if cp is not None else "Score: N/A"
            try: best_move_san = board.san(best_move)
            except Exception: best_move_san = best_move.uci()
            return best_move_san, score_str, score
        elif score is not None:
             pov_score = score.pov(board.turn); score_str = f"Score: {pov_score.score() / 100.0:+.2f}" if not pov_score.is_mate() else f"Mate {pov_score.mate()}"
             return None, f"Position Score: {score_str} (no pv)", score
        else: return None, "Engine could not provide suggestion.", None
    except chess.engine.EngineTerminatedError: print("Engine terminated unexpectedly."); return None, "Engine terminated.", None
    except Exception as e: print(f"Analysis error: {e}"); return None, f"Analysis error: {e}", None
def get_path_to_node(node):
    # (Code unchanged)
    path = []; current = node
    while current: path.append(current); current = current.parent
    return path[::-1]


# --- Drawing Functions --- (Board, Pieces, Highlights, Eval Bar, Plot - unchanged)
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
def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move): # (Code unchanged)
    if last_move:
        from_sq = last_move.from_square; to_sq = last_move.to_square
        for sq in [from_sq, to_sq]: rank = chess.square_rank(sq); file = chess.square_file(sq); highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE); s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill((186, 202, 68, 180)); surface.blit(s, highlight_rect.topleft)
    if selected_square is not None: rank = chess.square_rank(selected_square); file = chess.square_file(selected_square); highlight_rect = pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE); s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill(HIGHLIGHT_COLOR); surface.blit(s, highlight_rect.topleft)
    if legal_moves_for_selected:
        for move in legal_moves_for_selected: dest_sq = move.to_square; rank = chess.square_rank(dest_sq); file = chess.square_file(dest_sq); s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6); surface.blit(s, (file * SQ_SIZE, (7 - rank) * SQ_SIZE))
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
    try: plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e: print(f"Error loading plot image: {e}"); plot_surface = None
    finally: buf.close()
    return plot_surface
def draw_game_info(surface, board, analysis_text, message, font, current_node): # (Code unchanged)
    start_y = BOARD_SIZE + 5; line_height = font.get_height() + 2; ply = current_node.get_ply(); move_num = (ply + 1) // 2 + (ply % 2); lines = []
    turn = "White" if board.turn == chess.WHITE else "Black"; lines.append(f"Turn: {turn}"); last_move_san = "Start"
    if board.is_check(): lines.append("!! CHECK !!")
    if current_node.move and current_node.parent: parent_board = chess.Board(current_node.parent.fen, chess960=True); last_move_san = current_node.get_san(parent_board)
    lines.append(f"Last Move: {last_move_san}"); lines.append(f"Ply: {ply} | Move: {move_num}"); lines.append(f"Castling: {get_castling_rights_str(board)}")
    if analysis_text: lines.append("-" * 10); lines.append(analysis_text)
    if message: lines.append("-" * 10); lines.append(f"Msg: {message}")
    if board.is_game_over(claim_draw=True): lines.append("-" * 10); res = board.outcome(claim_draw=True); term = res.termination.name.replace('_',' ').title() if res else "N/A"; lines.append(f"GAME OVER: {res.result() if res else board.result(claim_draw=True)} ({term})")
    info_rect = pygame.Rect(0, BOARD_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT); surface.fill(DARK_GREY, info_rect)
    for i, line in enumerate(lines):
        if i * line_height > INFO_PANEL_HEIGHT - line_height: break
        text_surface = font.render(line, True, WHITE); surface.blit(text_surface, (5, start_y + i * line_height))


# --- Horizontal Game Tree Drawing ---
# NODE_RADIUS = 5 # No longer primary visual element
TREE_PIECE_SIZE = 20 # Size of piece images in the tree
NODE_DIAMETER = TREE_PIECE_SIZE # Use piece size for spacing calculations now
HORIZ_SPACING = 45 + TREE_PIECE_SIZE # Adjust spacing based on piece size
VERT_SPACING = 5 + TREE_PIECE_SIZE
TEXT_OFFSET_X = 3 # Small horizontal gap between piece and text
# TEXT_OFFSET_Y = -TREE_FONT_SIZE // 2 # Keep vertical offset for text centering relative to node.y

INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 2
drawn_tree_nodes = {} # Maps node object to its Rect on tree_render_surface
tree_scroll_x = 0
max_drawn_tree_x = 0
tree_render_surface = None
temp_san_board = chess.Board(chess960=True) # Reusable board object

# --- NEW: Cache for scaled tree images ---
scaled_tree_piece_images = {}

def get_scaled_tree_image(piece_key, target_size):
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
        except Exception as e:
            # print(f"Warn: Could not scale image {piece_key}: {e}") # Avoid spamming logs
            return None # Indicate failure
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node):
    """
    Calculates layout, draws piece images instead of circles, updates max_drawn_tree_x.
    Returns the actual y position used and the total vertical height of the drawn subtree.
    """
    global drawn_tree_nodes, max_drawn_tree_x, PIECE_IMAGES, temp_san_board

    node.x = x
    node.y = y_center # Simple vertical placement for now

    # --- Determine what to draw (Piece or Root Circle) ---
    piece_img = None
    piece_key = None
    is_root = not node.parent

    if not is_root:
        # Determine the piece that moved TO this node
        parent_board = chess.Board(node.parent.fen, chess960=True) # Board state *before* the move
        moved_piece = parent_board.piece_at(node.move.from_square)
        if moved_piece:
            piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
            piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE) # Use cached scaler

    # --- Calculate positions of children first (needed for vertical spacing) ---
    child_y_positions = []
    child_heights = []
    total_child_height_estimate = 0
    child_x = x + HORIZ_SPACING

    if node.children:
        total_child_height_estimate = (len(node.children) - 1) * VERT_SPACING
        current_child_y = y_center - total_child_height_estimate / 2

        for child in node.children:
            child_actual_y, child_subtree_height = layout_and_draw_tree_recursive(
                surface, child, child_x, current_child_y, level + 1, font, current_node
            )
            child_y_positions.append(child_actual_y)
            child_heights.append(child_subtree_height)
            current_child_y += max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2)


    # --- Draw the Node (Piece or Root Circle) ---
    node_rect = None
    if piece_img:
        img_rect = piece_img.get_rect(center=(int(node.x), int(node.y)))
        surface.blit(piece_img, img_rect.topleft)
        node_rect = img_rect # Use image rect for highlighting and text placement
        max_drawn_tree_x = max(max_drawn_tree_x, img_rect.right) # Update max extent
    elif is_root:
        # Draw root circle as before
        radius = TREE_PIECE_SIZE // 2 # Make root circle similar size to pieces
        pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
        max_drawn_tree_x = max(max_drawn_tree_x, node.x + radius)
    else:
        # Fallback: Draw a small dot if piece image failed? Or nothing.
        # Let's draw nothing for now if piece image is missing/failed.
        node_rect = pygame.Rect(node.x-2, node.y-2, 4, 4) # Minimal rect if no image
        max_drawn_tree_x = max(max_drawn_tree_x, node.x + 2)

    # --- Highlight Current Node ---
    if node == current_node and node_rect:
        pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(3, 3), 1) # Outline the rect

    # --- Draw the move text (SAN) ---
    move_text = ""
    text_rect = None
    if node.parent:
        # Re-create parent board state for SAN generation (less efficient but needed if not passed)
        # If performance becomes an issue, consider passing parent_board down the recursion
        parent_board = chess.Board(node.parent.fen, chess960=True)
        move_text = node.get_san(parent_board) # Use cached SAN
    # else: # No text for root, or maybe "Start"? Let's omit for root.
    #    move_text = "Start"

    if move_text and node_rect: # Only draw text if node was drawn
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
        # Position text to the right of the node_rect
        text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery))
        surface.blit(text_surf, text_rect)
        max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right) # Update max extent based on text


    # --- Store the clickable area ---
    # Make it cover the node visual and the text
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect:
        clickable_rect.width = text_rect.right - clickable_rect.left # Extend width to include text

    node.screen_rect = clickable_rect # Store rect relative to tree_render_surface
    drawn_tree_nodes[node] = node.screen_rect


    # --- Draw lines to children ---
    if node_rect: # Only draw lines *from* a visible node
        for i, child in enumerate(node.children):
            # Check if child was drawn and has coordinates
            if child in drawn_tree_nodes and hasattr(child, 'x') and hasattr(child, 'y'):
               child_rect = drawn_tree_nodes[child] # Get child's rect
               # Draw line from parent's right-center to child's left-center
               start_pos = (node_rect.right, node_rect.centery)
               # Use child's node_rect if available, otherwise its x,y center
               child_visual_rect = child.screen_rect if child.screen_rect else pygame.Rect(child.x-1, child.y-1,2,2)
               end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)


    # Return actual Y position and estimated height (same logic as before)
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE
    if child_y_positions:
        approx_child_height = max(child_heights) if child_heights else VERT_SPACING
        min_child_y = min(child_y_positions) - approx_child_height / 2
        max_child_y = max(child_y_positions) + approx_child_height / 2
        my_height = max(my_height, max_child_y - min_child_y)

    return node.y, my_height


def draw_game_tree(surface, root_node, current_node, font):
    """Draws the scrollable game history tree visualization with piece images."""
    global drawn_tree_nodes, tree_scroll_x, max_drawn_tree_x, tree_render_surface

    drawn_tree_nodes.clear()
    max_drawn_tree_x = 0 # Reset max extent for this draw call

    tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

    # --- Ensure tree_render_surface exists ---
    required_width = INITIAL_TREE_SURFACE_WIDTH # Still using fixed size for simplicity
    if tree_render_surface is None or tree_render_surface.get_width() < required_width or tree_render_surface.get_height() != tree_panel_rect.height:
         # print(f"Creating/Resizing tree surface to {required_width}x{tree_panel_rect.height}")
         tree_render_surface = pygame.Surface((required_width, tree_panel_rect.height))

    tree_render_surface.fill(TREE_BG_COLOR) # Fill background

    if not root_node:
        surface.blit(tree_render_surface, tree_panel_rect.topleft) # Blit empty background
        return

    start_x = 15 + TREE_PIECE_SIZE // 2 # Adjust start based on node size
    start_y = tree_render_surface.get_height() // 2

    # Recursive layout and drawing onto tree_render_surface
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node)

    # --- Scrolling Logic ---
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING # Add padding
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width)
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x)) # Clamp scroll_x

    # --- Blit the visible portion onto the main screen ---
    source_rect = pygame.Rect(tree_scroll_x, 0, tree_panel_rect.width, tree_panel_rect.height)
    surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)

    # --- Draw Scroll Indicator (Optional) --- (Same as before)
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width
        scrollbar_width = tree_panel_rect.width * ratio_visible
        scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio
        scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - 5, scrollbar_width, 5)
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
    global tree_scroll_x

    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: print("Exiting due to missing piece images."); return
    # Pre-scale images? Optional - for better performance, could call get_scaled_tree_image here for all pieces.

    # --- (Rest of the setup is identical to the previous version) ---
    board = chess.Board(chess960=True); engine = None; message = None; start_pos_num = -1
    game_root = None; current_node = None; last_raw_score = None
    one_off_analysis_text = None; meter_visible = True; plot_surface = None
    needs_redraw = True
    while True: # Get Start Pos
        print("\n--- Chess960 Setup ---")
        pos_choice = input("Enter Chess960 pos (0-959), 'random', or blank: ").strip().lower()
        if not pos_choice or pos_choice == 'random': start_pos_num = random.randint(0, 959); break
        else:
            try: 
                start_pos_num = int(pos_choice)
                if 0 <= start_pos_num <= 959: 
                    break
                else: print("Number must be 0-959.")
            except ValueError: print("Invalid input.")
    board.set_chess960_pos(start_pos_num); start_fen = board.fen()
    print(f"Starting Chess960 Position {start_pos_num} (FEN: {start_fen})"); print("Initializing Pygame window..."); time.sleep(0.5)
    try: # Init Engine/Root
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH); print(f"Stockfish engine loaded: {STOCKFISH_PATH}")
            try: engine.configure({"UCI_Chess960": True})
            except Exception: pass
            _, _, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5)
            game_root = GameNode(fen=start_fen, raw_score=initial_score); current_node = game_root; last_raw_score = initial_score; print("Initial analysis done.")
        else: print(f"Warning: Stockfish not found. Analysis disabled."); message="Engine unavailable"; game_root=GameNode(fen=start_fen); current_node=game_root
    except Exception as e: print(f"Error initializing Stockfish: {e}."); engine=None; message="Engine init failed"; game_root=GameNode(fen=start_fen); current_node=game_root

    running = True; clock = pygame.time.Clock(); selected_square = None; dragging_piece_info = None
    legal_moves_for_selected = []; last_move_displayed = None; tree_scroll_speed = 30

    # --- Main Game Loop --- (Event handling logic is identical to previous version)
    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        tree_panel_rect = pygame.Rect(0, BOARD_SIZE + INFO_PANEL_HEIGHT + PLOT_PANEL_HEIGHT, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL: # Scroll Tree
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    tree_scroll_x -= event.y * tree_scroll_speed; tree_scroll_x = max(0, tree_scroll_x); needs_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if tree_panel_rect.collidepoint(pos): # Click Tree
                    relative_panel_x = pos[0]-tree_panel_rect.left; relative_panel_y = pos[1]-tree_panel_rect.top
                    absolute_tree_click_x = relative_panel_x + tree_scroll_x; absolute_tree_click_y = relative_panel_y
                    clicked_node = None
                    # Sort nodes by x descending? Might make click check slightly faster if many overlaps
                    for node, rect in drawn_tree_nodes.items(): # Check click against absolute coords
                        if rect and rect.collidepoint(absolute_tree_click_x, absolute_tree_click_y): clicked_node = node; break
                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                        message = f"Jumped to ply {current_node.get_ply()}"; one_off_analysis_text = None
                        selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
                        needs_redraw = True
                elif pos[0] < BOARD_SIZE and pos[1] < BOARD_SIZE: # Click Board
                    if event.button == 1:
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn:
                                selected_square = sq; piece_symbol = piece.symbol(); piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                                dragging_piece_info = {'square': sq, 'piece': piece, 'img': PIECE_IMAGES.get(piece_key), 'pos': pos}
                                legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]; needs_redraw = True
                            else: selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []; needs_redraw = True
            elif event.type == pygame.MOUSEBUTTONUP: # Drop Piece
                 if event.button == 1 and dragging_piece_info:
                    pos = event.pos; to_sq = screen_to_square(pos); from_sq = dragging_piece_info['square']
                    if to_sq is not None and from_sq != to_sq:
                        promotion = None; piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]: promotion = chess.QUEEN
                        move = chess.Move(from_sq, to_sq, promotion=promotion)
                        if move in board.legal_moves:
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move: existing_child = child; break
                            if existing_child: # Navigate
                                current_node = existing_child; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score; message = "Navigated to existing move."
                            else: # Create new node
                                board.push(move); new_fen = board.fen(); board.pop()
                                temp_board_for_analysis = chess.Board(new_fen, chess960=True); new_raw_score = None; analysis_failed = False
                                if engine: _, _, new_raw_score = get_engine_analysis(temp_board_for_analysis, engine, ANALYSIS_TIME_LIMIT); analysis_failed = new_raw_score is None
                                new_node = GameNode(fen=new_fen, move=move, parent=current_node, raw_score=new_raw_score)
                                current_node.add_child(new_node); current_node = new_node
                                board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                                message = "Analysis failed for new move." if analysis_failed else None
                            one_off_analysis_text = None; needs_redraw = True
                        else: message = f"Illegal move"; needs_redraw = True # Keep msg short
                    dragging_piece_info = None; selected_square = None; legal_moves_for_selected = []; needs_redraw = True
            elif event.type == pygame.MOUSEMOTION: # Drag Piece
                if dragging_piece_info: dragging_piece_info['pos'] = event.pos; needs_redraw = True
            elif event.type == pygame.KEYDOWN: # Keyboard Commands
                if event.key == pygame.K_LEFT:
                    if current_node.parent: current_node = current_node.parent; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score; message = f"Back"; one_off_analysis_text = None; selected_square=None; dragging_piece_info=None; legal_moves_for_selected=[]; needs_redraw = True
                    else: message = "At start"
                elif event.key == pygame.K_RIGHT:
                    if current_node.children: current_node = current_node.children[0]; board.set_fen(current_node.fen); last_raw_score = current_node.raw_score; message = f"Forward"; one_off_analysis_text = None; selected_square=None; dragging_piece_info=None; legal_moves_for_selected=[]; needs_redraw = True
                    else: message = "End of line"
                elif event.key == pygame.K_a: # Analyze
                     if engine:
                         message = "Analyzing..."; needs_redraw = True; screen.fill(DARK_GREY); draw_board(screen); draw_highlights(screen,board,selected_square,legal_moves_for_selected,current_node.move if current_node.parent else None); draw_pieces(screen,board,PIECE_IMAGES,dragging_piece_info); draw_eval_bar(screen,current_node.white_percentage);
                         if meter_visible: path=get_path_to_node(current_node); plot_surface=create_eval_plot_surface(path,BOARD_SIZE,PLOT_PANEL_HEIGHT);
                         if plot_surface: screen.blit(plot_surface, (0,BOARD_SIZE+INFO_PANEL_HEIGHT))
                         draw_game_tree(screen,game_root,current_node,TREE_FONT); draw_game_info(screen,board,one_off_analysis_text,message,INFO_FONT,current_node); pygame.display.flip() # Redraw
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 3); current_node.raw_score = raw_score; new_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                         if new_percentage is not None and new_percentage != current_node.white_percentage: current_node.white_percentage = new_percentage
                         last_raw_score = raw_score; one_off_analysis_text = f"Suggests: {best_move_san} ({score_str})" if best_move_san else score_str if score_str else "Analysis complete."; message = None; needs_redraw = True
                     else: message = "No engine"
                elif event.key == pygame.K_m: meter_visible = not meter_visible; message = f"Meter {'ON' if meter_visible else 'OFF'}"; needs_redraw = True
                elif event.key == pygame.K_ESCAPE: running = False

        # --- Update state / Drawing --- (Identical to previous version)
        last_move_displayed = current_node.move if current_node.parent else None
        board.set_fen(current_node.fen)
        if needs_redraw:
            screen.fill(DARK_GREY); draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
            draw_eval_bar(screen, current_node.white_percentage)
            draw_game_info(screen, board, one_off_analysis_text, message, INFO_FONT, current_node)
            if meter_visible:
                path = get_path_to_node(current_node); plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                if plot_surface: plot_rect = plot_surface.get_rect(topleft=(0, BOARD_SIZE + INFO_PANEL_HEIGHT)); screen.blit(plot_surface, plot_rect)
            draw_game_tree(screen, game_root, current_node, TREE_FONT) # Call the updated tree drawing
            pygame.display.flip()
            needs_redraw = False
        clock.tick(30) # Limit FPS

    # --- Cleanup --- (Identical)
    print("\nExiting Pygame...");
    if engine:
        try: engine.quit(); print("Stockfish engine closed.")
        except Exception: print("Error closing engine.")
    plt.close('all'); pygame.quit(); sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()