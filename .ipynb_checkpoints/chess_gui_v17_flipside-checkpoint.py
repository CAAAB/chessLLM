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
import argparse
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

pygame.init()
pygame.font.init()

SETUP_PANEL_HEIGHT = 40
COORD_PADDING = 20

BOARD_SIZE = 700
SQ_SIZE = BOARD_SIZE // 8

PLOT_PANEL_HEIGHT = 100
TREE_PANEL_HEIGHT = 250
EVAL_BAR_WIDTH_PX = 20
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 5

SCREEN_WIDTH = BOARD_SIZE + SIDE_PANEL_WIDTH + 2 * COORD_PADDING
SCREEN_HEIGHT = SETUP_PANEL_HEIGHT + BOARD_SIZE + PLOT_PANEL_HEIGHT + TREE_PANEL_HEIGHT + 2 * COORD_PADDING
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess960")

BOARD_X = COORD_PADDING
BOARD_Y = SETUP_PANEL_HEIGHT + COORD_PADDING

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
TREE_NODE_CURRENT_OUTLINE = (255, 255, 255)
TREE_TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (80, 80, 100)
BUTTON_TEXT_COLOR = WHITE
COORD_COLOR = (180, 180, 180)

TREE_BADGE_RADIUS = 4
TREE_BADGE_BEST_COLOR = (0, 180, 0)
TREE_BADGE_EXCELLENT_COLOR = (50, 205, 50)
TREE_BADGE_GOOD_COLOR = (0, 100, 0)
TREE_BADGE_INACCURACY_COLOR = (240, 230, 140)
TREE_BADGE_MISTAKE_COLOR = (255, 140, 0)
TREE_BADGE_BLUNDER_COLOR = (200, 0, 0)
TREE_MOVE_QUALITY_COLORS = { "Best": TREE_BADGE_BEST_COLOR, "Excellent": TREE_BADGE_EXCELLENT_COLOR, "Good": TREE_BADGE_GOOD_COLOR, "Inaccuracy": TREE_BADGE_INACCURACY_COLOR, "Mistake": TREE_BADGE_MISTAKE_COLOR, "Blunder": TREE_BADGE_BLUNDER_COLOR }

TREE_PIECE_SIZE = 18; NODE_DIAMETER = TREE_PIECE_SIZE
HORIZ_SPACING = 40 + TREE_PIECE_SIZE; VERT_SPACING = 2 + TREE_PIECE_SIZE
TEXT_OFFSET_X = 4; TEXT_OFFSET_Y = TREE_PIECE_SIZE//2; INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 3
INITIAL_TREE_SURFACE_HEIGHT = TREE_PANEL_HEIGHT * 5

BOARD_BADGE_RADIUS = 14
BOARD_BADGE_OUTLINE_COLOR = DARK_GREY
BOARD_BADGE_IMAGE_SIZE = (22, 22)
BOARD_BADGE_OFFSET_X = SQ_SIZE - BOARD_BADGE_RADIUS - 2
BOARD_BADGE_OFFSET_Y = BOARD_BADGE_RADIUS + 2

def load_and_process_badge_image(path, target_size, target_color=WHITE):
    try:
        img = pygame.image.load(path).convert_alpha()
        for x in range(img.get_width()):
            for y in range(img.get_height()):
                color = img.get_at((x, y))
                if color[3] > 100 and color[0] < 100 and color[1] < 100 and color[2] < 100:
                     img.set_at((x, y), (*target_color, color[3]))
        img = pygame.transform.smoothscale(img, target_size)
        return img
    except pygame.error as e: print(f"Error processing badge image '{path}': {e}"); return None
    except Exception as e: print(f"Unexpected error processing badge image '{path}': {e}"); return None

BADGE_TYPES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
BADGE_IMAGES = {}
all_badges_loaded = True
print("Loading and processing badge images...")
badge_dir = "badges"
if not os.path.isdir(badge_dir):
    print(f"Error: Badge directory '{badge_dir}' not found. Badges disabled.")
    all_badges_loaded = False
else:
    for quality in BADGE_TYPES:
        file_path = os.path.join(badge_dir, f"{quality.lower()}.png")
        if not os.path.exists(file_path):
            print(f"  - Missing: {quality} ({file_path})")
            all_badges_loaded = False
            continue
        processed_img = load_and_process_badge_image(file_path, BOARD_BADGE_IMAGE_SIZE, WHITE)
        if processed_img: BADGE_IMAGES[quality] = processed_img; print(f"  - Loaded: {quality}")
        else: print(f"  - FAILED processing: {quality}"); all_badges_loaded = False
if not all_badges_loaded: print("Warning: Some badge images failed to load or process.")

TREE_FONT_SIZE = 12
BUTTON_FONT_SIZE = 14
COORD_FONT_SIZE = 12
try:
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
    BUTTON_FONT = pygame.font.SysFont("sans", BUTTON_FONT_SIZE)
    COORD_FONT = pygame.font.SysFont("sans", COORD_FONT_SIZE)
except Exception as e:
    print(f"Warning: Could not load specific fonts ({e}). Using default.")
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)
    BUTTON_FONT = pygame.font.Font(None, BUTTON_FONT_SIZE + 2)
    COORD_FONT = pygame.font.Font(None, COORD_FONT_SIZE + 2)

STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"
ANALYSIS_TIME_LIMIT = .5
BEST_MOVE_ANALYSIS_TIME = 5
NUM_BEST_MOVES_TO_SHOW = 5
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = EVAL_CLAMP_LIMIT * 1.5

PIECE_IMAGE_PATH = "pieces"
PIECE_IMAGES = {}
def load_piece_images(path=PIECE_IMAGE_PATH, sq_size=SQ_SIZE):
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    loaded_images = {}
    if not os.path.isdir(path): print(f"Error: Piece image directory not found: '{path}'"); return None
    all_loaded = True
    for piece in pieces:
        file_path = os.path.join(path, f"{piece}.png")
        if not os.path.exists(file_path):
            print(f"Error: Piece image file not found: '{file_path}'")
            all_loaded = False
            continue
        try:
            img = pygame.image.load(file_path).convert_alpha()
            img = pygame.transform.smoothscale(img, (sq_size, sq_size))
            loaded_images[piece] = img
        except pygame.error as e: print(f"Error loading piece image '{file_path}': {e}"); all_loaded = False
    if not all_loaded: print("Please ensure all 12 piece PNG files exist in the 'pieces' directory."); return None
    print(f"Loaded {len(loaded_images)} piece images.")
    return loaded_images

def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    if score is None: return 50.0
    pov_score = score.white()
    if pov_score.is_mate():
        mate_val = pov_score.mate()
        return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    else:
        cp_val = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        if cp_val is None: return 50.0
        clamped_cp = max(-clamp_limit, min(clamp_limit, cp_val))
        normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
        return normalized * 100.0

def classify_move_quality(white_percentage_before, white_percentage_after, turn_before_move):
    if turn_before_move == chess.WHITE:
        eval_before_pov = white_percentage_before / 100.0
        eval_after_pov = white_percentage_after / 100.0
    else:
        eval_before_pov = (100.0 - white_percentage_before) / 100.0
        eval_after_pov = (100.0 - white_percentage_after) / 100.0
    eval_drop = max(0.0, eval_before_pov - eval_after_pov)
    eval_drop_percent = eval_drop * 100
    if eval_drop_percent <= 2: return "Best"
    elif eval_drop_percent <= 5: return "Excellent"
    elif eval_drop_percent <= 10: return "Good"
    elif eval_drop_percent <= 20: return "Inaccuracy"
    elif eval_drop_percent <= 35: return "Mistake"
    else: return "Blunder"

class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen; self.move = move; self.parent = parent; self.children = []
        self.raw_score = raw_score;
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
        self._san_cache = None; self.x = 0; self.y = 0; self.screen_rect = None
        self.move_quality = None

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
        except Exception as e: return self.move.uci()

    def calculate_and_set_move_quality(self):
        if not self.parent or self.parent.white_percentage is None or self.white_percentage is None:
            self.move_quality = None; return
        wp_before = self.parent.white_percentage
        wp_after = self.white_percentage
        try:
            parent_board = chess.Board(self.parent.fen, chess960=True)
            turn_before_move = parent_board.turn
        except ValueError:
             self.move_quality = None
             print(f"Warning: Could not create board from parent FEN for quality check: {self.parent.fen}")
             return
        self.move_quality = classify_move_quality(wp_before, wp_after, turn_before_move)

def format_score(score, turn):
    if score is None: return "N/A"
    pov_score = score.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate"
    else:
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        return f"{cp / 100.0:+.2f}" if cp is not None else "N/A (No CP)"

def get_engine_analysis(board, engine, time_limit):
    if not engine: return None, "Engine unavailable.", None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit), info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
    except (chess.engine.EngineError, BrokenPipeError, AttributeError) as e:
        print(f"Engine analysis error: {e}")
        return None, f"Engine error.", None
    except Exception as e:
         print(f"Unexpected analysis error: {e}")
         return None, f"Analysis error.", None
    best_move = info.get("pv", [None])[0]
    score = info.get("score")
    if score is None:
         if best_move: return board.san(best_move), "Score N/A", None
         else: return None, "Analysis failed (no score/pv).", None
    score_str = format_score(score, board.turn)
    if best_move:
        try: best_move_san = board.san(best_move)
        except Exception: best_move_san = best_move.uci()
        return best_move_san, f"Score: {score_str}", score
    else: return None, f"Pos Score: {score_str}", score

def get_top_engine_moves(board, engine, time_limit, num_moves):
    if not engine: return [], "Engine not available."
    if board.is_game_over(): return [], "Game is over."
    moves_info = []
    try:
        with engine.analysis(board, chess.engine.Limit(time=time_limit), multipv=num_moves) as analysis:
            for info in analysis:
                if "pv" in info and info["pv"] and "score" in info:
                    move = info["pv"][0]
                    score = info.get("score")
                    score_str = format_score(score, board.turn)
                    try: move_san = board.san(move)
                    except Exception: move_san = move.uci()
                    moves_info.append({"move": move, "san": move_san, "score_str": score_str, "score_obj": score})
                    if len(moves_info) >= num_moves: break
        return moves_info, None
    except (chess.engine.EngineError, BrokenPipeError, AttributeError) as e:
        print(f"MultiPV Analysis engine error: {e}")
        return [], f"Engine error: {e}"
    except Exception as e:
        print(f"MultiPV Analysis unexpected error: {e}")
        return [], f"Analysis error: {e}"

def get_path_to_node(node):
    path = []; current = node;
    while current: path.append(current); current = current.parent;
    return path[::-1]

def draw_board(surface):
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (BOARD_X + c * SQ_SIZE, BOARD_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_coordinates(surface, font):
    for i in range(8):
        text = chr(ord('a') + i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y + BOARD_SIZE + COORD_PADDING // 2))
        surface.blit(text_surf, text_rect)
    for i in range(8):
        text = str(8 - i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X - COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2))
        surface.blit(text_surf, text_rect)

def draw_pieces(surface, board, piece_images, dragging_piece_info):
    if piece_images is None: return
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if dragging_piece_info and square == dragging_piece_info['square']: continue
            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
            img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square); file = chess.square_file(square)
                screen_x = BOARD_X + file * SQ_SIZE
                screen_y = BOARD_Y + (7 - rank) * SQ_SIZE
                surface.blit(img, (screen_x, screen_y))
    if dragging_piece_info and dragging_piece_info['img']:
        img = dragging_piece_info['img']
        img_rect = img.get_rect(center=dragging_piece_info['pos'])
        surface.blit(img, img_rect)

def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight):
    def highlight_squares(squares, color):
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(color)
        for sq in squares:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            surface.blit(s, highlight_rect.topleft)

    def highlight_legal_moves(moves, color):
        for move in moves:
            dest_sq = move.to_square
            rank = chess.square_rank(dest_sq)
            file = chess.square_file(dest_sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            is_capture = board.is_capture(move) or board.is_en_passant(move)
            center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2
            radius = SQ_SIZE // 6
            if is_capture:
                pygame.draw.circle(s, color, (center_x, center_y), radius + 3, 3)
            else:
                pygame.draw.circle(s, color, (center_x, center_y), radius)
            surface.blit(s, (BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE))

    if engine_move_to_highlight:
        highlight_squares([engine_move_to_highlight.from_square, engine_move_to_highlight.to_square], ENGINE_MOVE_HIGHLIGHT_COLOR)

    if last_move:
        highlight_squares([last_move.from_square, last_move.to_square], LAST_MOVE_HIGHLIGHT_COLOR)

    if selected_square is not None:
        highlight_squares([selected_square], HIGHLIGHT_COLOR)

    if legal_moves_for_selected:
        highlight_legal_moves(legal_moves_for_selected, POSSIBLE_MOVE_COLOR)


def draw_board_badge(surface, square, quality):
    if not all_badges_loaded: return
    badge_image = BADGE_IMAGES.get(quality); badge_color = TREE_MOVE_QUALITY_COLORS.get(quality)
    if quality is None or badge_image is None or badge_color is None: return
    rank = chess.square_rank(square); file = chess.square_file(square)
    square_base_x = BOARD_X + file * SQ_SIZE; square_base_y = BOARD_Y + (7 - rank) * SQ_SIZE
    center_x = square_base_x + BOARD_BADGE_OFFSET_X; center_y = square_base_y + BOARD_BADGE_OFFSET_Y
    pygame.draw.circle(surface, badge_color, (center_x, center_y), BOARD_BADGE_RADIUS)
    pygame.draw.circle(surface, BOARD_BADGE_OUTLINE_COLOR, (center_x, center_y), BOARD_BADGE_RADIUS, 1)
    badge_rect = badge_image.get_rect(center=(center_x, center_y))
    surface.blit(badge_image, badge_rect.topleft)

def draw_eval_bar(surface, white_percentage):
    bar_height = BOARD_SIZE
    bar_x = BOARD_X + BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = BOARD_Y
    white_percentage = 50.0 if white_percentage is None else white_percentage
    white_height = int(bar_height * (white_percentage / 100.0)); black_height = bar_height - white_height
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    if not node_path or len(node_path) < 1:
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    plies = [node.get_ply() for node in node_path]
    percentages = [(node.white_percentage if node.white_percentage is not None else 50.0) for node in node_path]
    dark_grey_mpl = tuple(c/255.0 for c in DARK_GREY); orange_mpl = tuple(c/255.0 for c in ORANGE)
    grey_mpl = tuple(c/255.0 for c in GREY); white_mpl = tuple(c/255.0 for c in WHITE)
    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl); ax.set_facecolor(dark_grey_mpl)
    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=0.6)
        ax.plot(plies, percentages, color=white_mpl, marker=None, linestyle='-', linewidth=1.5)
    elif len(plies) == 1: ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=3)
    ax.axhline(50, color=orange_mpl, linestyle='--', linewidth=1)
    ax.set_xlim(0, max(max(plies), 1) if plies else 1); ax.set_ylim(0, 100)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(""); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl, alpha=0.4)
    for spine in ax.spines.values(): spine.set_color(grey_mpl); spine.set_linewidth(0.5)
    plt.tight_layout(pad=0.1); buf = io.BytesIO()
    try: plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, bbox_inches='tight', pad_inches=0.05)
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig); buf.close()
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally: plt.close(fig)
    buf.seek(0); plot_surface = None
    try: plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e:
        print(f"Error loading plot image from buffer: {e}")
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally: buf.close()
    return plot_surface

drawn_tree_nodes = {}; tree_scroll_x = 0; tree_scroll_y = 0
max_drawn_tree_x = 0; max_drawn_tree_y = 0; tree_render_surface = None
temp_san_board = chess.Board(chess960=True); scaled_tree_piece_images = {}

def get_scaled_tree_image(piece_key, target_size):
    global PIECE_IMAGES, scaled_tree_piece_images
    if PIECE_IMAGES is None: return None
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images: return scaled_tree_piece_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key)
    if original_img:
        try:
            scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size))
            scaled_tree_piece_images[cache_key] = scaled_img
            return scaled_img
        except Exception as e: print(f"Error scaling tree image for {piece_key}: {e}"); return None
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node, helpers_visible):
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board
    node.x = x; node.y = y_center
    piece_img = None; is_root = not node.parent
    if node.move and node.parent:
        try:
            temp_san_board.set_fen(node.parent.fen)
            moved_piece = temp_san_board.piece_at(node.move.from_square)
            if moved_piece:
                piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
                piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE)
        except ValueError: print(f"Warning: Invalid parent FEN '{node.parent.fen}' for tree node piece.")
        except Exception as e: print(f"Warning: Error getting piece for tree node: {e}")
    child_y_positions = []; child_subtree_heights = []; total_child_height_estimate = 0
    child_x = x + HORIZ_SPACING
    if node.children:
        num_children = len(node.children); total_child_height_estimate = (num_children -1) * VERT_SPACING
        current_child_y_start = y_center - total_child_height_estimate / 2
        next_child_y = current_child_y_start
        for i, child in enumerate(node.children):
            child_center_y, child_subtree_height = layout_and_draw_tree_recursive(
                surface, child, child_x, next_child_y, level + 1, font, current_node, helpers_visible)
            child_y_positions.append(child_center_y); child_subtree_heights.append(child_subtree_height)
            spacing = max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2 if child_subtree_height > 0 else VERT_SPACING)
            next_child_y = child_center_y + child_subtree_height / 2 + spacing / 2
    node_rect = None
    if piece_img:
        img_rect = piece_img.get_rect(center=(int(node.x), int(node.y)))
        surface.blit(piece_img, img_rect.topleft); node_rect = img_rect
    elif is_root:
        radius = TREE_PIECE_SIZE // 2
        pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    else:
        radius = 3; pygame.draw.circle(surface, GREY, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    if node_rect: max_drawn_tree_x = max(max_drawn_tree_x, node_rect.right)
    if node == current_node and node_rect: pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(4, 4), 1)
    if node_rect and node.move_quality and node.move_quality in TREE_MOVE_QUALITY_COLORS and helpers_visible:
        badge_color = TREE_MOVE_QUALITY_COLORS[node.move_quality]
        badge_center_x = node_rect.right - TREE_BADGE_RADIUS - 1; badge_center_y = node_rect.bottom - TREE_BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS)
        pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS, 1)
    move_text = ""; text_rect = None
    if node.parent:
        try: temp_san_board.set_fen(node.parent.fen); move_text = node.get_san(temp_san_board)
        except ValueError: move_text = node.move.uci() + "?"
        except Exception: move_text = node.move.uci()
    if move_text and node_rect:
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
        text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery + TEXT_OFFSET_Y))
        surface.blit(text_surf, text_rect); max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect: clickable_rect.width = max(clickable_rect.width, text_rect.right - clickable_rect.left)
    node.screen_rect = clickable_rect; drawn_tree_nodes[node] = node.screen_rect
    if node_rect and node.children:
        for i, child in enumerate(node.children):
            if hasattr(child, 'x') and hasattr(child, 'y'):
               child_visual_rect = drawn_tree_nodes.get(child, pygame.Rect(child.x-1, child.y-1,2,2))
               start_pos = (node_rect.right, node_rect.centery); end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE; subtree_total_height = 0
    if child_y_positions:
        min_child_y_center = min(child_y_positions); max_child_y_center = max(child_y_positions)
        est_top = min_child_y_center - (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        est_bottom = max_child_y_center + (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        subtree_total_height = max(0, est_bottom - est_top)
    node_bottom_extent = node.y + max(my_height, subtree_total_height) / 2
    max_drawn_tree_y = max(max_drawn_tree_y, node_bottom_extent)
    return node.y, max(my_height, subtree_total_height)

def draw_game_tree(surface, root_node, current_node, font, helpers_visible):
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface
    drawn_tree_nodes.clear(); max_drawn_tree_x = 0; max_drawn_tree_y = 0
    plot_panel_y = BOARD_Y + BOARD_SIZE; tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
    tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
    estimated_required_width = max(INITIAL_TREE_SURFACE_WIDTH, int(max_drawn_tree_x + 2 * HORIZ_SPACING))
    estimated_required_height = max(INITIAL_TREE_SURFACE_HEIGHT, int(max_drawn_tree_y + 2 * VERT_SPACING))
    if tree_render_surface is None or tree_render_surface.get_width() < estimated_required_width or tree_render_surface.get_height() < estimated_required_height:
         try:
             new_width = max(estimated_required_width, tree_panel_rect.width)
             new_height = max(estimated_required_height, tree_panel_rect.height)
             tree_render_surface = pygame.Surface((new_width, new_height))
         except pygame.error as e:
             print(f"Error creating/resizing tree surface ({new_width}x{new_height}): {e}")
             pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect); pygame.draw.rect(surface, GREY, tree_panel_rect, 1)
             error_surf = font.render("Tree Error", True, ORANGE); surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
             return
    tree_render_surface.fill(TREE_BG_COLOR)
    if not root_node:
        surface.blit(tree_render_surface, tree_panel_rect.topleft, area=pygame.Rect(0,0,tree_panel_rect.width, tree_panel_rect.height))
        pygame.draw.rect(surface, GREY, tree_panel_rect, 1); return
    start_x = 15 + TREE_PIECE_SIZE // 2; start_y = tree_render_surface.get_height() // 2
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node, helpers_visible)
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING; total_tree_height = max_drawn_tree_y + VERT_SPACING
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width); max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)
    scroll_margin_x = HORIZ_SPACING * 1.5; scroll_margin_y = VERT_SPACING * 3
    if current_node and current_node.screen_rect:
        node_rect_on_surface = current_node.screen_rect
        if node_rect_on_surface.right > tree_scroll_x + tree_panel_rect.width - scroll_margin_x: tree_scroll_x = node_rect_on_surface.right - tree_panel_rect.width + scroll_margin_x
        if node_rect_on_surface.left < tree_scroll_x + scroll_margin_x: tree_scroll_x = node_rect_on_surface.left - scroll_margin_x
        if node_rect_on_surface.bottom > tree_scroll_y + tree_panel_rect.height - scroll_margin_y: tree_scroll_y = node_rect_on_surface.bottom - tree_panel_rect.height + scroll_margin_y
        if node_rect_on_surface.top < tree_scroll_y + scroll_margin_y: tree_scroll_y = node_rect_on_surface.top - scroll_margin_y
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x)); tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))
    source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, tree_panel_rect.width, tree_panel_rect.height)
    try: surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
    except pygame.error as e:
        print(f"Error blitting tree view: {e}")
        pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect); pygame.draw.rect(surface, GREY, tree_panel_rect, 1)
        error_surf = font.render("Tree Blit Error", True, ORANGE); surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
    scrollbar_thickness = 7
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width; scrollbar_width = max(15, tree_panel_rect.width * ratio_visible)
        scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio
        scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - scrollbar_thickness - 1, scrollbar_width, scrollbar_thickness)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height; scrollbar_height = max(15, tree_panel_rect.height * ratio_visible)
        scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0
        scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio
        scrollbar_rect = pygame.Rect(tree_panel_rect.right - scrollbar_thickness - 1, scrollbar_y, scrollbar_thickness, scrollbar_height)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)

def screen_to_square(pos):
    x, y = pos
    if x < BOARD_X or x >= BOARD_X + BOARD_SIZE or y < BOARD_Y or y >= BOARD_Y + BOARD_SIZE: return None
    file = (x - BOARD_X) // SQ_SIZE; rank = 7 - ((y - BOARD_Y) // SQ_SIZE)
    if 0 <= file <= 7 and 0 <= rank <= 7: return chess.square(file, rank)
    else: return None

def draw_setup_panel(surface, best_move_button_rect, best_move_button_text, toggle_helpers_button_rect, toggle_helpers_button_text, save_game_button_rect, save_game_button_text, button_font):
    panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SETUP_PANEL_HEIGHT)
    pygame.draw.rect(surface, DARK_GREY, panel_rect)
    pygame.draw.rect(surface, BUTTON_COLOR, best_move_button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, best_move_button_rect, 1, border_radius=3)
    best_move_text_surf = button_font.render(best_move_button_text, True, BUTTON_TEXT_COLOR)
    best_move_text_rect = best_move_text_surf.get_rect(center=best_move_button_rect.center)
    surface.blit(best_move_text_surf, best_move_text_rect)
    pygame.draw.rect(surface, BUTTON_COLOR, toggle_helpers_button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, toggle_helpers_button_rect, 1, border_radius=3)
    toggle_helpers_text_surf = button_font.render(toggle_helpers_button_text, True, BUTTON_TEXT_COLOR)
    toggle_helpers_text_rect = toggle_helpers_text_surf.get_rect(center=toggle_helpers_button_rect.center)
    surface.blit(toggle_helpers_text_surf, toggle_helpers_text_rect)
    pygame.draw.rect(surface, BUTTON_COLOR, save_game_button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, save_game_button_rect, 1, border_radius=3)
    save_game_text_surf = button_font.render(save_game_button_text, True, BUTTON_TEXT_COLOR)
    save_game_text_rect = save_game_text_surf.get_rect(center=save_game_button_rect.center)
    surface.blit(save_game_text_surf, save_game_text_rect)

def save_game_history(current_node, start_time, start_board_number):
    if not current_node:
        print("No game history to save.")
        return

    # Create the games directory if it doesn't exist
    games_dir = "./games"
    if not os.path.exists(games_dir):
        os.makedirs(games_dir)

    # Generate the filename using the timestamp at the beginning of the game
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(games_dir, f"game_{timestamp}_{start_board_number}.txt")

    # Write the game history to the file
    with open(filename, "w") as file:
        node = current_node
        moves = []
        while node:
            if node.move:
                try:
                    parent_board = chess.Board(node.parent.fen, chess960=True)
                    move_san = parent_board.san(node.move)
                except ValueError:
                    move_san = node.move.uci() + "?"
                except Exception:
                    move_san = node.move.uci()
                moves.append(move_san)
            node = node.parent

        # Write the moves in reverse order to get the correct sequence
        for move in reversed(moves):
            file.write(f"{move}\n")

    print(f"Game history saved to {filename}")

def play_chess960_pygame(start_board_number):
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y, BADGE_IMAGES, all_badges_loaded

    board = chess.Board(chess960=True)
    engine = None
    message = ""
    game_root = None
    current_node = None
    last_raw_score = None
    one_off_analysis_text = None
    meter_visible = True
    plot_surface = None
    needs_redraw = True
    selected_square = None
    dragging_piece_info = None
    legal_moves_for_selected = []
    last_move_displayed = None
    best_moves_cache = {}
    current_best_move_index = -1
    highlighted_engine_move = None
    helpers_visible = True
    start_time = datetime.now()

    setup_panel_ui_elements = {
        "best_move_button_width": 140,
        "left_start_pos": COORD_PADDING + 5,
        "show_best_move_button_rect": pygame.Rect(
            COORD_PADDING + 5, 5, 140, SETUP_PANEL_HEIGHT - 10),
        "toggle_helpers_button_rect": pygame.Rect(
            COORD_PADDING + 150, 5, 140, SETUP_PANEL_HEIGHT - 10),
        "save_game_button_rect": pygame.Rect(
            COORD_PADDING + 295, 5, 140, SETUP_PANEL_HEIGHT - 10)
    }

    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: pygame.quit(); print("Exiting: Missing piece images."); return
    if not all_badges_loaded: print("Warning: Running without all badge images.")

    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            print(f"Attempting to load engine from: {STOCKFISH_PATH}")
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try:
                engine.ping(); engine.configure({"UCI_Chess960": True})
                print(f"Stockfish engine loaded and configured for Chess960.")
            except (chess.engine.EngineError, BrokenPipeError, Exception) as config_err:
                print(f"Warning: Could not configure UCI_Chess960 or engine error: {config_err}")
        else: message = f"Stockfish not found at '{STOCKFISH_PATH}'. Analysis disabled."; print(message)
    except (FileNotFoundError, OSError, Exception) as e:
        message = f"Error initializing Stockfish: {e}. Analysis disabled."; print(message); engine = None

    def reset_game(start_pos_num):
        nonlocal board, game_root, current_node, last_raw_score, message
        nonlocal best_moves_cache, current_best_move_index, highlighted_engine_move
        nonlocal one_off_analysis_text, selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal needs_redraw, plot_surface, start_time
        try:
            board = chess.Board(chess960=True); board.set_chess960_pos(start_pos_num)
            start_fen = board.fen()
            print(f"Resetting to Chess960 Position {start_pos_num} (FEN: {start_fen})")
            game_root = None; current_node = None; last_raw_score = None
            drawn_tree_nodes.clear(); tree_scroll_x = 0; tree_scroll_y = 0
            reset_transient_state(clear_message=False)
            best_moves_cache = {}; plot_surface = None
            initial_score = None; analysis_msg = ""
            if engine and not board.is_game_over():
                _, analysis_msg, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT)
                if initial_score is None:
                    print(f"Warning: Initial analysis failed. {analysis_msg}")
                    message = f"Pos {start_pos_num}: Initial analysis failed."
                else: message = f"Position {start_pos_num} set. Initial Eval: {analysis_msg}"
            else: message = f"Position {start_pos_num} set."
            game_root = GameNode(fen=start_fen, raw_score=initial_score)
            current_node = game_root; last_raw_score = initial_score
            needs_redraw = True
            start_time = datetime.now()
        except ValueError as ve:
             print(f"Error setting Chess960 position {start_pos_num}: {ve}")
             message = f"Invalid position: {start_pos_num}."
             needs_redraw = True
        except Exception as e:
            print(f"Error resetting game to position {start_pos_num}: {e}")
            message = f"Error setting pos {start_pos_num}."
            needs_redraw = True

    def reset_transient_state(clear_message=True):
        nonlocal selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal one_off_analysis_text, needs_redraw, message
        nonlocal current_best_move_index, highlighted_engine_move
        selected_square = None; dragging_piece_info = None
        legal_moves_for_selected = []; one_off_analysis_text = None
        if clear_message: message = ""
        current_best_move_index = -1; highlighted_engine_move = None

    pygame.display.set_caption(f"Chess960 - Position {start_board_number}")
    reset_game(start_board_number)

    running = True; clock = pygame.time.Clock(); tree_scroll_speed = 30
    dragging_tree = False; drag_start_pos = None; zoom_level = 1.0

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        plot_panel_y = BOARD_Y + BOARD_SIZE
        tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
        tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False; break
            elif event.type == pygame.MOUSEWHEEL:
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    tree_scroll_y -= event.y * tree_scroll_speed
                    tree_scroll_y = max(0, tree_scroll_y)
                    if event.y > 0: zoom_level += 0.1
                    elif event.y < 0: zoom_level -= 0.1
                    zoom_level = max(0.5, min(2.0, zoom_level))
                    needs_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if tree_panel_rect.collidepoint(pos):
                    dragging_tree = True; drag_start_pos = pos
                elif BOARD_X <= pos[0] < BOARD_X + BOARD_SIZE and BOARD_Y <= pos[1] < BOARD_Y + BOARD_SIZE:
                    if event.button == 1:
                        reset_transient_state()
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn:
                                selected_square = sq
                                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                                img = PIECE_IMAGES.get(piece_key)
                                if img:
                                    dragging_piece_info = {'square': sq, 'piece': piece, 'img': img, 'pos': pos}
                                    legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                else: print(f"Error: Could not find image for piece {piece_key}"); reset_transient_state()
                                needs_redraw = True
                    elif event.button == 3:
                         reset_transient_state(); needs_redraw = True
                elif setup_panel_ui_elements["show_best_move_button_rect"].collidepoint(pos):
                    if not engine or board.is_game_over():
                        message = "Engine unavailable or game over."
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True
                        continue

                    current_fen = current_node.fen
                    cached_moves = best_moves_cache.get(current_fen)

                    if cached_moves:
                        current_best_move_index += 1
                        if current_best_move_index >= len(cached_moves):
                            current_best_move_index = 0

                        highlighted_engine_move = cached_moves[current_best_move_index]['move']
                        san = cached_moves[current_best_move_index]['san']
                        score_str = cached_moves[current_best_move_index]['score_str']
                        message = f"Best {current_best_move_index + 1}/{len(cached_moves)}: {san} ({score_str})"
                        one_off_analysis_text = None
                        needs_redraw = True

                    else:
                        message = "Analyzing for best moves..."
                        one_off_analysis_text = None
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True
                        pygame.display.flip()

                        moves_info, error_msg = get_top_engine_moves(board, engine, BEST_MOVE_ANALYSIS_TIME, NUM_BEST_MOVES_TO_SHOW)

                        if error_msg:
                            message = f"Analysis Error: {error_msg}"
                            best_moves_cache[current_fen] = []
                        elif not moves_info:
                            message = "No legal moves found by engine."
                            best_moves_cache[current_fen] = []
                        else:
                            best_moves_cache[current_fen] = moves_info
                            current_best_move_index = 0
                            highlighted_engine_move = moves_info[0]['move']
                            san = moves_info[0]['san']
                            score_str = moves_info[0]['score_str']
                            message = f"Best 1/{len(moves_info)}: {san} ({score_str})"
                        needs_redraw = True
                elif setup_panel_ui_elements["toggle_helpers_button_rect"].collidepoint(pos):
                    helpers_visible = not helpers_visible
                    message = f"Helpers {'ON' if helpers_visible else 'OFF'}"
                    needs_redraw = True
                elif setup_panel_ui_elements["save_game_button_rect"].collidepoint(pos):
                    save_game_history(current_node, start_time, start_board_number)
                    message = "Game history saved."
                    needs_redraw = True
            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 and dragging_piece_info:
                    pos = event.pos; to_sq = screen_to_square(pos)
                    from_sq = dragging_piece_info['square']; move_made = False
                    if to_sq is not None and from_sq != to_sq:
                        promotion = None; piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]: promotion = chess.QUEEN
                        move = chess.Move(from_sq, to_sq, promotion=promotion)
                        if move in board.legal_moves:
                            move_made = True
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move: existing_child = child; break
                            if existing_child:
                                parent_node_for_san = current_node
                                current_node = existing_child; board.set_fen(current_node.fen)
                                last_raw_score = current_node.raw_score
                                try: san = current_node.get_san(chess.Board(parent_node_for_san.fen, chess960=True))
                                except ValueError: san = move.uci()+"?"
                                message = f"Played {san}"
                            else:
                                parent_node = current_node; parent_fen = parent_node.fen
                                temp_board = chess.Board(parent_fen, chess960=True); temp_board.push(move)
                                new_fen = temp_board.fen(); new_raw_score = None
                                analysis_failed = False; analysis_msg = ""
                                if engine and not temp_board.is_game_over():
                                    _, analysis_msg, new_raw_score = get_engine_analysis(temp_board, engine, ANALYSIS_TIME_LIMIT)
                                    if new_raw_score is None: analysis_failed = True; print(f"Warning: Analysis failed. {analysis_msg}")
                                new_node = GameNode(fen=new_fen, move=move, parent=parent_node, raw_score=new_raw_score)
                                new_node.calculate_and_set_move_quality()
                                parent_node.add_child(new_node); current_node = new_node
                                board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                                try: san = new_node.get_san(chess.Board(parent_fen, chess960=True))
                                except ValueError: san = move.uci()+"?"
                                quality_msg = f" ({new_node.move_quality})" if new_node.move_quality else ""
                                message = f"Played {san}"#{quality_msg}"
                                if analysis_failed: message += " (Analysis Failed)"
                            reset_transient_state(clear_message=False); needs_redraw = True
                        else: message = f"Illegal move: {chess.square_name(from_sq)}->{chess.square_name(to_sq)}"; needs_redraw = True
                    dragging_piece_info = None
                    if not move_made:
                        selected_square = None; legal_moves_for_selected = []; needs_redraw = True
                 if event.button == 1 and dragging_tree:
                     dragging_tree = False
                     needs_redraw = True
            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info: dragging_piece_info['pos'] = event.pos; needs_redraw = True
                if dragging_tree:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    tree_scroll_x -= dx
                    tree_scroll_y -= dy
                    drag_start_pos = event.pos
                    needs_redraw = True
            elif event.type == pygame.KEYDOWN:
                node_changed = False; original_node = current_node
                if event.key == pygame.K_LEFT:
                    if current_node and current_node.parent: current_node = current_node.parent; node_changed = True; message = f"Back (Ply {current_node.get_ply()})"
                    else: message = "At start of game"
                elif event.key == pygame.K_RIGHT:
                    if current_node and current_node.children: current_node = current_node.children[0]; node_changed = True; message = f"Forward (Ply {current_node.get_ply()})"
                    else: message = "End of current line"
                elif event.key == pygame.K_UP:
                    if current_node and current_node.parent:
                        parent = current_node.parent
                        siblings = parent.children
                        current_index = siblings.index(current_node)
                        if current_index > 0:
                            current_node = siblings[current_index - 1]
                            node_changed = True
                            message = f"Up (Ply {current_node.get_ply()})"
                        else:
                            message = "At first sibling"
                    else:
                        message = "At start of game"
                elif event.key == pygame.K_DOWN:
                    if current_node and current_node.parent:
                        parent = current_node.parent
                        siblings = parent.children
                        current_index = siblings.index(current_node)
                        if current_index < len(siblings) - 1:
                            current_node = siblings[current_index + 1]
                            node_changed = True
                            message = f"Down (Ply {current_node.get_ply()})"
                        else:
                            message = "At last sibling"
                    else:
                        message = "At start of game"
                if node_changed:
                    try:
                        board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
                        reset_transient_state(clear_message=False); needs_redraw = True
                    except ValueError:
                        message = "Error: Invalid FEN in target node."; print(f"ERROR: Invalid FEN navigation: {current_node.fen}")
                        current_node = original_node; needs_redraw = True
                elif event.key == pygame.K_a:
                     if engine and current_node:
                         message = "Analyzing current position..."; one_off_analysis_text = "Analyzing..."
                         needs_redraw = True; pygame.display.flip()
                         best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT)
                         if raw_score is not None:
                             current_node.raw_score = raw_score; current_node.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                             last_raw_score = raw_score; current_node.calculate_and_set_move_quality()
                             analysis_result_text = f"Suggests: {best_move_san} ({score_str})" if best_move_san else score_str if score_str else "Analysis complete."
                         else: analysis_result_text = f"Analysis failed: {score_str}"
                         one_off_analysis_text = analysis_result_text; message = "";
                         reset_transient_state(clear_message=False); needs_redraw = True
                     else: message = "No engine or node for analysis."; needs_redraw = True
                elif event.key == pygame.K_m:
                    meter_visible = not meter_visible; message = f"Eval Plot {'ON' if meter_visible else 'OFF'}"
                    needs_redraw = True
                elif event.key == pygame.K_ESCAPE: running = False; break

        if not running: continue

        last_move_displayed = current_node.move if current_node and current_node.parent else None
        if current_node and board.fen() != current_node.fen:
             try:
                 board.set_fen(current_node.fen); last_raw_score = current_node.raw_score
             except ValueError:
                 print(f"CRITICAL ERROR: Invalid FEN from node! FEN: {current_node.fen}")
                 message = "CRITICAL FEN ERROR!"; needs_redraw = True

        if needs_redraw:
            screen.fill(DARK_GREY)

            best_move_button_text = "Show Best Move"
            current_fen_for_button = current_node.fen if current_node else None
            cached_moves_for_button = best_moves_cache.get(current_fen_for_button) if current_fen_for_button else None
            if highlighted_engine_move and cached_moves_for_button:
                 num_cached = len(cached_moves_for_button)
                 if num_cached > 0: best_move_button_text = f"Showing {current_best_move_index + 1}/{num_cached}"
                 else: best_move_button_text = "No moves found"
            elif engine and current_node and not board.is_game_over(): best_move_button_text = "Show Best (1)"
            elif current_fen_for_button in best_moves_cache and not cached_moves_for_button: best_move_button_text = "No moves found"
            elif board.is_game_over(): best_move_button_text = "Game Over"

            toggle_helpers_button_text = "Toggle Helpers"
            save_game_button_text = "Save Game"

            draw_setup_panel(screen, setup_panel_ui_elements["show_best_move_button_rect"], best_move_button_text, setup_panel_ui_elements["toggle_helpers_button_rect"], toggle_helpers_button_text, setup_panel_ui_elements["save_game_button_rect"], save_game_button_text, BUTTON_FONT)

            draw_coordinates(screen, COORD_FONT)
            draw_board(screen)
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)
            if helpers_visible:
                if current_node and current_node.move and current_node.parent and current_node.move_quality:
                    draw_board_badge(screen, current_node.move.to_square, current_node.move_quality)
                current_wp = current_node.white_percentage if current_node else 50.0
                draw_eval_bar(screen, current_wp)

                if meter_visible and current_node:
                    path = get_path_to_node(current_node)
                    plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                    if plot_surface:
                        plot_rect = plot_surface.get_rect(topleft=(BOARD_X, plot_panel_y))
                        screen.blit(plot_surface, plot_rect)
                    else:
                        plot_rect = pygame.Rect(BOARD_X, plot_panel_y, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                        pygame.draw.rect(screen, DARK_GREY, plot_rect); pygame.draw.rect(screen, GREY, plot_rect, 1)
                        error_surf = TREE_FONT.render("Plot Error", True, ORANGE); screen.blit(error_surf, error_surf.get_rect(center=plot_rect.center))

            draw_game_tree(screen, game_root, current_node, TREE_FONT, helpers_visible)

            status_y_offset = SCREEN_HEIGHT - 10
            if message:
                 status_surf = TREE_FONT.render(message, True, WHITE)
                 status_rect = status_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(status_surf, status_rect); status_y_offset -= status_rect.height
            if one_off_analysis_text:
                 analysis_surf = TREE_FONT.render(one_off_analysis_text, True, ORANGE)
                 analysis_rect = analysis_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(analysis_surf, analysis_rect)

            pygame.display.flip(); needs_redraw = False

        clock.tick(60)

    print("\nExiting Pygame...")
    if engine:
        try: time.sleep(0.1); engine.quit(); print("Stockfish engine closed.")
        except (AttributeError, BrokenPipeError, Exception) as e: print(f"Error closing engine: {e}")
    plt.close('all'); pygame.quit(); sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chess960 Analysis Board")
    parser.add_argument(
        "board_number",
        type=int,
        help="The Chess960 starting position number (0-959)."
    )
    args = parser.parse_args()

    if not (0 <= args.board_number <= 959):
        print(f"Error: Board number must be between 0 and 959 (inclusive). You provided: {args.board_number}")
        sys.exit(1)

    play_chess960_pygame(args.board_number)
