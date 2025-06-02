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
PLOT_PANEL_HEIGHT = 100 # Adjusted height maybe
TREE_PANEL_HEIGHT = 150 # Adjusted height maybe
EVAL_BAR_WIDTH_PX = 30
SIDE_PANEL_WIDTH = EVAL_BAR_WIDTH_PX + 20 # Eval bar area

# --- Screen Dimensions ---
# Increased width slightly to accommodate the new button
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
HIGHLIGHT_COLOR = (255, 255, 0, 150) # Yellow for selected square
LAST_MOVE_HIGHLIGHT_COLOR = (186, 202, 68, 180) # Yellow-green for last move
ENGINE_MOVE_HIGHLIGHT_COLOR = (0, 150, 255, 150) # Blue for engine's suggested move
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70) # Dark semi-transparent for move dots/circles
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
COORD_COLOR = (180, 180, 180) # Color for a-h, 1-8 (slightly less bright)

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
        # Make black parts the target color
        for x in range(img.get_width()):
            for y in range(img.get_height()):
                color = img.get_at((x, y))
                # If pixel is mostly black and has some alpha, change its color
                if color[3] > 100 and color[0] < 100 and color[1] < 100 and color[2] < 100:
                     img.set_at((x, y), (*target_color, color[3]))
                # Optional: make white parts transparent if needed
                # elif color[3] > 100 and color[0] > 200 and color[1] > 200 and color[2] > 200:
                #     img.set_at((x,y), (0,0,0,0))

        img = pygame.transform.smoothscale(img, target_size)
        return img
    except pygame.error as e: print(f"Error processing badge image '{path}': {e}"); return None
    except Exception as e: print(f"Unexpected error processing badge image '{path}': {e}"); return None


# Load badge images (Unchanged)
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


# --- Font ---
TREE_FONT_SIZE = 12
BUTTON_FONT_SIZE = 14
INPUT_FONT_SIZE = 16
COORD_FONT_SIZE = 12
try:
    TREE_FONT = pygame.font.SysFont("sans", TREE_FONT_SIZE)
    BUTTON_FONT = pygame.font.SysFont("sans", BUTTON_FONT_SIZE)
    INPUT_FONT = pygame.font.SysFont("monospace", INPUT_FONT_SIZE) # Font for input box
    COORD_FONT = pygame.font.SysFont("sans", COORD_FONT_SIZE)     # Font for a-h, 1-8
except Exception as e:
    print(f"Warning: Could not load specific fonts ({e}). Using default.")
    TREE_FONT = pygame.font.Font(None, TREE_FONT_SIZE + 2)
    BUTTON_FONT = pygame.font.Font(None, BUTTON_FONT_SIZE + 2)
    INPUT_FONT = pygame.font.Font(None, INPUT_FONT_SIZE + 2)
    COORD_FONT = pygame.font.Font(None, COORD_FONT_SIZE + 2)


# --- Chess Configuration ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # Make sure this path is correct
ANALYSIS_TIME_LIMIT = 0.4 # Time for background/move analysis
BEST_MOVE_ANALYSIS_TIME = 0.8 # Time for "Show Best Move" button click
NUM_BEST_MOVES_TO_SHOW = 5 # How many moves to fetch/cycle through
EVAL_CLAMP_LIMIT = 800 # Centipawn value clamp for eval bar/plot
MATE_SCORE_PLOT_VALUE = EVAL_CLAMP_LIMIT * 1.5 # Value used for mate scores in plot percentage calculation

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

# --- Move Quality Classification (Unchanged) ---
# Based on centipawn loss relative to engine's best move
# Converts raw score (centipawn/mate) to a percentage (0-100) favoring White
# Higher percentage means better for White
def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    if score is None: return 50.0 # Default to 50/50 if no score
    pov_score = score.white() # Get score from White's perspective
    if pov_score.is_mate():
        mate_val = pov_score.mate()
        # Mate for White is 100%, Mate for Black is 0%
        return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    else:
        # Use mate_score far enough away it doesn't interfere with clamp limit
        cp_val = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        if cp_val is None: return 50.0 # Default if score calculation fails

        # Clamp the score to [-clamp_limit, clamp_limit]
        clamped_cp = max(-clamp_limit, min(clamp_limit, cp_val))
        # Normalize to 0-1 range
        normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
        # Convert to percentage 0-100
        return normalized * 100.0

# Classifies based on the drop in evaluation percentage (from the current player's perspective)
# Requires the node to have parent with score, and current node to have score
def classify_move_quality(white_percentage_before, white_percentage_after, turn_before_move):
    # Convert percentages to evaluation from the player-to-move's perspective (0.0 to 1.0)
    if turn_before_move == chess.WHITE:
        eval_before_pov = white_percentage_before / 100.0
        eval_after_pov = white_percentage_after / 100.0
    else: # Black to move
        eval_before_pov = (100.0 - white_percentage_before) / 100.0
        eval_after_pov = (100.0 - white_percentage_after) / 100.0

    # Calculate the loss in evaluation percentage points (must be non-negative)
    eval_drop = max(0.0, eval_before_pov - eval_after_pov)
    eval_drop_percent = eval_drop * 100 # For easier thresholds

    # Define thresholds based on percentage point drop
    if eval_drop_percent <= 2: return "Best" # Allow tiny fluctuation for best
    elif eval_drop_percent <= 5: return "Excellent"
    elif eval_drop_percent <= 10: return "Good"
    elif eval_drop_percent <= 20: return "Inaccuracy"
    elif eval_drop_percent <= 35: return "Mistake"
    else: return "Blunder"


# --- Game History Tree Node (Modified Quality Calc) ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen; self.move = move; self.parent = parent; self.children = []
        self.raw_score = raw_score;
        # Store white percentage directly on the node
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
        self._san_cache = None; self.x = 0; self.y = 0; self.screen_rect = None
        # Move quality is determined *after* child nodes potentially exist or are updated
        self.move_quality = None # Calculated later

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
        except Exception as e: return self.move.uci() # Fallback

    # Method to calculate and SET the quality of the move *leading* to this node
    def calculate_and_set_move_quality(self):
        if not self.parent or self.parent.white_percentage is None or self.white_percentage is None:
            self.move_quality = None # Cannot calculate without parent score or own score
            return

        wp_before = self.parent.white_percentage
        wp_after = self.white_percentage

        # Determine whose turn it was *before* the move was made
        try:
            parent_board = chess.Board(self.parent.fen, chess960=True)
            turn_before_move = parent_board.turn
        except ValueError: # Invalid FEN in parent?
             self.move_quality = None
             print(f"Warning: Could not create board from parent FEN for quality check: {self.parent.fen}")
             return

        self.move_quality = classify_move_quality(wp_before, wp_after, turn_before_move)


# --- Helper Functions (Chess Logic) ---

def format_score(score, turn):
    if score is None: return "N/A"
    pov_score = score.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        # Ensure mate_in is not None before using abs()
        return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate"
    else:
        # Provide a large mate_score to prioritize mate lines
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        return f"{cp / 100.0:+.2f}" if cp is not None else "N/A (No CP)"

# get_engine_analysis (Unchanged - returns SAN, formatted score string, raw score object)
def get_engine_analysis(board, engine, time_limit):
    if not engine: return None, "Engine unavailable.", None
    try:
        # Use score for basic analysis, PV for best move
        info = engine.analyse(board, chess.engine.Limit(time=time_limit), info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
    except (chess.engine.EngineError, BrokenPipeError, AttributeError) as e:
        print(f"Engine analysis error: {e}")
        return None, f"Engine error.", None # Catch engine errors
    except Exception as e:
         print(f"Unexpected analysis error: {e}")
         return None, f"Analysis error.", None

    best_move = info.get("pv", [None])[0]
    score = info.get("score")

    if score is None:
         # Sometimes score might be missing but PV exists (e.g., during mate search?)
         if best_move:
             # Try analyzing just the move? Less ideal.
             return board.san(best_move), "Score N/A", None
         else:
            return None, "Analysis failed (no score/pv).", None

    score_str = format_score(score, board.turn)

    if best_move:
        try:
            best_move_san = board.san(best_move)
        except Exception:
            best_move_san = best_move.uci() # Fallback
        return best_move_san, f"Score: {score_str}", score
    else:
        # Return score even if no best move (e.g., mate position analysis)
        return None, f"Pos Score: {score_str}", score


# get_top_engine_moves (Unchanged - returns list of dicts, error message)
def get_top_engine_moves(board, engine, time_limit, num_moves):
    if not engine: return [], "Engine not available."
    if board.is_game_over(): return [], "Game is over."
    moves_info = []
    try:
        # Use context manager for analysis
        with engine.analysis(board, chess.engine.Limit(time=time_limit), multipv=num_moves) as analysis:
            for info in analysis:
                # Ensure keys exist before accessing
                if "pv" in info and info["pv"] and "score" in info:
                    move = info["pv"][0]
                    score = info.get("score") # Use .get for safety though 'score' is checked
                    score_str = format_score(score, board.turn)
                    try:
                        move_san = board.san(move)
                    except Exception: # Handle illegal move SAN generation etc.
                        move_san = move.uci()
                    moves_info.append({"move": move, "san": move_san, "score_str": score_str, "score_obj": score})
                    # Stop if we collected enough, analysis might yield more than requested if time allows
                    if len(moves_info) >= num_moves:
                        break
                # else:
                    # Optional: print("Skipping info line missing pv/score:", info) # Debugging
        # Sort by engine's ranking (usually implicit, but can be explicit if needed)
        # The engine usually provides them in order, but multipv results order isn't strictly guaranteed across all engines/versions
        # Sorting by score (descending for current player) might be needed if order isn't reliable
        # For now, assume engine provides in order.
        return moves_info, None
    except (chess.engine.EngineError, BrokenPipeError, AttributeError) as e:
        print(f"MultiPV Analysis engine error: {e}")
        return [], f"Engine error: {e}"
    except Exception as e:
        print(f"MultiPV Analysis unexpected error: {e}")
        return [], f"Analysis error: {e}"


def get_path_to_node(node): # (Unchanged)
    path = []; current = node;
    while current:
        path.append(current);
        current = current.parent;
    return path[::-1] # Return list from root to node

# --- Drawing Functions ---

def draw_board(surface): # Uses BOARD_X, BOARD_Y implicitly
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (BOARD_X + c * SQ_SIZE, BOARD_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# --- NEW: Draw Coordinates ---
def draw_coordinates(surface, font):
    # Files (a-h) - Below board
    for i in range(8):
        text = chr(ord('a') + i)
        text_surf = font.render(text, True, COORD_COLOR)
        # Center text below the respective file column
        text_rect = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y + BOARD_SIZE + COORD_PADDING // 2))
        surface.blit(text_surf, text_rect)

    # Ranks (1-8) - Left of board
    for i in range(8):
        text = str(8 - i) # Rank 8 is at top (i=0), Rank 1 is at bottom (i=7)
        text_surf = font.render(text, True, COORD_COLOR)
         # Center text to the left of the respective rank row
        text_rect = text_surf.get_rect(center=(BOARD_X - COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2))
        surface.blit(text_surf, text_rect)


def draw_pieces(surface, board, piece_images, dragging_piece_info): # Uses BOARD_X, BOARD_Y
    if piece_images is None: return
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Don't draw the piece being dragged from its original square
            if dragging_piece_info and square == dragging_piece_info['square']:
                continue
            piece_symbol = piece.symbol()
            # Determine piece key ('wP', 'bN', etc.)
            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
            img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                # Calculate screen coordinates based on board position and square size
                screen_x = BOARD_X + file * SQ_SIZE
                screen_y = BOARD_Y + (7 - rank) * SQ_SIZE # Pygame y increases downwards
                surface.blit(img, (screen_x, screen_y))

    # Draw the piece being dragged last so it's on top
    if dragging_piece_info and dragging_piece_info['img']:
        img = dragging_piece_info['img']
        # Center the image on the mouse cursor position stored in 'pos'
        img_rect = img.get_rect(center=dragging_piece_info['pos'])
        surface.blit(img, img_rect)


def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight): # Uses BOARD_X, BOARD_Y
    # Draw engine move highlight first (can be overlaid by last move/selection)
    if engine_move_to_highlight:
        # Use SRCALPHA surface for transparency
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(ENGINE_MOVE_HIGHLIGHT_COLOR)
        for sq in [engine_move_to_highlight.from_square, engine_move_to_highlight.to_square]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            surface.blit(s, highlight_rect.topleft)

    # Draw last move highlight (can be overlaid by selection)
    if last_move:
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(LAST_MOVE_HIGHLIGHT_COLOR)
        for sq in [last_move.from_square, last_move.to_square]:
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            surface.blit(s, highlight_rect.topleft)

    # Draw selected square highlight (on top of others for the selected square)
    if selected_square is not None:
        rank = chess.square_rank(selected_square); file = chess.square_file(selected_square)
        highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT_COLOR)
        surface.blit(s, highlight_rect.topleft)

    # Draw possible move indicators (dots/circles)
    if legal_moves_for_selected:
        for move in legal_moves_for_selected:
            dest_sq = move.to_square
            rank = chess.square_rank(dest_sq); file = chess.square_file(dest_sq)
            # Create a transparent surface for drawing the indicator
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            is_capture = board.is_capture(move) or board.is_en_passant(move)
            center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2
            radius = SQ_SIZE // 6 # Adjust radius as needed

            if is_capture:
                # Draw a ring/outline for captures
                pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius + 3, 3) # Slightly thicker outline
            else:
                # Draw a filled circle for non-captures
                pygame.draw.circle(s, POSSIBLE_MOVE_COLOR, (center_x, center_y), radius)

            surface.blit(s, (BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE))


def draw_board_badge(surface, square, quality): # Uses BOARD_X, BOARD_Y
    if not all_badges_loaded: return # Don't try if images failed
    badge_image = BADGE_IMAGES.get(quality)
    badge_color = TREE_MOVE_QUALITY_COLORS.get(quality)
    if quality is None or badge_image is None or badge_color is None: return

    rank = chess.square_rank(square); file = chess.square_file(square)
    # Base coordinates of the square
    square_base_x = BOARD_X + file * SQ_SIZE
    square_base_y = BOARD_Y + (7 - rank) * SQ_SIZE
    # Calculate center of the badge circle based on offsets
    center_x = square_base_x + BOARD_BADGE_OFFSET_X
    center_y = square_base_y + BOARD_BADGE_OFFSET_Y

    # Draw background circle
    pygame.draw.circle(surface, badge_color, (center_x, center_y), BOARD_BADGE_RADIUS)
    # Draw outline circle
    pygame.draw.circle(surface, BOARD_BADGE_OUTLINE_COLOR, (center_x, center_y), BOARD_BADGE_RADIUS, 1)
    # Draw the badge image centered on the circle
    badge_rect = badge_image.get_rect(center=(center_x, center_y))
    surface.blit(badge_image, badge_rect.topleft)


def draw_eval_bar(surface, white_percentage): # Needs positioning update
    bar_height = BOARD_SIZE # Make eval bar same height as the board
    # Position it to the right of the board, centered in the side panel area
    bar_x = BOARD_X + BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = BOARD_Y # Align with the top of the board
    white_percentage = 50.0 if white_percentage is None else white_percentage
    # Calculate heights
    white_height = int(bar_height * (white_percentage / 100.0))
    black_height = bar_height - white_height
    # Draw black part (top)
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    # Draw white part (bottom)
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    # Draw outline
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)


def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    """Creates a surface with the evaluation plot (line + filled area, no text)."""
    if not node_path or len(node_path) < 1: # Can plot even a single point
        # Return a blank surface or placeholder?
        placeholder = pygame.Surface((plot_width_px, plot_height_px))
        placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1) # Border
        return placeholder

    plies = [node.get_ply() for node in node_path]
    # Use 50.0 as default if percentage is None
    percentages = [(node.white_percentage if node.white_percentage is not None else 50.0) for node in node_path]

    # Convert Pygame colors to Matplotlib format (0-1 RGB)
    dark_grey_mpl = tuple(c/255.0 for c in DARK_GREY)
    orange_mpl = tuple(c/255.0 for c in ORANGE)
    grey_mpl = tuple(c/255.0 for c in GREY)
    white_mpl = tuple(c/255.0 for c in WHITE)

    # Create Matplotlib figure and axes
    # Adjust figsize based on desired pixel dimensions and DPI
    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl) # Set figure background
    ax.set_facecolor(dark_grey_mpl)      # Set axes background

    # Plotting
    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=0.6) # Fill area below line
        ax.plot(plies, percentages, color=white_mpl, marker=None, linestyle='-', linewidth=1.5) # No markers, straight line
    elif len(plies) == 1:
        # Plot a single point if only root node exists
        ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=3)

    # Styling - Minimal
    ax.axhline(50, color=orange_mpl, linestyle='--', linewidth=1) # 50% line
    # Set x-limits from ply 0 to max ply in path, or at least 1 ply wide
    ax.set_xlim(0, max(max(plies), 1) if plies else 1)
    ax.set_ylim(0, 100) # Y-axis is always 0-100%

    # --- REMOVE ALL TEXT ---
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([]) # Remove x-axis ticks and labels
    ax.set_yticks([]) # Remove y-axis ticks and labels
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl, alpha=0.4) # Keep faint grid

    # Keep faint spines (axes borders)
    for spine in ax.spines.values():
        spine.set_color(grey_mpl)
        spine.set_linewidth(0.5)

    # Save plot to buffer
    plt.tight_layout(pad=0.1) # Minimal padding
    buf = io.BytesIO()
    try:
        # Save with tight bounding box and minimal padding
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, bbox_inches='tight', pad_inches=0.05)
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig)
        buf.close()
        # Return placeholder on error
        placeholder = pygame.Surface((plot_width_px, plot_height_px))
        placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1)
        return placeholder
    finally:
        plt.close(fig) # Ensure figure is closed
    buf.seek(0)

    # Load buffer into Pygame surface
    plot_surface = None
    try:
        plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e:
        print(f"Error loading plot image from buffer: {e}")
        # Return placeholder on error
        placeholder = pygame.Surface((plot_width_px, plot_height_px))
        placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1)
        return placeholder
    finally:
        buf.close()
    return plot_surface

# --- Horizontal Game Tree Drawing (Unchanged logic, coordinate/panel position updates integrated) ---
TREE_PIECE_SIZE = 18 # Slightly smaller pieces in tree
NODE_DIAMETER = TREE_PIECE_SIZE
HORIZ_SPACING = 40 + TREE_PIECE_SIZE # Space between levels
VERT_SPACING = 2 + TREE_PIECE_SIZE # Base vertical space between siblings
TEXT_OFFSET_X = 4 # Space between piece/node and SAN text
INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 3 # Start large to minimize resizing
INITIAL_TREE_SURFACE_HEIGHT = TREE_PANEL_HEIGHT * 5
drawn_tree_nodes = {} # Cache screen rects of drawn nodes {node: rect}
tree_scroll_x = 0
tree_scroll_y = 0
max_drawn_tree_x = 0 # Track max horizontal extent
max_drawn_tree_y = 0 # Track max vertical extent
tree_render_surface = None # The large surface the tree is drawn onto
temp_san_board = chess.Board(chess960=True) # Reusable board for SAN generation
scaled_tree_piece_images = {} # Cache for resized piece images


def get_scaled_tree_image(piece_key, target_size): # (Unchanged)
    global PIECE_IMAGES, scaled_tree_piece_images
    if PIECE_IMAGES is None: return None # Check if base images loaded
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images: return scaled_tree_piece_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key)
    if original_img:
        try:
            scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size))
            scaled_tree_piece_images[cache_key] = scaled_img
            return scaled_img
        except Exception as e:
             print(f"Error scaling tree image for {piece_key}: {e}")
             return None # Indicate failure
    return None


def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node): # (Unchanged logic)
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board
    node.x = x
    node.y = y_center # Tentative Y, might be adjusted based on children

    piece_img = None
    is_root = not node.parent

    # Try to get the piece image for the move leading to this node
    if node.move and node.parent:
        try:
            # Use the temporary board, set parent FEN
            temp_san_board.set_fen(node.parent.fen)
            moved_piece = temp_san_board.piece_at(node.move.from_square)
            if moved_piece:
                piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
                piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE)
        except ValueError:
             print(f"Warning: Invalid parent FEN '{node.parent.fen}' for tree node piece.")
        except Exception as e:
             print(f"Warning: Error getting piece for tree node: {e}")

    # --- Layout Children Recursively First to Determine Vertical Spread ---
    child_y_positions = []
    child_subtree_heights = []
    total_child_height_estimate = 0
    child_x = x + HORIZ_SPACING

    if node.children:
        # Estimate the total height needed for children
        num_children = len(node.children)
        total_child_height_estimate = (num_children -1) * VERT_SPACING # Base spacing

        # Distribute children vertically around the parent's tentative y_center
        # This is complex for optimal layout; using a simpler approach here.
        # We recursively call, then adjust parent y if needed later.
        # Start placing children from top downwards.
        # Initial guess for starting Y: Parent Y - Estimated half height + Half spacing
        current_child_y_start = y_center - total_child_height_estimate / 2 # + VERT_SPACING / 2

        next_child_y = current_child_y_start
        for i, child in enumerate(node.children):
            # Recursive call gets the actual center Y used by the child and its subtree height
            child_center_y, child_subtree_height = layout_and_draw_tree_recursive(
                surface, child, child_x, next_child_y, level + 1, font, current_node
            )
            child_y_positions.append(child_center_y)
            child_subtree_heights.append(child_subtree_height)

            # Determine Y for the *next* child based on the space taken by the current one
            # Use half the height of the current child's subtree for spacing
            spacing = max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2 if child_subtree_height > 0 else VERT_SPACING)
            next_child_y = child_center_y + child_subtree_height / 2 + spacing / 2 # Move down below the current subtree


    # --- Adjust Parent Y Position ---
    # Optionally, adjust the parent's Y to be the average of its children's Y, if children exist
    # This can sometimes make the tree look more balanced.
    # if child_y_positions:
    #     node.y = sum(child_y_positions) / len(child_y_positions)


    # --- Draw Node at Final Position ---
    node_rect = None
    if piece_img:
        img_rect = piece_img.get_rect(center=(int(node.x), int(node.y)))
        surface.blit(piece_img, img_rect.topleft)
        node_rect = img_rect # Use image rect for bounds
    elif is_root:
        radius = TREE_PIECE_SIZE // 2
        pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    else: # Fallback if no piece image (e.g., null move?)
        radius = 3
        pygame.draw.circle(surface, GREY, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)


    # Update max drawn X coordinate
    if node_rect:
        max_drawn_tree_x = max(max_drawn_tree_x, node_rect.right)

    # Highlight current node
    if node == current_node and node_rect:
        # Make outline slightly larger than the node rect
        pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(4, 4), 1)

    # Draw move quality badge (small circle)
    if node_rect and node.move_quality and node.move_quality in TREE_MOVE_QUALITY_COLORS:
        badge_color = TREE_MOVE_QUALITY_COLORS[node.move_quality]
        # Position badge near the corner of the node rect
        badge_center_x = node_rect.right - TREE_BADGE_RADIUS - 1
        badge_center_y = node_rect.bottom - TREE_BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS)
        pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS, 1) # Outline


    # --- Draw Move Text (SAN) ---
    move_text = ""
    text_rect = None
    if node.parent: # Only moves have text
        # Generate SAN using the temp board set to parent's position
        try:
            temp_san_board.set_fen(node.parent.fen)
            move_text = node.get_san(temp_san_board) # Use cached SAN if available
        except ValueError:
             move_text = node.move.uci() + "?" # Indicate FEN issue
        except Exception:
             move_text = node.move.uci() # Fallback

    if move_text and node_rect:
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
        # Position text to the right of the node
        text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery))
        surface.blit(text_surf, text_rect)
        # Update max X if text extends further
        max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)

    # --- Define Clickable Area ---
    # Start with node rect, expand to include text rect if it exists
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect:
        clickable_rect.width = max(clickable_rect.width, text_rect.right - clickable_rect.left)
    node.screen_rect = clickable_rect # Store the calculated screen rect for click detection
    drawn_tree_nodes[node] = node.screen_rect # Add to global dictionary


    # --- Draw Lines to Children ---
    # Draw lines *after* parent and children nodes are placed
    if node_rect and node.children:
        for i, child in enumerate(node.children):
            # Ensure child has been processed and has coordinates
            if hasattr(child, 'x') and hasattr(child, 'y'):
               # Find the visual bounding box of the child (image or circle)
               child_visual_rect = None
               if child in drawn_tree_nodes:
                   child_visual_rect = drawn_tree_nodes[child] # Use the cached rect if available
               else:
                   # Fallback if rect not cached yet (shouldn't happen often with this flow)
                   child_visual_rect = pygame.Rect(child.x-1, child.y-1,2,2)

               # Draw line from parent's right edge to child's left edge
               start_pos = (node_rect.right, node_rect.centery)
               end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)


    # --- Calculate Subtree Height ---
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE
    subtree_total_height = 0
    if child_y_positions:
        # Find the min/max extent of the children's vertical positions
        min_child_y_center = min(child_y_positions)
        max_child_y_center = max(child_y_positions)
        # Estimate height based on child centers and their own subtree heights
        # This isn't perfect but gives an idea of vertical spread
        est_top = min_child_y_center - (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        est_bottom = max_child_y_center + (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        subtree_total_height = max(0, est_bottom - est_top)

    # Update max drawn Y coordinate (considering node's own height and subtree height)
    node_bottom_extent = node.y + max(my_height, subtree_total_height) / 2
    max_drawn_tree_y = max(max_drawn_tree_y, node_bottom_extent)

    # Return the node's final Y position and the total height of the subtree rooted here
    return node.y, max(my_height, subtree_total_height)


def draw_game_tree(surface, root_node, current_node, font): # Needs coordinate update for panel position
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface

    # Clear previous frame's node positions and max extents
    drawn_tree_nodes.clear()
    max_drawn_tree_x = 0
    max_drawn_tree_y = 0

    # --- Calculate Tree Panel Rect dynamically ---
    plot_panel_y = BOARD_Y + BOARD_SIZE
    tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
    # Tree panel takes remaining width and predefined height
    tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

    # --- Surface Management ---
    # Estimate required size based on node count? Or just use previous max?
    # For simplicity, resize if current surface is too small based on *previous* frame's max draw.
    # This might require one frame lag for resizing if tree grows significantly.
    estimated_required_width = max(INITIAL_TREE_SURFACE_WIDTH, int(max_drawn_tree_x + 2 * HORIZ_SPACING))
    estimated_required_height = max(INITIAL_TREE_SURFACE_HEIGHT, int(max_drawn_tree_y + 2 * VERT_SPACING))

    if tree_render_surface is None or tree_render_surface.get_width() < estimated_required_width or tree_render_surface.get_height() < estimated_required_height:
         try:
             # Create or resize the surface
             new_width = max(estimated_required_width, tree_panel_rect.width) # Ensure at least panel width
             new_height = max(estimated_required_height, tree_panel_rect.height) # Ensure at least panel height
             tree_render_surface = pygame.Surface((new_width, new_height))
             # print(f"Resized tree surface to {new_width}x{new_height}") # Debug
         except pygame.error as e:
             print(f"Error creating/resizing tree surface ({new_width}x{new_height}): {e}")
             # Draw placeholder/error message? For now, just return and skip drawing tree.
             pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect) # Fill panel bg
             pygame.draw.rect(surface, GREY, tree_panel_rect, 1) # Border
             error_surf = font.render("Tree Error", True, ORANGE)
             surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
             return

    # Fill the large tree surface with background color
    tree_render_surface.fill(TREE_BG_COLOR)

    # If no root node, just draw the empty panel background and return
    if not root_node:
        surface.blit(tree_render_surface, tree_panel_rect.topleft, area=pygame.Rect(0,0,tree_panel_rect.width, tree_panel_rect.height))
        pygame.draw.rect(surface, GREY, tree_panel_rect, 1) # Draw border for empty panel
        return

    # --- Layout and Draw Tree ---
    # Initial position for the root node on the large surface
    start_x = 15 + TREE_PIECE_SIZE // 2 # Padding from left edge
    # Try to center root vertically initially, layout will adjust
    start_y = tree_render_surface.get_height() // 2

    # Start recursive drawing. This updates max_drawn_tree_x/y and populates drawn_tree_nodes.
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node)

    # --- Scrolling Logic ---
    # Calculate total content size based on actual drawing extents
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING # Add some padding
    total_tree_height = max_drawn_tree_y + VERT_SPACING # Add some padding

    # Calculate maximum scroll values
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width)
    max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)

    # Auto-scroll to keep current node visible (with margin)
    scroll_margin_x = HORIZ_SPACING * 1.5
    scroll_margin_y = VERT_SPACING * 3
    if current_node and current_node.screen_rect: # Check if current node was drawn and has rect
        node_rect_on_surface = current_node.screen_rect # This rect is relative to tree_render_surface

        # Check right edge
        if node_rect_on_surface.right > tree_scroll_x + tree_panel_rect.width - scroll_margin_x:
            tree_scroll_x = node_rect_on_surface.right - tree_panel_rect.width + scroll_margin_x
        # Check left edge
        if node_rect_on_surface.left < tree_scroll_x + scroll_margin_x:
            tree_scroll_x = node_rect_on_surface.left - scroll_margin_x
        # Check bottom edge
        if node_rect_on_surface.bottom > tree_scroll_y + tree_panel_rect.height - scroll_margin_y:
            tree_scroll_y = node_rect_on_surface.bottom - tree_panel_rect.height + scroll_margin_y
        # Check top edge
        if node_rect_on_surface.top < tree_scroll_y + scroll_margin_y:
            tree_scroll_y = node_rect_on_surface.top - scroll_margin_y

    # Clamp scroll values to their max limits
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x))
    tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))

    # --- Blit Visible Portion ---
    # Define the rectangular area (viewport) of the large surface to display
    source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, tree_panel_rect.width, tree_panel_rect.height)
    try:
        # Blit the calculated portion onto the main screen at the tree panel's position
        surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
    except pygame.error as e:
        print(f"Error blitting tree view: {e}")
        # Draw placeholder on error
        pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
        pygame.draw.rect(surface, GREY, tree_panel_rect, 1)
        error_surf = font.render("Tree Blit Error", True, ORANGE)
        surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))

    # --- Draw Scrollbars (if needed) ---
    scrollbar_thickness = 7
    # Horizontal scrollbar
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width
        scrollbar_width = max(15, tree_panel_rect.width * ratio_visible) # Min width
        # Position based on scroll percentage
        scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio
        scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - scrollbar_thickness - 1, scrollbar_width, scrollbar_thickness)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)
    # Vertical scrollbar
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height
        scrollbar_height = max(15, tree_panel_rect.height * ratio_visible) # Min height
        # Position based on scroll percentage
        scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0
        scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio
        scrollbar_rect = pygame.Rect(tree_panel_rect.right - scrollbar_thickness - 1, scrollbar_y, scrollbar_thickness, scrollbar_height)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)



# --- Coordinate Conversion --- Uses BOARD_X, BOARD_Y offset
def screen_to_square(pos):
    x, y = pos
    # Check if click is within the board bounds including coordinates maybe? No, just board.
    if x < BOARD_X or x >= BOARD_X + BOARD_SIZE or y < BOARD_Y or y >= BOARD_Y + BOARD_SIZE:
        return None # Click was outside the board drawing area
    # Convert screen coordinates relative to board origin
    file = (x - BOARD_X) // SQ_SIZE
    rank = 7 - ((y - BOARD_Y) // SQ_SIZE) # Invert Y axis for chess rank
    # Ensure file/rank are within 0-7 range (should be if bounds check passed)
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    else:
        return None # Should not happen with initial check, but good safety


# --- NEW: Setup Panel Drawing (Includes Best Move Button) ---
def draw_setup_panel(surface, input_rect, setup_button_rect, best_move_button_rect, input_text, input_active, best_move_button_text, font, button_font):
    # Panel background
    panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SETUP_PANEL_HEIGHT)
    pygame.draw.rect(surface, DARK_GREY, panel_rect)

    # --- Input Field ---
    pygame.draw.rect(surface, INPUT_BG_COLOR, input_rect, border_radius=3)
    border_color = INPUT_BORDER_ACTIVE_COLOR if input_active else INPUT_BORDER_INACTIVE_COLOR
    pygame.draw.rect(surface, border_color, input_rect, 2, border_radius=3)
    text_surface = font.render(input_text, True, INPUT_TEXT_COLOR)
    text_rect = text_surface.get_rect(midleft=(input_rect.left + 5, input_rect.centery))
    # Basic text clipping (optional, could use pygame.Rect.clip)
    input_area_rect = input_rect.inflate(-10, -4) # Area for text inside border
    surface.blit(text_surface, text_rect, area=input_area_rect.clip(text_rect))
    # Blinking cursor
    if input_active and int(time.time() * 1.5) % 2 == 0: # Blink faster
         cursor_x = text_rect.right + 1
         # Ensure cursor stays within the box bounds visually
         cursor_x = min(cursor_x, input_rect.right - 4)
         cursor_y_start = input_rect.top + 4
         cursor_y_end = input_rect.bottom - 4
         pygame.draw.line(surface, INPUT_TEXT_COLOR, (cursor_x, cursor_y_start), (cursor_x, cursor_y_end), 1)

    # --- Set/Random Button ---
    pygame.draw.rect(surface, BUTTON_COLOR, setup_button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, setup_button_rect, 1, border_radius=3) # Border
    btn_text_surf = button_font.render("Set/Random", True, BUTTON_TEXT_COLOR)
    btn_text_rect = btn_text_surf.get_rect(center=setup_button_rect.center)
    surface.blit(btn_text_surf, btn_text_rect)

    # --- Show Best Move Button ---
    pygame.draw.rect(surface, BUTTON_COLOR, best_move_button_rect, border_radius=3)
    pygame.draw.rect(surface, GREY, best_move_button_rect, 1, border_radius=3) # Border
    # Use the dynamic button text passed as argument
    best_move_text_surf = button_font.render(best_move_button_text, True, BUTTON_TEXT_COLOR)
    best_move_text_rect = best_move_text_surf.get_rect(center=best_move_button_rect.center)
    surface.blit(best_move_text_surf, best_move_text_rect)


# --- Main Game Function ---
def play_chess960_pygame():
    global PIECE_IMAGES, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y, BADGE_IMAGES, all_badges_loaded
    # --- Game State Variables ---
    board = chess.Board(chess960=True) # Main board object
    engine = None
    message = "" # Persistent status message area
    game_root = None
    current_node = None
    last_raw_score = None # Store the raw score object for percentage calc
    one_off_analysis_text = None # For temporary analysis results (like key 'A')
    meter_visible = True # Eval plot visibility toggle
    plot_surface = None # Surface for the eval plot
    needs_redraw = True # Flag to redraw screen contents
    selected_square = None # Chess square index (0-63) if a piece is selected
    dragging_piece_info = None # Dict: {'square', 'piece', 'img', 'pos'} if dragging
    legal_moves_for_selected = [] # List of legal chess.Move for selected piece
    last_move_displayed = None # The chess.Move object of the last move shown on board
    # --- Best Move Button State ---
    best_moves_cache = {} # Cache: {fen_string: [{'move': Move, 'san': str, 'score_str': str, 'score_obj': Score}, ...]}
    current_best_move_index = -1 # Index of the currently highlighted move in the cache (-1 if none)
    highlighted_engine_move = None # chess.Move object for the engine's highlighted move


    # --- Setup Panel UI Elements ---
    # Position elements relative to each other and panel edges
    input_field_width = 80
    setup_button_width = 100
    best_move_button_width = 140 # Wider for longer text like "Showing 1/5"
    button_padding = 10
    left_start_pos = COORD_PADDING + 5 # Start slightly indented from left coordinate space

    input_field_rect = pygame.Rect(
        left_start_pos,
        5, # Padding from top
        input_field_width,
        SETUP_PANEL_HEIGHT - 10 # Padding top/bottom
    )
    setup_button_rect = pygame.Rect(
        input_field_rect.right + button_padding,
        input_field_rect.top, # Align top
        setup_button_width,
        input_field_rect.height # Same height
    )
    show_best_move_button_rect = pygame.Rect( # <<< Define Rect for the new button
        setup_button_rect.right + button_padding,
        setup_button_rect.top, # Align top
        best_move_button_width,
        setup_button_rect.height # Same height
    )

    # Adjust screen width calculation AGAIN to ensure it fits all buttons
    required_setup_width = show_best_move_button_rect.right + COORD_PADDING # Width needed by setup panel elements + right padding
    required_board_width = BOARD_SIZE + SIDE_PANEL_WIDTH + 2 * COORD_PADDING # Width needed by board + eval bar + coords
    global SCREEN_WIDTH
    new_screen_width = max(required_setup_width, required_board_width, SCREEN_WIDTH) # Use the max required width
    if new_screen_width > SCREEN_WIDTH:
        SCREEN_WIDTH = new_screen_width
        print(f"Adjusting screen width to {SCREEN_WIDTH}px to fit UI elements.")
        # Need to re-initialize screen if width changed after initial setup
        global screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # Also recalculate tree panel width dependent on SCREEN_WIDTH? No, it uses SCREEN_WIDTH directly.


    input_text = "" # Text currently in the FEN input box
    input_active = False # Is the input box currently selected?


    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None:
        # Maybe show an error message on screen?
        pygame.quit()
        print("Exiting: Missing piece images.")
        return
    if not all_badges_loaded:
        print("Warning: Running without all badge images.")


    # --- Engine Initialization ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            print(f"Attempting to load engine from: {STOCKFISH_PATH}")
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try:
                # Check engine responsiveness
                engine.ping()
                 # Configure for Chess960
                engine.configure({"UCI_Chess960": True})
                print(f"Stockfish engine loaded and configured for Chess960.")
            except (chess.engine.EngineError, BrokenPipeError, Exception) as config_err:
                print(f"Warning: Could not configure UCI_Chess960 or engine error: {config_err}")
                # Continue without the option if it fails, engine might still work
        else:
            message = f"Stockfish not found at '{STOCKFISH_PATH}'. Analysis disabled."
            print(message)
    except (FileNotFoundError, OSError, Exception) as e:
        message = f"Error initializing Stockfish: {e}. Analysis disabled."
        print(message)
        engine = None


    # --- Reset Game Function (Includes Clearing Best Move State) ---
    def reset_game(start_pos_num):
        nonlocal board, game_root, current_node, last_raw_score, message
        nonlocal best_moves_cache, current_best_move_index, highlighted_engine_move
        nonlocal one_off_analysis_text, selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal needs_redraw
        try:
            board = chess.Board(chess960=True) # Create a fresh board object
            board.set_chess960_pos(start_pos_num)
            start_fen = board.fen()
            print(f"Resetting to Chess960 Position {start_pos_num} (FEN: {start_fen})")

            # Clear game tree and related state
            game_root = None
            current_node = None
            last_raw_score = None

            drawn_tree_nodes.clear() # Clear clickable areas cache
            tree_scroll_x = 0 # Reset tree scroll
            tree_scroll_y = 0

            # Clear transient UI state and best move state
            reset_transient_state(clear_message=False) # Clears selections, highlights etc.
            best_moves_cache = {} # Clear the cache completely on reset
            plot_surface = None # Clear plot surface

            # Get initial analysis for the root node if engine exists
            initial_score = None
            analysis_msg = ""
            if engine and not board.is_game_over(): # Don't analyze if already mate/stalemate
                _, analysis_msg, initial_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 1.5) # Longer initial analysis
                if initial_score is None:
                    print(f"Warning: Initial analysis failed. {analysis_msg}")
                    message = f"Pos {start_pos_num}: Initial analysis failed."
                else:
                    message = f"Position {start_pos_num} set. Initial Eval: {analysis_msg}"
            else:
                 message = f"Position {start_pos_num} set." # Update message

            # Create the root node
            game_root = GameNode(fen=start_fen, raw_score=initial_score) # Pass initial score here
            current_node = game_root
            last_raw_score = initial_score # Store the raw score

            needs_redraw = True # Flag for redraw

        except ValueError as ve: # Catch specific errors like invalid position number
             print(f"Error setting Chess960 position {start_pos_num}: {ve}")
             message = f"Invalid position: {start_pos_num}."
             # Keep the old board state? Or reset to default? Let's keep old for now.
             needs_redraw = True
        except Exception as e:
            print(f"Error resetting game to position {start_pos_num}: {e}")
            message = f"Error setting pos {start_pos_num}."
            needs_redraw = True


    # --- Reset transient state helper (modified to clear engine highlights) ---
    def reset_transient_state(clear_message=True):
        nonlocal selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal one_off_analysis_text, needs_redraw, message
        nonlocal current_best_move_index, highlighted_engine_move
        selected_square = None
        dragging_piece_info = None
        legal_moves_for_selected = []
        one_off_analysis_text = None # Clear temporary analysis text
        if clear_message:
             message = "" # Optionally clear persistent message
        # --- Clear Best Move Highlight State ---
        current_best_move_index = -1
        highlighted_engine_move = None
        # needs_redraw should generally be set by the caller context that uses this


    # --- Initial Game Setup ---
    initial_random_pos = random.randint(0, 959)
    input_text = str(initial_random_pos) # Show initial random pos in box
    reset_game(initial_random_pos) # Setup the initial board and game state

    # --- Main Game Loop ---
    running = True; clock = pygame.time.Clock(); tree_scroll_speed = 30

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        # Calculate panel Y positions dynamically based on constants
        plot_panel_y = BOARD_Y + BOARD_SIZE
        tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
        tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT) # Used for tree scroll events

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 running = False; break # Exit loop immediately

            # --- Mouse Wheel (Tree Scroll) ---
            elif event.type == pygame.MOUSEWHEEL:
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    # Adjust scroll based on wheel direction
                    tree_scroll_y -= event.y * tree_scroll_speed
                    # Clamp scroll at the top (minimum scroll is 0)
                    tree_scroll_y = max(0, tree_scroll_y)
                    # Max clamping happens within draw_game_tree after content size is known
                    needs_redraw = True

            # --- Mouse Click ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # --- Input Field Activation ---
                if input_field_rect.collidepoint(pos):
                    input_active = True
                    needs_redraw = True # Redraw to show active border/cursor
                else:
                    # Deactivate if clicking anywhere else
                    if input_active:
                        input_active = False
                        needs_redraw = True # Redraw to show inactive border

                # --- Setup Button Click ---
                if setup_button_rect.collidepoint(pos):
                    if input_active: input_active = False # Deactivate input on button press
                    chosen_pos = -1
                    try:
                        num = int(input_text)
                        if 0 <= num <= 959:
                            chosen_pos = num
                        else: # Invalid number range
                            print(f"Input number out of range (0-959): '{input_text}'. Using random.")
                            message = f"Invalid #: {input_text}. Random used."
                            chosen_pos = random.randint(0, 959)
                            input_text = str(chosen_pos) # Update input field
                    except ValueError: # Not an integer
                         print(f"Invalid input: '{input_text}'. Using random.")
                         message = f"Invalid input: {input_text}. Random used."
                         chosen_pos = random.randint(0, 959)
                         input_text = str(chosen_pos) # Update input field

                    if chosen_pos != -1:
                        reset_game(chosen_pos) # Resets the game state
                    needs_redraw = True


                # --- NEW: Show Best Move Button Click ---
                elif show_best_move_button_rect.collidepoint(pos):
                    if input_active: input_active = False # Deactivate input on button press
                    if not engine or board.is_game_over():
                        message = "Engine unavailable or game over."
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True
                        continue # Skip the rest

                    current_fen = current_node.fen
                    cached_moves = best_moves_cache.get(current_fen)

                    if cached_moves: # Moves are cached, cycle through them
                        current_best_move_index += 1
                        if current_best_move_index >= len(cached_moves):
                            current_best_move_index = 0 # Wrap around

                        highlighted_engine_move = cached_moves[current_best_move_index]['move']
                        san = cached_moves[current_best_move_index]['san']
                        score_str = cached_moves[current_best_move_index]['score_str']
                        message = f"Best {current_best_move_index + 1}/{len(cached_moves)}: {san} ({score_str})"
                        one_off_analysis_text = None # Clear other analysis text
                        needs_redraw = True

                    else: # Moves not cached, analyze now
                        message = "Analyzing for best moves..."
                        one_off_analysis_text = None # Clear other analysis text
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True
                        pygame.display.flip() # Show "Analyzing..." message immediately

                        moves_info, error_msg = get_top_engine_moves(board, engine, BEST_MOVE_ANALYSIS_TIME, NUM_BEST_MOVES_TO_SHOW)

                        if error_msg:
                            message = f"Analysis Error: {error_msg}"
                            best_moves_cache[current_fen] = [] # Cache empty list on error
                        elif not moves_info:
                            message = "No legal moves found by engine."
                            best_moves_cache[current_fen] = [] # Cache empty list
                        else:
                            best_moves_cache[current_fen] = moves_info
                            current_best_move_index = 0
                            highlighted_engine_move = moves_info[0]['move']
                            san = moves_info[0]['san']
                            score_str = moves_info[0]['score_str']
                            message = f"Best 1/{len(moves_info)}: {san} ({score_str})"

                        needs_redraw = True


                # --- Tree Click ---
                elif tree_panel_rect.collidepoint(pos):
                    # Calculate click position relative to the tree's scrollable surface
                    relative_panel_x = pos[0]-tree_panel_rect.left
                    relative_panel_y = pos[1]-tree_panel_rect.top
                    absolute_tree_click_x = relative_panel_x + tree_scroll_x
                    absolute_tree_click_y = relative_panel_y + tree_scroll_y

                    clicked_node = None
                    # Iterate through the cached node rectangles
                    # Need to iterate safely if dict could change (it shouldn't here)
                    nodes_to_check = list(drawn_tree_nodes.items())
                    for node, rect in nodes_to_check:
                        # Check collision using the absolute click coordinates
                        if rect and rect.collidepoint(absolute_tree_click_x, absolute_tree_click_y):
                            clicked_node = node
                            break # Found the node

                    if clicked_node and clicked_node != current_node:
                        current_node = clicked_node
                        try:
                            board.set_fen(current_node.fen) # Update board state
                            last_raw_score = current_node.raw_score # Update score display
                            message = f"Jumped to ply {current_node.get_ply()}"
                            reset_transient_state(clear_message=False) # Clear selections, keep message
                            needs_redraw = True
                        except ValueError:
                             message = "Error: Invalid FEN in clicked node."
                             print(f"ERROR: Invalid FEN in tree node: {current_node.fen}")
                             # Don't change current_node if FEN is bad? Or reset?
                             needs_redraw = True


                # --- Board Click (Piece Selection/Drag Start) ---
                elif BOARD_X <= pos[0] < BOARD_X + BOARD_SIZE and BOARD_Y <= pos[1] < BOARD_Y + BOARD_SIZE:
                    if event.button == 1: # Left Click
                        reset_transient_state() # Clear previous selections and highlights
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn: # Select own piece
                                selected_square = sq
                                piece_symbol = piece.symbol()
                                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                                img = PIECE_IMAGES.get(piece_key)
                                if img:
                                    # Start dragging
                                    dragging_piece_info = {'square': sq, 'piece': piece, 'img': img, 'pos': pos}
                                    # Find legal moves for this piece
                                    legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                else:
                                    print(f"Error: Could not find image for piece {piece_key}")
                                    reset_transient_state()
                                needs_redraw = True
                            # else: # Clicked empty/opponent square - handled by reset_transient_state above
                            #    pass
                        # else: # Clicked off board / game over - handled by reset_transient_state above
                        #    pass
                    elif event.button == 3: # Right Click on board cancels selection/drag
                         reset_transient_state()
                         needs_redraw = True

                # Click elsewhere (outside board, tree, setup panel)
                # This case is implicitly handled by the input_active = False logic
                # at the start of the MOUSEBUTTONDOWN handler if not clicking input field.

            # --- Mouse Button Release (Drop Piece / Make Move) ---
            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 and dragging_piece_info: # Left button release while dragging
                    pos = event.pos
                    to_sq = screen_to_square(pos)
                    from_sq = dragging_piece_info['square']
                    move_made = False

                    if to_sq is not None and from_sq != to_sq:
                        # Check for promotion
                        promotion = None
                        piece = dragging_piece_info['piece']
                        if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                            promotion = chess.QUEEN # Auto-promote to Queen for simplicity

                        move = chess.Move(from_sq, to_sq, promotion=promotion)

                        # Check if this is a legal move in the current position
                        if move in board.legal_moves:
                            move_made = True
                            # Check if this move already exists as a child node
                            existing_child = None
                            for child in current_node.children:
                                if child.move == move:
                                    existing_child = child
                                    break

                            if existing_child:
                                # Navigate to existing node
                                parent_node_for_san = current_node # Board state before move
                                current_node = existing_child
                                board.set_fen(current_node.fen)
                                last_raw_score = current_node.raw_score
                                try:
                                    san = current_node.get_san(chess.Board(parent_node_for_san.fen, chess960=True))
                                except ValueError: san = move.uci()+"?"
                                message = f"Played {san}"
                            else:
                                # Create new node
                                parent_node = current_node
                                parent_fen = parent_node.fen # FEN before move

                                # Make the move on a temporary board to get the resulting FEN
                                temp_board = chess.Board(parent_fen, chess960=True)
                                temp_board.push(move)
                                new_fen = temp_board.fen()

                                # Analyze the resulting position
                                new_raw_score = None
                                analysis_failed = False
                                analysis_msg = ""
                                if engine and not temp_board.is_game_over():
                                    _, analysis_msg, new_raw_score = get_engine_analysis(temp_board, engine, ANALYSIS_TIME_LIMIT)
                                    if new_raw_score is None:
                                        analysis_failed = True
                                        print(f"Warning: Analysis failed for new FEN {new_fen}. Msg: {analysis_msg}")

                                # Create the new game node
                                new_node = GameNode(fen=new_fen, move=move, parent=parent_node, raw_score=new_raw_score)
                                # Calculate quality of the move leading to this new node
                                new_node.calculate_and_set_move_quality()
                                # Add to parent's children
                                parent_node.add_child(new_node)
                                # Set new node as current
                                current_node = new_node
                                # Update main board state
                                board.set_fen(current_node.fen)
                                last_raw_score = current_node.raw_score # Update score display

                                # Generate SAN for message
                                try:
                                    san = new_node.get_san(chess.Board(parent_fen, chess960=True))
                                except ValueError: san = move.uci()+"?"
                                quality_msg = f" ({new_node.move_quality})" if new_node.move_quality else ""
                                message = f"Played {san}{quality_msg}"
                                if analysis_failed: message += " (Analysis Failed)"

                            reset_transient_state(clear_message=False) # Keep move message, clear selections
                            needs_redraw = True

                        else:
                             # Dropped on an invalid square for this piece
                             message = f"Illegal move: {chess.square_name(from_sq)} to {chess.square_name(to_sq)}"
                             needs_redraw = True
                             # Don't clear drag info yet, happens below

                    # Always clear dragging info after mouse up, regardless of success
                    dragging_piece_info = None
                    # If move wasn't made successfully (illegal drop, off-board drop), clear selection too
                    if not move_made:
                        selected_square = None
                        legal_moves_for_selected = []
                        needs_redraw = True


            # --- Mouse Motion (Drag Piece) ---
            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info:
                    dragging_piece_info['pos'] = event.pos # Update piece position to follow mouse
                    needs_redraw = True

            # --- Key Presses ---
            elif event.type == pygame.KEYDOWN:
                 # Handle Input Field Typing
                if input_active:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        # Simulate setup button click on Enter
                        if setup_button_rect: # Check if button exists
                            # --- Replicate button logic ---
                            chosen_pos = -1
                            try:
                                num = int(input_text)
                                if 0 <= num <= 959: chosen_pos = num
                                else: chosen_pos = random.randint(0, 959); input_text = str(chosen_pos); message = "Invalid #. Random used."
                            except ValueError: chosen_pos = random.randint(0, 959); input_text = str(chosen_pos); message = "Invalid input. Random used."
                            if chosen_pos != -1: reset_game(chosen_pos)
                            input_active = False # Deactivate input after Enter
                            needs_redraw = True
                            # --- End replicate ---
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                        needs_redraw = True
                    elif event.unicode.isdigit(): # Only allow digits
                        if len(input_text) < 3: # Limit length to 3 digits
                            input_text += event.unicode
                            needs_redraw = True
                    elif event.key == pygame.K_ESCAPE: # Allow Esc to cancel input
                         input_active = False
                         needs_redraw = True
                else: # Handle game navigation/commands if input not active
                    node_changed = False
                    original_node = current_node
                    if event.key == pygame.K_LEFT: # Navigate back in history
                        if current_node and current_node.parent:
                             current_node = current_node.parent
                             node_changed = True
                             message = f"Back (Ply {current_node.get_ply()})"
                        else: message = "At start of game"
                    elif event.key == pygame.K_RIGHT: # Navigate forward (main line variation 0)
                        if current_node and current_node.children:
                            current_node = current_node.children[0] # Go to first child
                            node_changed = True
                            message = f"Forward (Ply {current_node.get_ply()})"
                        else: message = "End of current line"
                    # Add UP/DOWN for sibling variations? Maybe later.

                    if node_changed:
                        try:
                            board.set_fen(current_node.fen)
                            last_raw_score = current_node.raw_score
                            reset_transient_state(clear_message=False) # Clear selections, keep nav message
                            needs_redraw = True
                        except ValueError:
                            message = "Error: Invalid FEN in target node."
                            print(f"ERROR: Invalid FEN in tree node during navigation: {current_node.fen}")
                            current_node = original_node # Revert if FEN is bad
                            needs_redraw = True


                    # --- Other commands ---
                    elif event.key == pygame.K_a: # Analyze Current Position (explicitly)
                         if engine and current_node:
                             message = "Analyzing current position..."; one_off_analysis_text = "Analyzing..." # Show immediate feedback
                             needs_redraw = True; pygame.display.flip() # Quick update display

                             # Perform analysis (maybe longer than standard move analysis)
                             best_move_san, score_str, raw_score = get_engine_analysis(board, engine, ANALYSIS_TIME_LIMIT * 2.5) # Longer time for 'A'

                             if raw_score is not None:
                                 # Update the score on the *current* node
                                 current_node.raw_score = raw_score
                                 current_node.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
                                 last_raw_score = raw_score # Update display score
                                 # Recalculate quality of the move *leading* to this node, now that score is updated
                                 current_node.calculate_and_set_move_quality()
                                 # Potentially update quality of child moves if parent score changed? More complex.
                                 analysis_result_text = f"Suggests: {best_move_san} ({score_str})" if best_move_san else score_str if score_str else "Analysis complete."
                             else:
                                 analysis_result_text = f"Analysis failed: {score_str}"

                             one_off_analysis_text = analysis_result_text # Display result temporarily
                             message = ""; # Clear persistent message
                             reset_transient_state(clear_message=False) # Clear highlights but keep analysis text
                             needs_redraw = True
                         else:
                             message = "No engine or node available for analysis."
                             needs_redraw = True


                    elif event.key == pygame.K_m: # Toggle Eval Plot visibility
                        meter_visible = not meter_visible
                        message = f"Eval Plot {'ON' if meter_visible else 'OFF'}"
                        needs_redraw = True
                    elif event.key == pygame.K_ESCAPE: # Escape key quits
                        running = False; break # Exit loop


        if not running: continue # Skip rest of loop if quit event occurred

        # --- Update state before drawing ---
        # Determine the last move that led to the current node
        last_move_displayed = current_node.move if current_node and current_node.parent else None

        # Ensure board object matches current node FEN (critical after navigation/reset)
        if current_node and board.fen() != current_node.fen:
             try:
                 board.set_fen(current_node.fen)
                 # Make sure score display also matches
                 last_raw_score = current_node.raw_score
             except ValueError:
                 print(f"CRITICAL ERROR: Attempted to set invalid FEN from node! FEN: {current_node.fen}")
                 message = "CRITICAL FEN ERROR! Resetting might be needed."
                 # Maybe try to revert to parent or root?
                 if current_node.parent: current_node = current_node.parent
                 else: # Reset completely if root FEN is bad? Very unlikely.
                      print("Trying to reset to random position due to FEN error.")
                      reset_game(random.randint(0, 959))
                 needs_redraw = True


        # --- Drawing ---
        if needs_redraw:
            screen.fill(DARK_GREY) # Background for the entire window

            # --- Draw UI Elements ---
            # Determine button text dynamically
            best_move_button_text = "Show Best Move" # Default
            current_fen_for_button = current_node.fen if current_node else None
            cached_moves_for_button = best_moves_cache.get(current_fen_for_button) if current_fen_for_button else None

            if highlighted_engine_move and cached_moves_for_button:
                 num_cached = len(cached_moves_for_button)
                 if num_cached > 0:
                     best_move_button_text = f"Showing {current_best_move_index + 1}/{num_cached}"
                 else: # Cache exists but is empty (analysis failed/no moves)
                     best_move_button_text = "No moves found"
            elif engine and current_node and not board.is_game_over():
                 best_move_button_text = "Show Best (1)" # Ready to analyze
            elif current_fen_for_button in best_moves_cache and not cached_moves_for_button: # Analyzed but no moves found
                 best_move_button_text = "No moves found"
            elif board.is_game_over():
                 best_move_button_text = "Game Over"

            # Draw Setup Panel (including the new button)
            draw_setup_panel(screen, input_field_rect, setup_button_rect, show_best_move_button_rect,
                             input_text, input_active, best_move_button_text, INPUT_FONT, BUTTON_FONT)

            # Draw Board and related items
            draw_coordinates(screen, COORD_FONT) # <<< Draw coordinates
            draw_board(screen)
            # Pass the highlighted engine move to draw_highlights
            draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info)

            # Draw board badge for the move quality leading to current node
            if current_node and current_node.move and current_node.parent and current_node.move_quality:
                draw_board_badge(screen, current_node.move.to_square, current_node.move_quality)

            # Draw Eval Bar using current node's white percentage
            current_wp = current_node.white_percentage if current_node else 50.0
            draw_eval_bar(screen, current_wp)


            # Draw Eval Plot
            if meter_visible and current_node:
                path = get_path_to_node(current_node)
                # Regenerate plot surface if needed (could optimize by caching based on path hash?)
                # For simplicity, regenerate each time redraw is needed and plot is visible
                plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                if plot_surface:
                    # Position plot below the board, aligned with board's left edge
                    plot_rect = plot_surface.get_rect(topleft=(BOARD_X, plot_panel_y))
                    screen.blit(plot_surface, plot_rect)
                else: # Draw placeholder if plot fails
                    plot_rect = pygame.Rect(BOARD_X, plot_panel_y, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                    pygame.draw.rect(screen, DARK_GREY, plot_rect)
                    pygame.draw.rect(screen, GREY, plot_rect, 1)
                    error_surf = TREE_FONT.render("Plot Error", True, ORANGE)
                    screen.blit(error_surf, error_surf.get_rect(center=plot_rect.center))


            # Draw Game Tree
            draw_game_tree(screen, game_root, current_node, TREE_FONT)


            # Draw Status Message Area (e.g., bottom right)
            status_y_offset = SCREEN_HEIGHT - 10
            if message:
                 status_surf = TREE_FONT.render(message, True, WHITE)
                 status_rect = status_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(status_surf, status_rect)
                 status_y_offset -= status_rect.height # Move up for next line

            # Draw One-off Analysis Text (above status message)
            if one_off_analysis_text:
                 analysis_surf = TREE_FONT.render(one_off_analysis_text, True, ORANGE) # Use different color
                 analysis_rect = analysis_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(analysis_surf, analysis_rect)


            pygame.display.flip() # Update the full display
            needs_redraw = False # Reset redraw flag

        # Limit frame rate
        clock.tick(60)

    # --- Cleanup ---
    print("\nExiting Pygame...")
    if engine:
        try:
            # Add a short delay before quitting, sometimes helps if engine is busy
            time.sleep(0.1)
            engine.quit()
            print("Stockfish engine closed.")
        except (AttributeError, BrokenPipeError, Exception) as e: # Catch more potential errors on quit
            print(f"Error closing engine: {e}")
    plt.close('all'); # Close any lingering matplotlib plots
    pygame.quit()
    sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    play_chess960_pygame()