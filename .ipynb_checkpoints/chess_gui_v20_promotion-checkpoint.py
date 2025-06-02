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
import threading
import queue
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')

pygame.init()
pygame.font.init()

# --- Constants ---
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
pygame.display.set_caption("Chess960 Analysis") # Updated Caption

BOARD_X = COORD_PADDING
BOARD_Y = SETUP_PANEL_HEIGHT + COORD_PADDING

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
ORANGE = (255, 165, 0)
DARK_GREY = (60, 60, 60)
#LIGHT_SQ_COLOR = (238, 238, 210) # light for green
#DARK_SQ_COLOR = (118, 150, 86) # green
DARK_SQ_COLOR = (185, 148, 115) # brown
LIGHT_SQ_COLOR = (238, 220, 190) # light for brown
HIGHLIGHT_COLOR = (255, 255, 0, 150)
LAST_MOVE_HIGHLIGHT_COLOR = (186, 202, 68, 180)
ENGINE_MOVE_HIGHLIGHT_COLOR = (0, 150, 255, 150)
POSSIBLE_MOVE_COLOR = (0, 0, 0, 70)
TREE_BG_COLOR = DARK_GREY
TREE_LINE_COLOR = (150, 150, 150)
TREE_NODE_ROOT_COLOR = (100, 100, 150)
TREE_NODE_CURRENT_OUTLINE = (255, 255, 255)
TREE_TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (80, 80, 100)
BUTTON_TEXT_COLOR = WHITE
COORD_COLOR = (180, 180, 180)
PROMOTION_POPUP_BG = (90, 90, 90, 220) # Semi-transparent dark background
PROMOTION_POPUP_BORDER = (150, 150, 150)
PROMOTION_HIGHLIGHT_BG = (130, 130, 150, 220)

# Tree Node Config
TREE_PIECE_SIZE = 40; NODE_DIAMETER = TREE_PIECE_SIZE
HORIZ_SPACING = 40 + TREE_PIECE_SIZE; VERT_SPACING = 2 + TREE_PIECE_SIZE
TEXT_OFFSET_X = 4; TEXT_OFFSET_Y = TREE_PIECE_SIZE//2; INITIAL_TREE_SURFACE_WIDTH = SCREEN_WIDTH * 3
INITIAL_TREE_SURFACE_HEIGHT = TREE_PANEL_HEIGHT * 5

# Tree Badge Colors & Config
TREE_BADGE_RADIUS = TREE_PIECE_SIZE // 5
TREE_BADGE_BEST_COLOR = (0, 180, 0)
TREE_BADGE_EXCELLENT_COLOR = (50, 205, 50)
TREE_BADGE_GOOD_COLOR = (0, 100, 0)
TREE_BADGE_INACCURACY_COLOR = (240, 230, 140)
TREE_BADGE_MISTAKE_COLOR = ORANGE
TREE_BADGE_BLUNDER_COLOR = (200, 0, 0)
TREE_MOVE_QUALITY_COLORS = { "Best": TREE_BADGE_BEST_COLOR, "Excellent": TREE_BADGE_EXCELLENT_COLOR, "Good": TREE_BADGE_GOOD_COLOR, "Inaccuracy": TREE_BADGE_INACCURACY_COLOR, "Mistake": TREE_BADGE_MISTAKE_COLOR, "Blunder": TREE_BADGE_BLUNDER_COLOR }

# Board Badge Config
BOARD_BADGE_RADIUS = 14
BOARD_BADGE_OUTLINE_COLOR = DARK_GREY
BOARD_BADGE_IMAGE_SIZE = (22, 22)
BOARD_BADGE_OFFSET_X = SQ_SIZE - BOARD_BADGE_RADIUS - 2
BOARD_BADGE_OFFSET_Y = BOARD_BADGE_RADIUS + 2

# --- Promotion Popup Config ---
PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT] # Order matters for drawing
PROMOTION_POPUP_WIDTH = SQ_SIZE
PROMOTION_POPUP_HEIGHT = SQ_SIZE * len(PROMOTION_PIECES)
PROMOTION_PIECE_SIZE = (int(SQ_SIZE * 0.8), int(SQ_SIZE * 0.8)) # Size of piece icons in popup

# --- Engine Config ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # ADJUST AS NEEDED
# Short time limit for continuous evaluation used for meters/live score
CONTINUOUS_ANALYSIS_TIME_LIMIT = 0.5
# Longer time limit for finding the top N moves when requested
BEST_MOVE_ANALYSIS_TIME_LIMIT = 5.0
NUM_BEST_MOVES_TO_SHOW = 5
EVAL_CLAMP_LIMIT = 800 # Centipawns for eval bar scaling
MATE_SCORE_PLOT_VALUE = EVAL_CLAMP_LIMIT * 1.5 # Value used in plot for mate scores

# --- Asset Paths ---
PIECE_IMAGE_PATH = "pieces"
BADGE_IMAGE_PATH = "badges"

# --- Asset Loading ---
PIECE_IMAGES = {}
BADGE_IMAGES = {}
all_badges_loaded = True
# Cache for scaled promotion piece images
scaled_promotion_images = {}

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
            # Load original size first
            loaded_images[piece + "_orig"] = img # Store original for scaling
            # Scale for board drawing
            scaled_img = pygame.transform.smoothscale(img, (sq_size, sq_size))
            loaded_images[piece] = scaled_img
        except pygame.error as e: print(f"Error loading piece image '{file_path}': {e}"); all_loaded = False
    if not all_loaded: print("Please ensure all 12 piece PNG files exist in the 'pieces' directory."); return None
    print(f"Loaded {len(loaded_images)//2} piece types.")
    return loaded_images

def get_scaled_promotion_image(piece_key, target_size):
    """Gets or creates a scaled version of a piece image for promotion popup."""
    global PIECE_IMAGES, scaled_promotion_images
    if PIECE_IMAGES is None: return None
    cache_key = (piece_key, target_size)
    if cache_key in scaled_promotion_images: return scaled_promotion_images[cache_key]

    original_img = PIECE_IMAGES.get(piece_key + "_orig") # Get original image
    if original_img:
        try:
            scaled_img = pygame.transform.smoothscale(original_img, target_size)
            scaled_promotion_images[cache_key] = scaled_img
            return scaled_img
        except Exception as e: print(f"Error scaling promotion image for {piece_key}: {e}"); return None
    return None


def load_and_process_badge_image(path, target_size, target_color=WHITE):
    try:
        img = pygame.image.load(path).convert_alpha()
        # Recolor black parts to target_color (keeping transparency)
        for x in range(img.get_width()):
            for y in range(img.get_height()):
                color = img.get_at((x, y))
                # Check if pixel is mostly black and somewhat opaque
                if color[3] > 100 and color[0] < 100 and color[1] < 100 and color[2] < 100:
                     img.set_at((x, y), (*target_color, color[3])) # Keep original alpha
        img = pygame.transform.smoothscale(img, target_size)
        return img
    except pygame.error as e: print(f"Error processing badge image '{path}': {e}"); return None
    except Exception as e: print(f"Unexpected error processing badge image '{path}': {e}"); return None

def load_badges(path=BADGE_IMAGE_PATH):
    global all_badges_loaded
    BADGE_TYPES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
    loaded_badges = {}
    all_loaded = True
    print("Loading and processing badge images...")
    if not os.path.isdir(path):
        print(f"Error: Badge directory '{path}' not found. Badges disabled.")
        all_loaded = False
    else:
        for quality in BADGE_TYPES:
            file_path = os.path.join(path, f"{quality.lower()}.png")
            if not os.path.exists(file_path):
                print(f"  - Missing: {quality} ({file_path})")
                all_loaded = False
                continue
            processed_img = load_and_process_badge_image(file_path, BOARD_BADGE_IMAGE_SIZE, WHITE)
            if processed_img: loaded_badges[quality] = processed_img; print(f"  - Loaded: {quality}")
            else: print(f"  - FAILED processing: {quality}"); all_loaded = False
    if not all_loaded: print("Warning: Some badge images failed to load or process.")
    all_badges_loaded = all_loaded
    return loaded_badges

# Fonts
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

# --- Helper Functions ---

def score_to_white_percentage(score, clamp_limit=EVAL_CLAMP_LIMIT):
    """Converts a chess.engine.Score object to a percentage (0-100) from White's perspective."""
    if score is None: return 50.0 # Neutral if no score
    pov_score = score.white()
    if pov_score.is_mate():
        mate_val = pov_score.mate()
        # 100% if White is mating, 0% if Black is mating
        return 100.0 if mate_val is not None and mate_val > 0 else 0.0
    else:
        cp_val = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2) # Use a high mate score
        if cp_val is None: return 50.0 # Neutral if score is invalid
        # Clamp the centipawn value
        clamped_cp = max(-clamp_limit, min(clamp_limit, cp_val))
        # Normalize to 0-1 range
        normalized = (clamped_cp + clamp_limit) / (2 * clamp_limit)
        # Convert to percentage
        return normalized * 100.0

def classify_move_quality(white_percentage_before, white_percentage_after, turn_before_move):
    """Classifies move quality based on the change in evaluation percentage."""
    if white_percentage_before is None or white_percentage_after is None:
        return None # Cannot classify if evals are missing

    # Convert percentages to POV (Point of View) evaluation (0.0 to 1.0)
    if turn_before_move == chess.WHITE:
        eval_before_pov = white_percentage_before / 100.0
        eval_after_pov = white_percentage_after / 100.0
    else: # Black's turn
        eval_before_pov = (100.0 - white_percentage_before) / 100.0
        eval_after_pov = (100.0 - white_percentage_after) / 100.0

    # Calculate the drop in evaluation from the player's perspective
    eval_drop = max(0.0, eval_before_pov - eval_after_pov) # Drop cannot be negative
    eval_drop_percent = eval_drop * 100

    # Classify based on percentage drop thresholds
    if eval_drop_percent <= 2: return "Best"
    elif eval_drop_percent <= 5: return "Excellent"
    elif eval_drop_percent <= 10: return "Good"
    elif eval_drop_percent <= 20: return "Inaccuracy"
    elif eval_drop_percent <= 35: return "Mistake"
    else: return "Blunder"

def format_score(score, turn):
    """Formats a chess.engine.Score object into a human-readable string."""
    if score is None: return "N/A"
    pov_score = score.pov(turn) # Get score from the perspective of the current player
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate"
    else:
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2)
        return f"{cp / 100.0:+.2f}" if cp is not None else "N/A (No CP)"

# --- Game Tree Node ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen
        self.move = move # The move that led to this node (from parent)
        self.parent = parent
        self.children = []
        self.raw_score = raw_score # chess.engine.Score object (can be None)
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
        self._san_cache = None # Cache for the SAN representation of the move
        self.x = 0 # X coordinate for drawing tree
        self.y = 0 # Y coordinate for drawing tree
        self.screen_rect = None # Pygame Rect for click detection in tree view
        self.move_quality = None # String like "Best", "Blunder", etc.

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_ply(self):
        """Calculates the ply number (depth) of this node in the tree."""
        count = 0; node = self
        while node.parent:
            count += 1; node = node.parent
        return count

    def get_san(self, board_at_parent):
        """Gets the Standard Algebraic Notation (SAN) for the move leading to this node."""
        if self._san_cache is not None: return self._san_cache
        if not self.move or not self.parent: return "root" # Root node has no move
        try:
            # Ensure the board is in the parent's state before getting SAN
            san = board_at_parent.san(self.move)
            self._san_cache = san
            return san
        except Exception as e:
            # Fallback to UCI if SAN fails (e.g., illegal move in context somehow)
            # print(f"SAN calculation error for move {self.move.uci()} from FEN {self.parent.fen}: {e}")
            return self.move.uci() # Use UCI as fallback

    def calculate_and_set_move_quality(self):
        """Calculates and stores the quality of the move leading to this node."""
        # Cannot calculate quality for root or if parent/current score is missing
        if not self.parent or self.parent.white_percentage is None or self.white_percentage is None:
            self.move_quality = None
            return

        wp_before = self.parent.white_percentage
        wp_after = self.white_percentage
        try:
            # Need the board state *before* the move to know whose turn it was
            parent_board = chess.Board(self.parent.fen, chess960=True)
            turn_before_move = parent_board.turn
        except ValueError:
             self.move_quality = None # Invalid FEN
             print(f"Warning: Could not create board from parent FEN for quality check: {self.parent.fen}")
             return

        self.move_quality = classify_move_quality(wp_before, wp_after, turn_before_move)

    def __str__(self):
        move_str = self.move.uci() if self.move else "root"
        score_str = format_score(self.raw_score, chess.WHITE if self.get_ply() % 2 == 0 else chess.BLACK) # Approx turn
        return f"Node(Ply:{self.get_ply()}, Move:{move_str}, Score:{score_str}, Children:{len(self.children)})"


# --- Asynchronous Analysis Manager ---
class AnalysisManager:
    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._target_fen = None
        self._request_type = 'idle' # 'idle', 'continuous', 'best_moves'
        self._best_moves_params = (NUM_BEST_MOVES_TO_SHOW, BEST_MOVE_ANALYSIS_TIME_LIMIT)
        self._results_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._last_continuous_result = None # Store the last score for quick access

    def start(self):
        if not self.engine:
            print("Analysis Manager: No engine provided, cannot start.")
            return
        print("Analysis Manager: Starting analysis thread.")
        self._analysis_thread.start()

    def stop(self):
        print("Analysis Manager: Signalling analysis thread to stop.")
        self._stop_event.set()
        # Give the thread a moment to finish its current task
        self._analysis_thread.join(timeout=1.0)
        if self._analysis_thread.is_alive():
            print("Analysis Manager: Warning - Analysis thread did not stop gracefully.")
        else:
            print("Analysis Manager: Analysis thread stopped.")

    def set_target(self, fen):
        """Request continuous analysis for a given FEN."""
        if not self.engine: return
        with self._lock:
            if self._target_fen != fen or self._request_type != 'continuous':
                # print(f"Analysis Manager: Setting target for continuous analysis: {fen}")
                self._target_fen = fen
                self._request_type = 'continuous'
                # Clear last result when target changes significantly
                if self._target_fen != fen:
                    self._last_continuous_result = None

    def request_best_moves(self, fen, num_moves=NUM_BEST_MOVES_TO_SHOW, time_limit=BEST_MOVE_ANALYSIS_TIME_LIMIT):
        """Request a deeper analysis for the top N moves."""
        if not self.engine: return
        with self._lock:
            print(f"Analysis Manager: Requesting best {num_moves} moves for FEN: {fen} (Time: {time_limit}s)")
            self._target_fen = fen
            self._request_type = 'best_moves'
            self._best_moves_params = (num_moves, time_limit)
            # Clear last result as we're switching modes
            self._last_continuous_result = None


    def get_latest_result(self):
        """Non-blockingly get the latest result from the queue."""
        try:
            result = self._results_queue.get_nowait()
            # If it's a continuous result, update the cached latest score
            if result and result.get('type') == 'continuous':
                 self._last_continuous_result = result
            return result
        except queue.Empty:
            return None

    def get_last_continuous_score(self, fen):
         """ Get the most recent continuous score, but only if it matches the requested FEN """
         if self._last_continuous_result and self._last_continuous_result.get('fen') == fen:
              return self._last_continuous_result.get('score')
         return None

    def _analysis_loop(self):
        """The main loop for the background analysis thread."""
        # Removed: current_analysis_process - No longer needed for continuous
        last_analyzed_fen_continuous = None # Track FEN for continuous separately
        last_analyzed_fen_best_moves = None
        last_request_type = None
        last_best_move_params = None
        best_moves_analysis_handle = None # Specific handle for best_moves

        while not self._stop_event.is_set():
            target_fen = None
            request_type = 'idle'
            best_moves_params = None

            with self._lock:
                target_fen = self._target_fen
                request_type = self._request_type
                best_moves_params = self._best_moves_params

            # --- Stop Best Moves Analysis if Target/Type Changes ---
            if best_moves_analysis_handle and \
               (request_type != 'best_moves' or target_fen != last_analyzed_fen_best_moves or best_moves_params != last_best_move_params):
                # print(f"Analysis Manager: Stopping previous best_moves analysis.")
                try: best_moves_analysis_handle.stop()
                except Exception: pass
                best_moves_analysis_handle = None
                last_analyzed_fen_best_moves = None

            # --- Handle Idle State ---
            if not target_fen or request_type == 'idle':
                last_analyzed_fen_continuous = None # Reset FEN tracking when idle
                time.sleep(0.1) # Sleep briefly when idle
                continue

            # --- Perform Analysis Based on Request Type ---
            try:
                board = chess.Board(target_fen, chess960=True)

                # --- Continuous Analysis (Using engine.analyse) ---
                if request_type == 'continuous':
                    # Run analysis only if the FEN is new for continuous mode
                    if target_fen != last_analyzed_fen_continuous:
                        # print(f"Analysis Manager: Starting continuous analysis for {target_fen}") # Debug
                        limit = chess.engine.Limit(time=CONTINUOUS_ANALYSIS_TIME_LIMIT)
                        try:
                            # engine.analyse is blocking, but runs in this thread
                            info = self.engine.analyse(board, limit, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
                            score = info.get("score")
                            pv = info.get("pv")
                            best_move = pv[0] if pv else None
                            result = {
                                'type': 'continuous', 'fen': target_fen,
                                'score': score, 'best_move': best_move, 'error': None
                            }
                            self._results_queue.put(result)
                            last_analyzed_fen_continuous = target_fen # Mark FEN as analyzed
                        except (chess.engine.EngineError, BrokenPipeError, AttributeError, ValueError, chess.engine.EngineTerminatedError) as e:
                            print(f"Analysis Manager: Engine error during continuous analysis for {target_fen}: {e}")
                            self._results_queue.put({'type': 'continuous', 'fen': target_fen, 'score': None, 'best_move': None, 'error': str(e)})
                            last_analyzed_fen_continuous = None # Allow retry
                            time.sleep(0.5) # Wait a bit before retrying

                    else:
                         # FEN hasn't changed, don't re-analyze immediately unless forced
                         # This prevents spamming the engine if main loop is fast.
                         # Add a small sleep to yield CPU.
                         time.sleep(0.05)


                # --- Best Moves Analysis (Using engine.analysis) ---
                elif request_type == 'best_moves':
                    # Start analysis only if FEN or params are new for best_moves mode
                    if not best_moves_analysis_handle or target_fen != last_analyzed_fen_best_moves or best_moves_params != last_best_move_params:
                        # print(f"Analysis Manager: Starting best moves analysis for {target_fen}") # Debug
                        limit = chess.engine.Limit(time=best_moves_params[1])
                        num_moves = best_moves_params[0]
                        try:
                            # Start the analysis process
                            best_moves_analysis_handle = self.engine.analysis(board, limit, multipv=num_moves, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
                            last_analyzed_fen_best_moves = target_fen
                            last_request_type = request_type
                            last_best_move_params = best_moves_params

                            # --- Collect Results Synchronously (within the thread) ---
                            # Since this is a specific request, we wait for it to finish here.
                            moves_info = []
                            # This loop blocks until analysis completes or is stopped
                            for info in best_moves_analysis_handle:
                                if self._stop_event.is_set(): break # Allow interruption

                                if "pv" in info and info["pv"] and "score" in info:
                                    move = info["pv"][0]
                                    score = info.get("score")
                                    score_str = format_score(score, board.turn)
                                    try: move_san = board.san(move)
                                    except Exception: move_san = move.uci()
                                    if not any(m['move'] == move for m in moves_info): # Avoid duplicates
                                        moves_info.append({"move": move, "san": move_san, "score_str": score_str, "score_obj": score})
                                # Check if we have enough moves early? Optional.
                                # if len(moves_info) >= num_moves:
                                #     break

                            # Put the final result after analysis finishes
                            if not self._stop_event.is_set():
                                self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': moves_info, 'error': None})
                                # print(f"Analysis Manager: Best moves result sent for {target_fen}") # Debug

                        except (chess.engine.EngineError, BrokenPipeError, AttributeError, ValueError, chess.engine.EngineTerminatedError) as e:
                            print(f"Analysis Manager: Engine error during best moves analysis for {target_fen}: {e}")
                            self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': [], 'error': str(e)})
                        except Exception as e: # Catch other unexpected errors
                              print(f"Analysis Manager: Unexpected error during best moves analysis for {target_fen}: {e}")
                              self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': [], 'error': f"Unexpected: {e}"})
                        finally:
                            # Analysis finished or errored, clear the handle
                            best_moves_analysis_handle = None
                            last_analyzed_fen_best_moves = None # Ready for a new request
                            # --- IMPORTANT: Switch back to continuous ---
                            # After best_moves finishes (or errors), tell the manager
                            # to go back to continuous mode for this FEN unless
                            # the main thread requested something else in the meantime.
                            with self._lock:
                                if self._target_fen == target_fen and self._request_type == 'best_moves':
                                    # print(f"Analysis Manager: Reverting to continuous mode for {target_fen} after best moves.")
                                    self._request_type = 'continuous'
                                    last_analyzed_fen_continuous = None # Force re-analysis in continuous mode next cycle


            # --- Handle Invalid FEN ---
            except ValueError as e: # Catch FEN errors
                print(f"Analysis Manager: Invalid FEN received: {target_fen} - Error: {e}")
                self._results_queue.put({'type': request_type, 'fen': target_fen, 'score': None, 'best_move': None, 'moves': [], 'error': f"Invalid FEN: {e}"})
                with self._lock: # Set state to idle to prevent looping on bad FEN
                     self._target_fen = None
                     self._request_type = 'idle'
                last_analyzed_fen_continuous = None
                last_analyzed_fen_best_moves = None
                time.sleep(0.5)

            # --- Handle Unexpected Errors in Loop ---
            except Exception as e:
                 print(f"Analysis Manager: Unexpected error in analysis loop: {e}")
                 import traceback
                 traceback.print_exc()
                 # Clear state to try and recover
                 with self._lock:
                      self._request_type = 'idle'
                 last_analyzed_fen_continuous = None
                 last_analyzed_fen_best_moves = None
                 if best_moves_analysis_handle:
                      try: best_moves_analysis_handle.stop()
                      except: pass
                      best_moves_analysis_handle = None
                 time.sleep(1) # Wait before trying again

            # Small sleep to prevent 100% CPU usage if tasks finish instantly
            time.sleep(0.01)

        # --- Cleanup ---
        if best_moves_analysis_handle:
             print("Analysis Manager: Stopping final best_moves analysis process on exit.")
             try: best_moves_analysis_handle.stop()
             except Exception: pass
        print("Analysis Manager: Analysis loop finished.")

# --- Drawing Functions (Mostly unchanged, minor adaptations) ---

def get_path_to_node(node):
    """Returns the list of nodes from the root to the given node."""
    path = []; current = node;
    while current: path.append(current); current = current.parent;
    return path[::-1] # Reverse to get root -> node order

def draw_board(surface):
    """Draws the chessboard squares."""
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (BOARD_X + c * SQ_SIZE, BOARD_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_coordinates(surface, font):
    """Draws the rank and file labels."""
    # Files (a-h)
    for i in range(8):
        text = chr(ord('a') + i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y + BOARD_SIZE + COORD_PADDING // 2))
        surface.blit(text_surf, text_rect)
    # Ranks (1-8)
    for i in range(8):
        text = str(8 - i)
        text_surf = font.render(text, True, COORD_COLOR)
        text_rect = text_surf.get_rect(center=(BOARD_X - COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2))
        surface.blit(text_surf, text_rect)

def draw_pieces(surface, board, piece_images, dragging_piece_info, promotion_pending=False):
    """Draws the pieces on the board, skipping dragging piece or pawn to be promoted."""
    if piece_images is None: return # Don't draw if images aren't loaded
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Skip drawing the piece being dragged
            if dragging_piece_info and square == dragging_piece_info['square']: continue
            # Skip drawing the pawn waiting for promotion
            if promotion_pending and pending_move and square == pending_move['from_sq']: continue


            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
            img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square); file = chess.square_file(square)
                screen_x = BOARD_X + file * SQ_SIZE
                screen_y = BOARD_Y + (7 - rank) * SQ_SIZE # Pygame y increases downwards
                surface.blit(img, (screen_x, screen_y))

    # Draw the dragging piece last, centered at the mouse position
    if dragging_piece_info and dragging_piece_info['img']:
        img = dragging_piece_info['img']
        # Center the image on the mouse cursor
        img_rect = img.get_rect(center=dragging_piece_info['pos'])
        surface.blit(img, img_rect)


def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight):
    """Draws highlights for selected square, legal moves, last move, and engine move."""

    # Helper to draw semi-transparent rectangles
    def highlight_squares(squares, color):
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA) # Use SRCALPHA for transparency
        s.fill(color)
        for sq in squares:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            surface.blit(s, highlight_rect.topleft)

    # Helper to draw circles for legal moves
    def highlight_legal_moves(moves, color):
        for move in moves:
            dest_sq = move.to_square
            rank = chess.square_rank(dest_sq)
            file = chess.square_file(dest_sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)

            is_capture = board.is_capture(move) or board.is_en_passant(move)
            center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2
            radius = SQ_SIZE // 6 # Radius for non-captures

            if is_capture:
                # Draw a ring/border for captures
                pygame.draw.circle(s, color, (center_x, center_y), radius + 3, 3) # Outer ring
            else:
                # Draw a filled circle for non-captures
                pygame.draw.circle(s, color, (center_x, center_y), radius)

            surface.blit(s, (BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE))

    # Order matters: Draw engine move first, then last move, then selection/legal moves
    if engine_move_to_highlight:
        highlight_squares([engine_move_to_highlight.from_square, engine_move_to_highlight.to_square], ENGINE_MOVE_HIGHLIGHT_COLOR)

    if last_move:
        highlight_squares([last_move.from_square, last_move.to_square], LAST_MOVE_HIGHLIGHT_COLOR)

    if selected_square is not None:
        highlight_squares([selected_square], HIGHLIGHT_COLOR)

    if legal_moves_for_selected:
        highlight_legal_moves(legal_moves_for_selected, POSSIBLE_MOVE_COLOR)


def draw_board_badge(surface, square, quality):
    """Draws a move quality badge on a specific board square."""
    if not all_badges_loaded: return # Skip if images failed
    badge_image = BADGE_IMAGES.get(quality); badge_color = TREE_MOVE_QUALITY_COLORS.get(quality)
    if quality is None or badge_image is None or badge_color is None: return # Skip if no quality or assets missing

    rank = chess.square_rank(square); file = chess.square_file(square)
    square_base_x = BOARD_X + file * SQ_SIZE; square_base_y = BOARD_Y + (7 - rank) * SQ_SIZE

    # Calculate badge center position (top-right corner of the square)
    center_x = square_base_x + BOARD_BADGE_OFFSET_X; center_y = square_base_y + BOARD_BADGE_OFFSET_Y

    # Draw the colored circle background
    pygame.draw.circle(surface, badge_color, (center_x, center_y), BOARD_BADGE_RADIUS)
    # Draw the outline
    pygame.draw.circle(surface, BOARD_BADGE_OUTLINE_COLOR, (center_x, center_y), BOARD_BADGE_RADIUS, 1)

    # Draw the badge icon centered on the circle
    badge_rect = badge_image.get_rect(center=(center_x, center_y))
    surface.blit(badge_image, badge_rect.topleft)

def draw_eval_bar(surface, white_percentage):
    """Draws the evaluation bar based on white's percentage advantage."""
    bar_height = BOARD_SIZE
    bar_x = BOARD_X + BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2
    bar_y = BOARD_Y

    # Handle None case, default to 50%
    white_percentage = 50.0 if white_percentage is None else white_percentage

    # Calculate heights for white and black portions
    white_height = int(bar_height * (white_percentage / 100.0))
    black_height = bar_height - white_height

    # Draw black bar (top part)
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height))
    # Draw white bar (bottom part)
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    # Draw border around the whole bar
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)


def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    """Generates a Pygame surface containing the evaluation plot using Matplotlib."""
    # If no path or only root node, return a placeholder
    if not node_path or len(node_path) < 1: # Changed from < 2 to < 1 to allow plotting single point
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder

    plies = [node.get_ply() for node in node_path]
    # Use 50.0 for nodes where percentage couldn't be calculated
    percentages = [(node.white_percentage if node.white_percentage is not None else 50.0) for node in node_path]

    # Convert Pygame colors to Matplotlib format (0-1 range)
    dark_grey_mpl = tuple(c/255.0 for c in DARK_GREY); orange_mpl = tuple(c/255.0 for c in ORANGE)
    grey_mpl = tuple(c/255.0 for c in GREY); white_mpl = tuple(c/255.0 for c in WHITE)
    black_mpl = tuple(c/255.0 for c in BLACK); darker_grey_mpl = tuple(c/255. for c in [30]*3)

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80) # Use DPI 80 for 1:1 pixel mapping
    fig.patch.set_facecolor(dark_grey_mpl); ax.set_facecolor(darker_grey_mpl)

    # Plot the evaluation line and fill area
    if len(plies) > 1:
        ax.fill_between(plies, percentages, color=white_mpl, alpha=1) # Fill below the line
        ax.plot(plies, percentages, color=white_mpl, marker=None, linestyle='-', linewidth=1.5)
    elif len(plies) == 1: # Handle single point case
         ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=3)


    # Add 50% reference line
    ax.axhline(50, color=orange_mpl, linestyle='-', linewidth=.75)

    # Set plot limits and appearance
    ax.set_xlim(0, max(max(plies), 1) if plies else 1); ax.set_ylim(0, 100) # Ensure x-axis starts at 0, handle empty plies
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("") # No labels/title
    ax.set_xticks([]); ax.set_yticks([]) # No ticks

    # Add subtle grid
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl, alpha=0.4)
    # Style the plot spines (borders)
    for spine in ax.spines.values(): spine.set_color(grey_mpl); spine.set_linewidth(0.5)

    # Render plot to a buffer
    plt.tight_layout(pad=0.1); buf = io.BytesIO()
    try:
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, bbox_inches='tight', pad_inches=0.05)
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig); buf.close()
        # Return placeholder on error
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally:
        plt.close(fig) # Ensure figure is closed

    # Load the plot image from buffer into a Pygame surface
    buf.seek(0); plot_surface = None
    try:
        plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e:
        print(f"Error loading plot image from buffer: {e}")
        # Return placeholder on error
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally:
        buf.close()

    return plot_surface


# --- Game Tree Drawing (Stateful, uses global vars for scrolling/rendering) ---
drawn_tree_nodes = {} # Maps Node object -> screen Rect for click detection
tree_scroll_x = 0
tree_scroll_y = 0
max_drawn_tree_x = 0 # Tracks the max width needed for the tree content
max_drawn_tree_y = 0 # Tracks the max height needed for the tree content
tree_render_surface = None # The large surface the whole tree is drawn onto
temp_san_board = chess.Board(chess960=True) # Reusable board for SAN calculation
scaled_tree_piece_images = {} # Cache for piece images scaled for the tree

def get_scaled_tree_image(piece_key, target_size):
    """Gets or creates a scaled version of a piece image for the tree."""
    global PIECE_IMAGES, scaled_tree_piece_images
    if PIECE_IMAGES is None: return None
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images: return scaled_tree_piece_images[cache_key]

    original_img = PIECE_IMAGES.get(piece_key + "_orig") # Use original for scaling
    if original_img:
        try:
            scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size))
            scaled_tree_piece_images[cache_key] = scaled_img
            return scaled_img
        except Exception as e: print(f"Error scaling tree image for {piece_key}: {e}"); return None
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node, helpers_visible):
    """Recursively calculates positions and draws nodes/lines for the game tree."""
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board

    # Set node's position
    node.x = x
    node.y = y_center

    # Determine piece image for the node (based on the move *leading* to it)
    piece_img = None
    is_root = not node.parent
    if node.move and node.parent:
        try:
            # Use the temporary board set to the PARENT's state
            temp_san_board.set_fen(node.parent.fen)
            moved_piece = temp_san_board.piece_at(node.move.from_square)
            if moved_piece:
                piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
                piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE)
        except ValueError: pass # Ignore invalid parent FEN
        except Exception as e: print(f"Warning: Error getting piece for tree node: {e}") # Other errors

    # --- Layout Children Recursively ---
    child_y_positions = []
    child_subtree_heights = []
    total_child_height_estimate = 0 # Rough estimate for centering
    child_x = x + HORIZ_SPACING # Children are placed to the right

    if node.children:
        num_children = len(node.children)
        # Estimate vertical space needed based on number of children
        total_child_height_estimate = (num_children -1) * VERT_SPACING

        # Start placing children centered around the parent's y
        current_child_y_start = y_center - total_child_height_estimate / 2
        next_child_y = current_child_y_start

        # Recursive call for each child
        for i, child in enumerate(node.children):
            # This recursive call calculates the child's layout *and* draws it
            child_center_y, child_subtree_height = layout_and_draw_tree_recursive(
                surface, child, child_x, next_child_y, level + 1, font, current_node, helpers_visible)

            child_y_positions.append(child_center_y)
            child_subtree_heights.append(child_subtree_height)

            # Determine spacing for the *next* child based on the current child's subtree height
            # Ensure minimum spacing (VERT_SPACING)
            spacing = max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2 if child_subtree_height > 0 else VERT_SPACING)
            next_child_y = child_center_y + child_subtree_height / 2 + spacing / 2 # Position next child below current subtree


    # --- Draw Current Node ---
    node_rect = None # Bounding box of the node visual (piece or circle)
    if piece_img:
        img_rect = piece_img.get_rect(center=(int(node.x), int(node.y)))
        surface.blit(piece_img, img_rect.topleft)
        node_rect = img_rect
    elif is_root:
        # Draw a distinct circle for the root
        radius = TREE_PIECE_SIZE // 2
        pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    else:
        # Draw a small circle for intermediate nodes without a piece (should be rare)
        radius = 3
        pygame.draw.circle(surface, GREY, (int(node.x), int(node.y)), radius)
        node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)

    # Update maximum drawn X coordinate
    if node_rect: max_drawn_tree_x = max(max_drawn_tree_x, node_rect.right)

    # Highlight the currently selected node
    if node == current_node and node_rect:
        pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(4, 4), 1) # Inflate for visibility

    # Draw move quality badge (if applicable and helpers are on)
    if node_rect and node.move_quality and node.move_quality in TREE_MOVE_QUALITY_COLORS and helpers_visible:
        badge_color = TREE_MOVE_QUALITY_COLORS[node.move_quality]
        # Position badge at bottom-right of node rect
        badge_center_x = node_rect.right - TREE_BADGE_RADIUS - 1
        badge_center_y = node_rect.bottom - TREE_BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS)
        pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS, 1) # Outline


    # Draw move text (SAN) next to the node
    move_text = ""
    text_rect = None
    if node.parent: # Only draw text for non-root nodes
        try:
            # Use the temp board set to the PARENT's state for SAN
            temp_san_board.set_fen(node.parent.fen)
            move_text = node.get_san(temp_san_board)
        except ValueError: move_text = node.move.uci() + "?" # Indicate potential issue
        except Exception: move_text = node.move.uci() # Fallback

    if move_text and node_rect:
        text_surf = font.render(move_text, True, TREE_TEXT_COLOR)
        # Position text to the right of the node, slightly below center
        text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery + TEXT_OFFSET_Y))
        surface.blit(text_surf, text_rect)
        # Update max X if text extends further
        max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)

    # --- Store Clickable Area ---
    # Make the clickable area encompass both the node visual and its text
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0) # Start with node rect or point
    if text_rect: # Expand width to include text if it exists
        clickable_rect.width = max(clickable_rect.width, text_rect.right - clickable_rect.left)
    node.screen_rect = clickable_rect # Store the calculated rect on the node itself
    drawn_tree_nodes[node] = node.screen_rect # Add to the global dict for click detection


    # --- Draw Connecting Lines ---
    if node_rect and node.children:
        # Draw lines from this node to each of its children
        for i, child in enumerate(node.children):
            # Check if child has been positioned yet (it should have been by the recursive call)
            if hasattr(child, 'x') and hasattr(child, 'y'):
               # Use the stored screen_rect if available, otherwise estimate center
               child_visual_rect = drawn_tree_nodes.get(child, pygame.Rect(child.x-1, child.y-1,2,2))
               # Draw line from right-center of parent to left-center of child
               start_pos = (node_rect.right, node_rect.centery)
               end_pos = (child_visual_rect.left, child_visual_rect.centery)
               pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)


    # --- Calculate Subtree Height for Parent ---
    # This node's visual height
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE
    subtree_total_height = 0
    if child_y_positions: # If node has children
        # Find the vertical span of the children's centers
        min_child_y_center = min(child_y_positions)
        max_child_y_center = max(child_y_positions)
        # Estimate the total height occupied by children's subtrees
        est_top = min_child_y_center - (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        est_bottom = max_child_y_center + (max(child_subtree_heights)/2 if child_subtree_heights else 0)
        subtree_total_height = max(0, est_bottom - est_top) # Ensure non-negative

    # Update the maximum Y coordinate reached by drawing
    node_bottom_extent = node.y + max(my_height, subtree_total_height) / 2
    max_drawn_tree_y = max(max_drawn_tree_y, node_bottom_extent)

    # Return this node's center Y and the total vertical space its subtree occupies
    return node.y, max(my_height, subtree_total_height)


def draw_game_tree(surface, root_node, current_node, font, helpers_visible):
    """Draws the interactive game tree panel, handling scrolling and rendering."""
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface

    # Reset state for this draw call
    drawn_tree_nodes.clear()
    max_drawn_tree_x = 0
    max_drawn_tree_y = 0

    # Define the area on the main screen where the tree panel is drawn
    plot_panel_y = BOARD_Y + BOARD_SIZE
    tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
    tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

    # --- Determine required size for the off-screen tree surface ---
    # Estimate based on max coordinates reached during the last draw, add padding
    # Ensure minimum size based on initial constants or panel size itself
    min_width = max(INITIAL_TREE_SURFACE_WIDTH, tree_panel_rect.width)
    min_height = max(INITIAL_TREE_SURFACE_HEIGHT, tree_panel_rect.height)
    estimated_required_width = max(min_width, int(max_drawn_tree_x + 2 * HORIZ_SPACING))
    estimated_required_height = max(min_height, int(max_drawn_tree_y + 2 * VERT_SPACING))

    # Create or resize the off-screen surface if needed
    resize_needed = False
    if tree_render_surface is None or \
       tree_render_surface.get_width() < estimated_required_width or \
       tree_render_surface.get_height() < estimated_required_height:
        resize_needed = True
        # Calculate new size, ensuring it's at least the estimated required size
        new_width = estimated_required_width
        new_height = estimated_required_height

        try:
            # print(f"Attempting to resize tree surface to: {new_width}x{new_height}") # Debug
            tree_render_surface = pygame.Surface((new_width, new_height))
            # print("Resize successful.") # Debug
        except (pygame.error, MemoryError, ValueError) as e: # Catch pygame errors, memory errors, value errors (size too big)
            # Handle potential memory errors or size errors if surface is too large
            print(f"Error creating/resizing tree surface ({new_width}x{new_height}): {e}")
            # Option 1: Keep the old surface if resizing failed
            if tree_render_surface:
                print("Keeping previous tree surface.")
                resize_needed = False # Indicate we didn't actually resize
            # Option 2: Invalidate the surface and show error (more explicit)
            else:
                tree_render_surface = None # Ensure it's None if creation failed entirely
            # Draw error message directly onto the screen panel and return
            pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
            pygame.draw.rect(surface, GREY, tree_panel_rect, 1) # Border
            error_msg = f"Tree too large ({new_width}x{new_height})! Error: {e}"
            # Shorten error for display if needed
            error_surf = font.render(error_msg[:80], True, ORANGE) # Limit length
            surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
            return # Cannot proceed without a valid render surface if it failed completely

    # If surface is invalid after trying, exit drawing
    if tree_render_surface is None:
        pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
        pygame.draw.rect(surface, GREY, tree_panel_rect, 1)
        error_surf = font.render("Tree Surface Error", True, ORANGE)
        surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
        return

    # --- Draw the tree onto the off-screen surface ---
    tree_render_surface.fill(TREE_BG_COLOR) # Clear the surface

    if not root_node:
        # If no game started yet, just show empty panel
        surface.blit(tree_render_surface, tree_panel_rect.topleft, area=pygame.Rect(0,0,tree_panel_rect.width, tree_panel_rect.height))
        pygame.draw.rect(surface, GREY, tree_panel_rect, 1) # Border
        return

    # Starting position for drawing the root node
    start_x = 15 + TREE_PIECE_SIZE // 2
    start_y = tree_render_surface.get_height() // 2 # Start roughly in the middle vertically

    # Adjust start_y based on estimated total height if known? Maybe later.

    # Recursive drawing call - Updates max_drawn_tree_x/y and drawn_tree_nodes
    layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node, helpers_visible)

    # --- Handle Scrolling ---
    # Total content size based on what was actually drawn
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING
    total_tree_height = max_drawn_tree_y + VERT_SPACING * 2 # Add some bottom padding

    # Maximum scroll offsets based on the *actual* surface size vs panel size
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width)
    # Adjust max_scroll_y considering potential vertical centering offset
    max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)


    # Auto-scroll logic (check if current_node exists and has screen_rect)
    scroll_margin_x = HORIZ_SPACING * 1.5
    scroll_margin_y = VERT_SPACING * 3
    if current_node and current_node.screen_rect:
        node_rect_on_surface = current_node.screen_rect

        # Calculate target scroll positions to center the node approximately
        target_scroll_x = node_rect_on_surface.centerx - tree_panel_rect.width // 2
        target_scroll_y = node_rect_on_surface.centery - tree_panel_rect.height // 2

        # Apply scroll only if node is outside the margins
        if node_rect_on_surface.right > tree_scroll_x + tree_panel_rect.width - scroll_margin_x:
            tree_scroll_x = node_rect_on_surface.right - tree_panel_rect.width + scroll_margin_x
        elif node_rect_on_surface.left < tree_scroll_x + scroll_margin_x:
            tree_scroll_x = node_rect_on_surface.left - scroll_margin_x
        if node_rect_on_surface.bottom > tree_scroll_y + tree_panel_rect.height - scroll_margin_y:
            tree_scroll_y = node_rect_on_surface.bottom - tree_panel_rect.height + scroll_margin_y
        elif node_rect_on_surface.top < tree_scroll_y + scroll_margin_y:
            tree_scroll_y = node_rect_on_surface.top - scroll_margin_y

    # Clamp scroll values to the valid range based on actual surface size
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x))
    tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))

    # --- Blit the visible portion of the tree surface to the screen ---
    blit_width = min(tree_panel_rect.width, total_tree_width - tree_scroll_x)
    blit_height = min(tree_panel_rect.height, total_tree_height - tree_scroll_y)


    if blit_width > 0 and blit_height > 0:
        source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, blit_width, blit_height)
        try:
            # Blit onto the screen at the panel's top-left position
            surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
        except pygame.error as e: # Catch potential blitting errors
            print(f"Error blitting tree view (source_rect={source_rect}): {e}")
            # Draw error message in the panel
            pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
            pygame.draw.rect(surface, GREY, tree_panel_rect, 1)
            error_surf = font.render("Tree Blit Error", True, ORANGE)
            surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
    else:
        # If calculated blit size is zero or negative, something is wrong (e.g., scroll beyond surface)
        # Fill the area anyway to avoid artifacts
        pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
        # print(f"Warning: Blit size zero or negative. W:{blit_width} H:{blit_height} ScrollX:{tree_scroll_x} ScrollY:{tree_scroll_y} TotalW:{total_tree_width} TotalH:{total_tree_height}")


    # --- Draw Scrollbars ---
    scrollbar_thickness = 7
    # Horizontal scrollbar
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width
        scrollbar_width = max(15, tree_panel_rect.width * ratio_visible)
        scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0
        scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio
        scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - scrollbar_thickness - 1, scrollbar_width, scrollbar_thickness)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)

    # Vertical scrollbar
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height
        scrollbar_height = max(15, tree_panel_rect.height * ratio_visible)
        scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0
        scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio
        scrollbar_rect = pygame.Rect(tree_panel_rect.right - scrollbar_thickness - 1, scrollbar_y, scrollbar_thickness, scrollbar_height)
        pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)


def screen_to_square(pos):
    """Converts screen coordinates (x, y) to a chess square index (0-63)."""
    x, y = pos
    # Check if click is within the board boundaries
    if x < BOARD_X or x >= BOARD_X + BOARD_SIZE or y < BOARD_Y or y >= BOARD_Y + BOARD_SIZE:
        return None
    # Calculate file (0-7) and rank (0-7)
    file = (x - BOARD_X) // SQ_SIZE
    rank = 7 - ((y - BOARD_Y) // SQ_SIZE) # Y is inverted
    # Validate range (should always be true if initial check passes)
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    else:
        return None # Should not happen with the initial boundary check


def draw_setup_panel(surface, ui_elements, button_font, best_move_button_text, helpers_visible):
    """Draws the top panel with buttons."""
    panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SETUP_PANEL_HEIGHT)
    pygame.draw.rect(surface, DARK_GREY, panel_rect)

    # Button Data: (rect_key, text, highlight_condition)
    buttons = [
        ("show_best_move_button_rect", best_move_button_text, False),
        ("toggle_helpers_button_rect", f"Helpers: {'ON' if helpers_visible else 'OFF'}", helpers_visible),
        ("save_game_button_rect", "Save Game", False),
        # Add more buttons here if needed
    ]

    for rect_key, text, highlighted in buttons:
        button_rect = ui_elements[rect_key]
        # Draw button background and border
        bg_color = ORANGE if highlighted else BUTTON_COLOR
        pygame.draw.rect(surface, bg_color, button_rect, border_radius=3)
        pygame.draw.rect(surface, GREY, button_rect, 1, border_radius=3) # Border

        # Draw button text
        text_surf = button_font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=button_rect.center)
        surface.blit(text_surf, text_rect)


def save_game_history(current_node, start_time, start_board_number):
    """Saves the principal variation leading to the current node as SAN moves in PGN format."""
    if not current_node or not current_node.parent: # Need at least one move
        print("No game history to save (at root or no node).")
        return

    # Create the games directory if it doesn't exist
    games_dir = "./games"
    try:
        if not os.path.exists(games_dir):
            os.makedirs(games_dir)
    except OSError as e:
        print(f"Error creating save directory '{games_dir}': {e}")
        return

    # Generate the filename
    timestamp = start_time.strftime("%Y%m%d_%H%M%S") # Use compact timestamp
    filename = os.path.join(games_dir, f"game_{timestamp}_pos{start_board_number}.pgn") # Save as PGN

    # --- Get Moves in SAN ---
    node_path = get_path_to_node(current_node)
    moves_san = []
    move_num = 1
    is_white_move = True # Assume starting position is White's move

    # Define the reusable board *before* the try block
    temp_board_for_san = chess.Board(chess960=True)

    try:
        # Set starting position to determine first mover
        initial_fen = node_path[0].fen
        temp_board_for_san.set_fen(initial_fen)
        is_white_move = (temp_board_for_san.turn == chess.WHITE)

        for i in range(len(node_path) - 1):
            parent_node = node_path[i]
            child_node = node_path[i+1]
            move = child_node.move

            if move:
                # Set board to parent state to get correct SAN
                temp_board_for_san.set_fen(parent_node.fen) # Use the already defined board
                san = temp_board_for_san.san(move)

                move_prefix = ""
                if is_white_move:
                    move_prefix = f"{move_num}. "
                # Add ellipsis for black's first move if white didn't move first
                elif i == 0 and not is_white_move:
                     move_prefix = f"{move_num}... "


                moves_san.append(move_prefix + san)

                # Update turn tracking
                if not is_white_move:
                    move_num += 1
                is_white_move = not is_white_move

            else: # Should not happen in path after root
                 print(f"Warning: Node at index {i+1} has no move.")


    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error generating SAN for PGN: {e}. Saving aborted.")
        return

    # --- Build PGN Header ---
    # Use the initial FEN from the actual path start
    start_fen_for_header = node_path[0].fen if node_path else chess.STARTING_BOARD_FEN # Fallback
    pgn_header = f"""[Event "Chess960 Analysis"]
[Site "Local"]
[Date "{start_time.strftime('%Y.%m.%d')}"]
[Round "-"]
[White "Player"]
[Black "Player"]
[Result "*"]
[FEN "{start_fen_for_header}"]
[SetUp "1"]
[Variant "Chess960"]

""" # Result is unknown '*'


    # Format moves for PGN (e.g., "1. e4 e5 2. Nf3 Nc6")
    pgn_moves = ""
    ply_count = 0
    # Re-check who starts based on the actual starting FEN's turn
    temp_board_check_turn = chess.Board(start_fen_for_header, chess960=True)
    pgn_start_turn_is_white = (temp_board_check_turn.turn == chess.WHITE)

    for san_entry in moves_san:
        pgn_moves += san_entry + " "
        ply_count += 1
        # Add newline occasionally for readability, trying to break after black's move
        if ply_count % 10 == 0 : # Approx every 5 full moves
             pgn_moves += "\n"


    # --- Write to File ---
    try:
        with open(filename, "w") as file:
            file.write(pgn_header)
            file.write(pgn_moves.strip()) # Remove trailing space
            file.write(" *") # End game marker
        print(f"Game history saved to {filename}")
    except IOError as e:
        print(f"Error writing game history to file '{filename}': {e}")


def square_to_screen_coords(square):
    """Converts a chess square index to screen coordinates (center of the square)."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    screen_x = BOARD_X + file * SQ_SIZE + SQ_SIZE // 2
    screen_y = BOARD_Y + (7 - rank) * SQ_SIZE + SQ_SIZE // 2
    return screen_x, screen_y

def draw_promotion_popup(surface, pos, color_to_promote, mouse_pos):
    """Draws the promotion choice popup and returns the overall rect and choice rects."""
    global promotion_popup_rect_cache, promotion_choice_rects_cache # Use global caches

    popup_x = pos[0] - PROMOTION_POPUP_WIDTH // 2
    popup_y = pos[1] - PROMOTION_POPUP_HEIGHT // 2

    # Clamp popup position to be fully visible on screen
    popup_x = max(0, min(popup_x, SCREEN_WIDTH - PROMOTION_POPUP_WIDTH))
    popup_y = max(0, min(popup_y, SCREEN_HEIGHT - PROMOTION_POPUP_HEIGHT))

    overall_rect = pygame.Rect(popup_x, popup_y, PROMOTION_POPUP_WIDTH, PROMOTION_POPUP_HEIGHT)

    # Draw background using Surface for transparency
    popup_surf = pygame.Surface(overall_rect.size, pygame.SRCALPHA)
    popup_surf.fill(PROMOTION_POPUP_BG)
    surface.blit(popup_surf, overall_rect.topleft)

    # Draw border
    pygame.draw.rect(surface, PROMOTION_POPUP_BORDER, overall_rect, 1)

    choice_rects = {}
    color_prefix = 'w' if color_to_promote == chess.WHITE else 'b'

    for i, piece_type in enumerate(PROMOTION_PIECES):
        piece_key = color_prefix + chess.piece_symbol(piece_type).upper()
        img = get_scaled_promotion_image(piece_key, PROMOTION_PIECE_SIZE)
        if img:
            # Calculate rect for this piece choice
            choice_y = overall_rect.top + i * SQ_SIZE
            choice_rect = pygame.Rect(overall_rect.left, choice_y, SQ_SIZE, SQ_SIZE)
            choice_rects[piece_type] = choice_rect

            # Center image within the choice rect
            img_rect = img.get_rect(center=choice_rect.center)

            # Highlight if mouse is over this choice
            if choice_rect.collidepoint(mouse_pos):
                 highlight_surf = pygame.Surface(choice_rect.size, pygame.SRCALPHA)
                 highlight_surf.fill(PROMOTION_HIGHLIGHT_BG)
                 surface.blit(highlight_surf, choice_rect.topleft)

            surface.blit(img, img_rect.topleft)

    # Cache the calculated rects
    promotion_popup_rect_cache = overall_rect
    promotion_choice_rects_cache = choice_rects

    return overall_rect, choice_rects


# Global state for promotion popup
promotion_pending = False
pending_move = None # Stores {'from_sq': ..., 'to_sq': ...}
promotion_popup_pos = None # Screen coordinates for the popup center
promotion_popup_color = None # chess.WHITE or chess.BLACK
promotion_popup_rect_cache = None # Cache the overall popup rect for click detection
promotion_choice_rects_cache = {} # Cache choice rects {piece_type: rect}

# --- Main Game Loop ---
def play_chess960_pygame(start_board_number):
    global PIECE_IMAGES, BADGE_IMAGES, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y, all_badges_loaded
    # Add promotion state variables to global scope access
    global promotion_pending, pending_move, promotion_popup_pos, promotion_popup_color
    global promotion_popup_rect_cache, promotion_choice_rects_cache


    # --- Initialize Game State ---
    board = chess.Board(chess960=True) # The main board object for display/interaction
    engine = None
    analysis_manager = None
    message = "Loading..." # Status message displayed at bottom
    game_root = None # The root GameNode of the move tree
    current_node = None # The GameNode currently being displayed/interacted with
    live_raw_score = None # The latest score from continuous async analysis
    live_best_move = None # The latest best move from continuous async analysis
    best_moves_result = None # Stores the result from a 'best_moves' request {fen:..., moves:..., error:...}
    analysis_error_message = None # Displays persistent analysis errors

    meter_visible = True # Toggles the evaluation plot visibility
    plot_surface = None # Surface holding the rendered eval plot
    needs_redraw = True # Flag to trigger screen redraw

    # Interaction state
    selected_square = None # Chess square index (0-63) currently selected
    dragging_piece_info = None # Info if a piece is being dragged {'square', 'piece', 'img', 'pos'}
    legal_moves_for_selected = [] # List of legal moves for the selected piece
    last_move_displayed = None # The last move made (for highlighting)
    highlighted_engine_move = None # Engine's suggested move (from best_moves analysis)
    current_best_move_index = -1 # Index for cycling through top N moves

    helpers_visible = True # Toggles move quality badges, eval bar, plot
    start_time = datetime.now() # Timestamp for saving game filename

    # UI Element Rects (defined once)
    setup_panel_ui_elements = {
        "button_width": 140,
        "button_height": SETUP_PANEL_HEIGHT - 10,
        "button_y": 5,
        "button_spacing": 10,
        "left_start_pos": COORD_PADDING + 5,
    }
    # Calculate rects dynamically
    btn_x = setup_panel_ui_elements["left_start_pos"]
    setup_panel_ui_elements["show_best_move_button_rect"] = pygame.Rect(
        btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"]
    )
    btn_x += setup_panel_ui_elements["button_width"] + setup_panel_ui_elements["button_spacing"]
    setup_panel_ui_elements["toggle_helpers_button_rect"] = pygame.Rect(
        btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"]
    )
    btn_x += setup_panel_ui_elements["button_width"] + setup_panel_ui_elements["button_spacing"]
    setup_panel_ui_elements["save_game_button_rect"] = pygame.Rect(
        btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"]
    )

    # --- Load Assets ---
    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: pygame.quit(); print("Exiting: Missing piece images."); return
    BADGE_IMAGES = load_badges()
    # all_badges_loaded is set within load_badges

    # --- Initialize Engine ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            print(f"Attempting to load engine from: {STOCKFISH_PATH}")
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try:
                # Attempt to configure 960 mode, but don't crash if it fails
                try: engine.configure({"UCI_Chess960": "true"}) # Use string "true"
                except Exception as cfg_err: print(f"Note: Could not configure UCI_Chess960 (might be default): {cfg_err}")
                engine.ping() # Check if engine is responsive
                print(f"Stockfish engine loaded and seems responsive.")
                analysis_manager = AnalysisManager(engine) # Create manager AFTER engine is confirmed working
                analysis_manager.start()
            except (chess.engine.EngineError, BrokenPipeError, Exception) as config_err:
                print(f"Warning: Engine error during setup/ping: {config_err}")
                message = "Engine error during setup. Analysis may fail."
                if engine: engine.quit(); engine = None
        else:
            message = f"Engine not found at '{STOCKFISH_PATH}'. Analysis disabled."
            print(message)
    except (FileNotFoundError, OSError, Exception) as e:
        message = f"Error initializing engine process: {e}. Analysis disabled."
        print(message); engine = None # Ensure engine is None if init fails


    # --- Helper Function to Make Move and Update State ---
    def make_move_and_update(move_to_make):
        nonlocal current_node, board, live_raw_score, message, needs_redraw
        nonlocal best_moves_result, highlighted_engine_move, current_best_move_index

        parent_node = current_node # Node before the move

        # Check if this move already exists as a child
        existing_child = None
        for child in parent_node.children:
            if child.move == move_to_make: existing_child = child; break

        if existing_child:
            # Navigate to existing node
            current_node = existing_child
            try:
                 board.set_fen(current_node.fen)
                 live_raw_score = current_node.raw_score # Use stored score
                 # Get SAN using the parent board state
                 temp_san_board.set_fen(parent_node.fen) # Use global temp_san_board
                 san = current_node.get_san(temp_san_board) # Use global temp_san_board
                 message = f"Played {san} (existing)"
            except ValueError:
                 message = f"Played {move_to_make.uci()} (invalid FEN?)"
                 board = chess.Board(current_node.fen, chess960=True) # Force board sync

        else:
            # Create a new node for the move
            parent_fen = parent_node.fen
            # Apply move to a temporary board to get the new FEN
            temp_board = chess.Board(parent_fen, chess960=True); temp_board.push(move_to_make)
            new_fen = temp_board.fen()
            # Create node without score initially (async will provide)
            new_node = GameNode(fen=new_fen, move=move_to_make, parent=parent_node, raw_score=None)

            # Get SAN using the parent board state
            try:
                temp_san_board.set_fen(parent_fen) # Use global temp_san_board
                san = new_node.get_san(temp_san_board) # Use global temp_san_board
                message = f"Played {san}"
            except ValueError:
                san = move_to_make.uci()+"?"
                message = f"Played {san} (FEN Error?)"

            # Add the new node to the tree
            parent_node.add_child(new_node)
            current_node = new_node
            board.set_fen(current_node.fen) # Update main board
            live_raw_score = None # Clear score, wait for async update

            # Quality calculation is deferred until async result arrives...

        # --- Post-Move Updates ---
        reset_transient_state(clear_message=False) # Keep move message
        # Clear best move analysis as position changed
        best_moves_result = None
        highlighted_engine_move = None
        current_best_move_index = -1
        # Request analysis for the new position
        if analysis_manager:
             analysis_manager.set_target(current_node.fen)
             if "Played" in message: message += " - Analyzing..."

        needs_redraw = True


    # --- Game Setup ---
    def reset_game(start_pos_num):
        nonlocal board, game_root, current_node, live_raw_score, message, start_time
        nonlocal best_moves_result, highlighted_engine_move, analysis_error_message
        nonlocal plot_surface, needs_redraw
        global tree_scroll_x, tree_scroll_y # Use global for these
        global promotion_pending # Reset promotion state on game reset

        try:
            board = chess.Board(chess960=True)
            board.set_chess960_pos(start_pos_num)
            start_fen = board.fen()
            print(f"Resetting to Chess960 Position {start_pos_num} (FEN: {start_fen})")

            # Reset game state variables
            game_root = None; current_node = None; live_raw_score = None
            best_moves_result = None; highlighted_engine_move = None
            analysis_error_message = None
            drawn_tree_nodes.clear()
            # Reset global scroll positions
            tree_scroll_x = 0
            tree_scroll_y = 0
            reset_transient_state(clear_message=False) # Keep message potentially
            plot_surface = None # Clear plot
            promotion_pending = False # Ensure promotion state is reset


            # Create root node (no initial score yet, async will provide)
            game_root = GameNode(fen=start_fen, raw_score=None)
            current_node = game_root
            message = f"Position {start_pos_num} set."

            # Start analysis for the new root position
            if analysis_manager:
                 analysis_manager.set_target(start_fen)
                 message += " Analyzing..." # Indicate analysis started

            needs_redraw = True
            start_time = datetime.now() # Reset start time for saving
            pygame.display.set_caption(f"Chess960 Analysis - Position {start_board_number}") # Update caption


        except ValueError as ve:
             print(f"Error setting Chess960 position {start_pos_num}: {ve}")
             message = f"Invalid position: {start_pos_num}."
             if game_root is None: # Ensure there's a root even on error
                  board = chess.Board(chess960=True) # Default board
                  game_root = GameNode(fen=board.fen())
                  current_node = game_root
             # Reset scroll on error too
             tree_scroll_x = 0
             tree_scroll_y = 0
             promotion_pending = False
             needs_redraw = True
        except Exception as e:
            print(f"Error resetting game to position {start_pos_num}: {e}")
            message = f"Error setting pos {start_pos_num}."
            if game_root is None:
                 board = chess.Board(chess960=True)
                 game_root = GameNode(fen=board.fen())
                 current_node = game_root
                 # Reset scroll on error too
                 tree_scroll_x = 0
                 tree_scroll_y = 0
                 promotion_pending = False
                 needs_redraw = True

    def reset_transient_state(clear_message=True):
        """Resets UI interaction state like selection, dragging, etc."""
        nonlocal selected_square, dragging_piece_info, legal_moves_for_selected
        nonlocal needs_redraw, message # Keep message by default now
        nonlocal current_best_move_index
        # Do NOT reset promotion state here, only on move completion/cancellation
        selected_square = None; dragging_piece_info = None
        legal_moves_for_selected = []
        # Don't clear highlighted_engine_move here, it persists until changed
        # Don't clear best_moves_result here
        if clear_message: message = ""


    # Initialize the game to the specified position
    reset_game(start_board_number)

    # --- Main Loop ---
    running = True; clock = pygame.time.Clock(); tree_scroll_speed = 30
    dragging_tree = False; drag_start_pos = None

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        # Calculate tree panel rect for mouse collision checks
        plot_panel_y = BOARD_Y + BOARD_SIZE
        tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
        tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

        # --- Process Async Analysis Results ---
        if analysis_manager:
            analysis_result = analysis_manager.get_latest_result()
            if analysis_result:
                needs_redraw = True # Got new data, need to update display
                result_fen = analysis_result.get('fen')
                result_type = analysis_result.get('type')
                error = analysis_result.get('error')

                # Store persistent error messages
                if error:
                    analysis_error_message = f"Analysis Err: {error}"
                    print(f"Received analysis error for FEN {result_fen}: {error}") # Log it
                    # Clear potentially stale data on error
                    if result_type == 'continuous':
                        live_raw_score = None
                        live_best_move = None
                    elif result_type == 'best_moves':
                        best_moves_result = analysis_result # Keep error info
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        message = "Best move analysis failed."


                # --- Handle Continuous Results ---
                elif result_type == 'continuous' and current_node and result_fen == current_node.fen:
                    # Update live score only if the result matches the currently viewed node
                    new_score = analysis_result.get('score')
                    new_best_move = analysis_result.get('best_move')
                    # print(f"Received continuous: FEN={result_fen}, Score={new_score}, Move={new_best_move}") # Debug
                    if new_score is not None:
                         live_raw_score = new_score
                         # Optionally update the node's score if it was missing
                         if current_node.raw_score is None:
                              current_node.raw_score = new_score
                              current_node.white_percentage = score_to_white_percentage(new_score, EVAL_CLAMP_LIMIT)
                              # Calculate quality *now* that we have the score for this node
                              # (Requires parent score to be available too)
                              current_node.calculate_and_set_move_quality()

                    if new_best_move is not None:
                         live_best_move = new_best_move
                    # Clear general analysis error if we get a good result
                    analysis_error_message = None


                # --- Handle Best Moves Results ---
                elif result_type == 'best_moves' and current_node and result_fen == current_node.fen:
                    # Store the full result (list of moves)
                    best_moves_result = analysis_result
                    moves_found = analysis_result.get('moves', [])
                    # print(f"Received best_moves: FEN={result_fen}, Moves: {len(moves_found)}") # Debug

                    if moves_found:
                        highlighted_engine_move = moves_found[0]['move'] # Highlight the top one initially
                        current_best_move_index = 0
                        num_found = len(moves_found)
                        san = moves_found[0]['san']
                        score_str = moves_found[0]['score_str']
                        message = f"Best 1/{num_found}: {san} ({score_str})"
                    else:
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        message = "Engine found no moves."

                    # Clear general analysis error if we get a good result
                    analysis_error_message = None


        # --- Process Pygame Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False; break

            # --- If Promotion Popup is Active, Handle Clicks First ---
            if promotion_pending:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    # Check if click is inside the popup area
                    if promotion_popup_rect_cache and promotion_popup_rect_cache.collidepoint(pos):
                        # Check which piece choice was clicked
                        chosen_piece = None
                        for piece_type, rect in promotion_choice_rects_cache.items():
                            if rect.collidepoint(pos):
                                chosen_piece = piece_type
                                break

                        if chosen_piece:
                            # Construct the move with the chosen promotion
                            move = chess.Move(pending_move['from_sq'], pending_move['to_sq'], promotion=chosen_piece)
                            # Ensure the constructed move is actually legal (should always be if pending_move was valid)
                            if move in board.legal_moves:
                                make_move_and_update(move) # Use the helper function
                            else:
                                print(f"Error: Constructed promotion move {move.uci()} is illegal?")
                                message = "Illegal promotion move?"

                            # Reset promotion state
                            promotion_pending = False
                            pending_move = None
                            promotion_popup_rect_cache = None
                            promotion_choice_rects_cache = {}
                            needs_redraw = True
                        # else: click inside popup but missed a piece rect (unlikely with current layout)

                    else:
                        # Clicked *outside* the popup - Cancel the move
                        message = "Promotion cancelled."
                        promotion_pending = False
                        pending_move = None
                        promotion_popup_rect_cache = None
                        promotion_choice_rects_cache = {}
                        needs_redraw = True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: # Allow cancelling with Esc
                         message = "Promotion cancelled."; promotion_pending = False; pending_move = None
                         promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}
                         needs_redraw = True

                # Ignore other events while promotion is pending
                continue # Skip the rest of the event processing loop


            # --- Regular Event Processing (Promotion not pending) ---
            if event.type == pygame.MOUSEWHEEL:
                if tree_panel_rect.collidepoint(current_mouse_pos):
                    # Scroll vertically
                    tree_scroll_y -= event.y * tree_scroll_speed
                    # Clamp scroll position (handled later in draw_game_tree)
                    needs_redraw = True

            # --- Mouse Button Down ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos

                # -- Tree Click/Drag --
                if tree_panel_rect.collidepoint(pos):
                    if event.button == 1: # Left click
                        dragging_tree = True; drag_start_pos = pos
                        # Check for click on a tree node
                        # Convert mouse pos to coordinates relative to the tree render surface
                        tree_surface_x = pos[0] - tree_panel_rect.left + tree_scroll_x
                        tree_surface_y = pos[1] - tree_panel_rect.top + tree_scroll_y
                        clicked_node = None
                        for node, rect in drawn_tree_nodes.items():
                             # Check collision with rects stored relative to tree_render_surface
                             if rect and rect.collidepoint(tree_surface_x, tree_surface_y):
                                 clicked_node = node
                                 break

                        if clicked_node and clicked_node != current_node:
                             # Navigate to clicked node
                             current_node = clicked_node
                             try:
                                board.set_fen(current_node.fen)
                                live_raw_score = current_node.raw_score # Use stored score initially
                                if analysis_manager: analysis_manager.set_target(current_node.fen)
                                reset_transient_state()
                                message = f"Navigated to ply {current_node.get_ply()}"
                                # Clear best move analysis as position changed
                                best_moves_result = None
                                highlighted_engine_move = None
                                current_best_move_index = -1
                                needs_redraw = True
                             except ValueError:
                                message = "Error: Invalid FEN in clicked node."
                                # Optionally revert navigation
                                # current_node = previous_node # Need to store previous node
                                needs_redraw = True


                # -- Board Click/Drag --
                elif BOARD_X <= pos[0] < BOARD_X + BOARD_SIZE and BOARD_Y <= pos[1] < BOARD_Y + BOARD_SIZE:
                    if event.button == 1: # Left click - Start drag or select
                        reset_transient_state()
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            # Can only drag pieces of the current turn
                            if piece and piece.color == board.turn:
                                selected_square = sq
                                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                                img = PIECE_IMAGES.get(piece_key)
                                if img:
                                    # Start dragging
                                    dragging_piece_info = {'square': sq, 'piece': piece, 'img': img, 'pos': pos}
                                    # Get legal moves for highlighting
                                    legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                else: print(f"Error: Could not find image for piece {piece_key}"); reset_transient_state()
                                needs_redraw = True
                            else:
                                 # Clicked empty square or opponent's piece
                                 selected_square = None
                                 legal_moves_for_selected = []
                                 needs_redraw = True

                    elif event.button == 3: # Right click - Cancel selection/drag
                         reset_transient_state(); needs_redraw = True

                # -- Button Clicks --
                elif setup_panel_ui_elements["show_best_move_button_rect"].collidepoint(pos):
                     if not analysis_manager or not current_node or board.is_game_over():
                         message = "Engine unavailable or game over."
                         highlighted_engine_move = None
                         current_best_move_index = -1
                         best_moves_result = None
                         needs_redraw = True
                         continue

                     current_fen = current_node.fen

                     # If we already have results for this FEN, cycle through them
                     if best_moves_result and best_moves_result.get('fen') == current_fen and best_moves_result.get('moves'):
                         moves = best_moves_result['moves']
                         num_moves = len(moves)
                         if num_moves > 0:
                             current_best_move_index = (current_best_move_index + 1) % num_moves
                             highlighted_engine_move = moves[current_best_move_index]['move']
                             san = moves[current_best_move_index]['san']
                             score_str = moves[current_best_move_index]['score_str']
                             message = f"Best {current_best_move_index + 1}/{num_moves}: {san} ({score_str})"
                         else: # Have result object, but no moves in it
                              highlighted_engine_move = None
                              current_best_move_index = -1
                              message = "No moves found previously."
                         needs_redraw = True

                     else:
                         # No results yet, request them
                         message = "Analyzing for best moves..."
                         highlighted_engine_move = None # Clear previous highlight
                         current_best_move_index = -1
                         best_moves_result = None # Clear old results
                         needs_redraw = True
                         pygame.display.flip() # Show "Analyzing..." message immediately

                         analysis_manager.request_best_moves(current_fen) # Use defaults
                         # Result will arrive asynchronously via the results queue

                elif setup_panel_ui_elements["toggle_helpers_button_rect"].collidepoint(pos):
                    helpers_visible = not helpers_visible
                    message = f"Helpers {'ON' if helpers_visible else 'OFF'}"
                    needs_redraw = True

                elif setup_panel_ui_elements["save_game_button_rect"].collidepoint(pos):
                    if current_node:
                         save_game_history(current_node, start_time, start_board_number)
                         message = "Game history saved."
                    else:
                         message = "No game to save."
                    needs_redraw = True


            # --- Mouse Button Up ---
            elif event.type == pygame.MOUSEBUTTONUP:
                 # -- Finish Dragging Piece --
                 if event.button == 1 and dragging_piece_info:
                    pos = event.pos; to_sq = screen_to_square(pos)
                    from_sq = dragging_piece_info['square']
                    piece = dragging_piece_info['piece'] # Get piece info
                    dragging_piece_info = None # Stop dragging visually now
                    move_made_or_pending = False # Track if we proceed

                    if to_sq is not None and from_sq != to_sq:
                        # Check if this is a promotion move first
                        is_promotion = (piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7])

                        if is_promotion:
                            # --- Enter Promotion Pending State ---
                            # Check if the move *without* promotion is legal (basic check)
                            potential_move = chess.Move(from_sq, to_sq)
                            is_legal_target = False
                            for legal_move in board.legal_moves:
                                if legal_move.from_square == from_sq and legal_move.to_square == to_sq:
                                    is_legal_target = True
                                    break

                            if is_legal_target:
                                promotion_pending = True
                                pending_move = {'from_sq': from_sq, 'to_sq': to_sq}
                                promotion_popup_pos = square_to_screen_coords(to_sq) # Center popup on target square
                                promotion_popup_color = board.turn
                                promotion_popup_rect_cache = None # Clear cache until drawn
                                promotion_choice_rects_cache = {}
                                message = "Select promotion piece"
                                needs_redraw = True
                                move_made_or_pending = True
                            else:
                                # Target square isn't legal for the pawn
                                message = "Illegal move"
                                needs_redraw = True

                        else:
                            # --- Regular Move (Not Promotion) ---
                            move = chess.Move(from_sq, to_sq)
                            if move in board.legal_moves:
                                make_move_and_update(move) # Use the helper
                                move_made_or_pending = True # Move was made
                            else:
                                # Illegal move attempt
                                try: from_name = chess.square_name(from_sq); to_name = chess.square_name(to_sq)
                                except: from_name = "?"; to_name = "?"
                                message = f"Illegal move: {from_name}->{to_name}"; needs_redraw = True


                    # Reset selection if no move was made or pending
                    if not move_made_or_pending:
                        selected_square = None; legal_moves_for_selected = []; needs_redraw = True

                 # -- Finish Dragging Tree --
                 if event.button == 1 and dragging_tree:
                     dragging_tree = False
                     # No action needed, position is updated during MOUSEMOTION
                     needs_redraw = True # Redraw once at the end


            # --- Mouse Motion ---
            elif event.type == pygame.MOUSEMOTION:
                # -- Update Dragging Piece Position --
                if dragging_piece_info:
                    dragging_piece_info['pos'] = event.pos; needs_redraw = True
                # -- Update Tree Scroll on Drag --
                if dragging_tree:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    # Move scroll opposite to mouse movement
                    tree_scroll_x -= dx
                    tree_scroll_y -= dy
                    drag_start_pos = event.pos # Update start pos for next motion delta
                    needs_redraw = True
                # -- Update promotion popup highlight if pending --
                if promotion_pending:
                    needs_redraw = True # Need to redraw to show highlight changes


            # --- Key Presses ---
            elif event.type == pygame.KEYDOWN:
                node_changed = False; previous_node = current_node # Store for potential revert

                # --- Tree Navigation ---
                if event.key == pygame.K_LEFT: # Go to parent
                    if current_node and current_node.parent:
                         current_node = current_node.parent; node_changed = True
                         message = f"Back (Ply {current_node.get_ply()})"
                    else: message = "At start of game"
                elif event.key == pygame.K_RIGHT: # Go to first child
                    if current_node and current_node.children:
                         current_node = current_node.children[0]; node_changed = True
                         message = f"Forward (Ply {current_node.get_ply()})"
                    else: message = "End of current line"
                elif event.key == pygame.K_UP: # Go to previous sibling
                    if current_node and current_node.parent:
                        parent = current_node.parent
                        siblings = parent.children
                        try:
                             current_index = siblings.index(current_node)
                             if current_index > 0:
                                 current_node = siblings[current_index - 1]; node_changed = True
                                 message = f"Sibling Up (Ply {current_node.get_ply()})"
                             else: message = "At first sibling"
                        except ValueError: pass # Should not happen if tree is consistent
                    else: message = "No parent/siblings" # Root or single child
                elif event.key == pygame.K_DOWN: # Go to next sibling
                    if current_node and current_node.parent:
                        parent = current_node.parent
                        siblings = parent.children
                        try:
                             current_index = siblings.index(current_node)
                             if current_index < len(siblings) - 1:
                                 current_node = siblings[current_index + 1]; node_changed = True
                                 message = f"Sibling Down (Ply {current_node.get_ply()})"
                             else: message = "At last sibling"
                        except ValueError: pass
                    else: message = "No parent/siblings"

                # --- Handle Node Change ---
                if node_changed:
                    try:
                        board.set_fen(current_node.fen)
                        live_raw_score = current_node.raw_score # Use node's stored score initially
                        if analysis_manager: analysis_manager.set_target(current_node.fen)
                        reset_transient_state(clear_message=False) # Keep navigation message
                        # Clear best move analysis as position changed
                        best_moves_result = None
                        highlighted_engine_move = None
                        current_best_move_index = -1
                        needs_redraw = True
                    except ValueError:
                        message = "Error: Invalid FEN in target node."; print(f"ERROR: Invalid FEN navigation: {current_node.fen}")
                        current_node = previous_node # Revert to previous node
                        needs_redraw = True

                # --- Other Key Actions ---
                elif event.key == pygame.K_a: # Manual Analysis Request (now redundant?)
                     if analysis_manager and current_node:
                         message = "Requesting best moves analysis..."
                         highlighted_engine_move = None
                         current_best_move_index = -1
                         best_moves_result = None
                         needs_redraw = True; pygame.display.flip()
                         analysis_manager.request_best_moves(current_node.fen)
                     else: message = "No engine or node for analysis."; needs_redraw = True

                elif event.key == pygame.K_m: # Toggle Eval Plot (Meter)
                    meter_visible = not meter_visible; message = f"Eval Plot {'ON' if meter_visible else 'OFF'}"
                    needs_redraw = True

                elif event.key == pygame.K_h: # Toggle Helpers
                     helpers_visible = not helpers_visible
                     message = f"Helpers {'ON' if helpers_visible else 'OFF'}"
                     needs_redraw = True

                elif event.key == pygame.K_s: # Save Game
                    if current_node:
                         save_game_history(current_node, start_time, start_board_number)
                         message = "Game history saved."
                    else:
                         message = "No game to save."
                    needs_redraw = True


                elif event.key == pygame.K_ESCAPE: running = False; break


        if not running: continue

        # --- Ensure Board State Matches Current Node (Safety Check) ---
        if not promotion_pending and current_node and board.fen() != current_node.fen:
             try:
                 print(f"Warning: Board desync detected. Forcing board to FEN: {current_node.fen}")
                 board.set_fen(current_node.fen)
                 live_raw_score = current_node.raw_score # Resync score too
                 reset_transient_state(clear_message=False) # Reset interactions
                 needs_redraw = True
             except ValueError:
                 print(f"CRITICAL ERROR: Invalid FEN in current_node! FEN: {current_node.fen}")
                 message = "CRITICAL FEN ERROR!"; needs_redraw = True
                 # Consider stopping or resetting?


        # --- Get Last Move for Highlighting ---
        last_move_displayed = current_node.move if current_node and current_node.parent else None


        # --- Update Dynamic UI Text ---
        # Best Move Button Text
        best_move_button_text = "Show Best"
        if best_moves_result and current_node and best_moves_result.get('fen') == current_node.fen:
            moves = best_moves_result.get('moves')
            error = best_moves_result.get('error')
            if error:
                best_move_button_text = "Analysis Failed"
            elif moves:
                num_found = len(moves)
                best_move_button_text = f"Showing {current_best_move_index + 1}/{num_found}"
            else: # Result exists, but no moves list or empty list
                best_move_button_text = "No moves found"
        elif analysis_manager and current_node and board.is_game_over():
             best_move_button_text = "Game Over"
        elif not analysis_manager:
            best_move_button_text = "Engine Off"


        # --- Redraw Screen (if needed) ---
        if needs_redraw:
            screen.fill(DARK_GREY)

            # Draw UI Panels
            draw_setup_panel(screen, setup_panel_ui_elements, BUTTON_FONT, best_move_button_text, helpers_visible)

            # Draw Board Elements
            draw_coordinates(screen, COORD_FONT)
            draw_board(screen)
            # Don't draw highlights if promotion is pending, except last move? Maybe skip all.
            if not promotion_pending:
                draw_highlights(screen, board, selected_square, legal_moves_for_selected, last_move_displayed, highlighted_engine_move)
            # Draw pieces, handling promotion pending state
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info, promotion_pending)

            # Draw Helpers (if enabled and not promoting)
            if helpers_visible and not promotion_pending:
                # Board Badge for last move quality
                if current_node and current_node.move and current_node.parent and current_node.move_quality:
                    # Draw badge on the TO square of the move that LED TO current_node
                    draw_board_badge(screen, current_node.move.to_square, current_node.move_quality)

                # Eval Bar using the latest live score
                current_wp = score_to_white_percentage(live_raw_score, EVAL_CLAMP_LIMIT) if current_node else 50.0
                draw_eval_bar(screen, current_wp)

                # Eval Plot (if enabled)
                if meter_visible and current_node:
                    path = get_path_to_node(current_node)
                    # Recreate plot only if path changed significantly? For now, always redraw.
                    plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                    if plot_surface:
                        plot_rect = plot_surface.get_rect(topleft=(BOARD_X, plot_panel_y))
                        screen.blit(plot_surface, plot_rect)
                    else: # Draw placeholder if plot creation failed
                        plot_rect = pygame.Rect(BOARD_X, plot_panel_y, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                        pygame.draw.rect(screen, DARK_GREY, plot_rect); pygame.draw.rect(screen, GREY, plot_rect, 1)
                        error_surf = TREE_FONT.render("Plot Error", True, ORANGE); screen.blit(error_surf, error_surf.get_rect(center=plot_rect.center))

            # Draw Game Tree
            draw_game_tree(screen, game_root, current_node, TREE_FONT, helpers_visible)

            # --- Draw Promotion Popup (if active) --- TOP LAYER ---
            if promotion_pending:
                # This function now also updates the global cache variables
                draw_promotion_popup(screen, promotion_popup_pos, promotion_popup_color, current_mouse_pos)


            # Draw Status Messages (Bottom Right)
            status_y_offset = SCREEN_HEIGHT - 10 # Start from bottom
            # Persistent Analysis Error Message
            if analysis_error_message:
                 error_surf = TREE_FONT.render(analysis_error_message, True, ORANGE)
                 error_rect = error_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(error_surf, error_rect); status_y_offset -= error_rect.height + 2
            # General Status Message
            if message:
                 status_surf = TREE_FONT.render(message, True, WHITE)
                 status_rect = status_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset))
                 screen.blit(status_surf, status_rect)


            # Update the display
            pygame.display.flip()
            needs_redraw = False # Reset flag until next change

        # Limit frame rate
        clock.tick(60)

    # --- Cleanup ---
    print("\nExiting Pygame...")
    if analysis_manager:
        analysis_manager.stop() # Stop the analysis thread first
    if engine:
        try:
            time.sleep(0.1) # Short pause before quitting engine
            engine.quit()
            print("Stockfish engine closed.")
        except (AttributeError, BrokenPipeError, Exception) as e:
            print(f"Error closing engine: {e}")
    plt.close('all'); # Close any dangling matplotlib plots
    pygame.quit();
    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chess960 Analysis Board with Async Engine")
    parser.add_argument(
        "board_number",
        type=int,
        nargs='?',  # Make the argument optional
        help="The Chess960 starting position number (0-959). If not provided, a random number will be picked."
    )
    args = parser.parse_args()

    if args.board_number is None:
        args.board_number = random.randint(0, 959)
        print(f"No board number provided. Using random board number: {args.board_number}")
    elif not (0 <= args.board_number <= 959):
        print(f"Error: Board number must be between 0 and 959 (inclusive). You provided: {args.board_number}")
        sys.exit(1)

    # Start the application
    play_chess960_pygame(args.board_number)