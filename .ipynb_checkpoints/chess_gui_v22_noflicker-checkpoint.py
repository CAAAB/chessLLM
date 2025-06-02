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
pygame.mixer.init() # Initialize the mixer
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
pygame.display.set_caption("Chess960 Analysis")

BOARD_X = COORD_PADDING
BOARD_Y = SETUP_PANEL_HEIGHT + COORD_PADDING

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
ORANGE = (255, 165, 0)
DARK_GREY = (60, 60, 60)
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
PROMOTION_POPUP_BG = (90, 90, 90, 220)
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
PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMOTION_POPUP_WIDTH = SQ_SIZE
PROMOTION_POPUP_HEIGHT = SQ_SIZE * len(PROMOTION_PIECES)
PROMOTION_PIECE_SIZE = (int(SQ_SIZE * 0.8), int(SQ_SIZE * 0.8))

# --- Engine Config ---
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon" # ADJUST AS NEEDED
CONTINUOUS_ANALYSIS_TIME_LIMIT = 0.5
BEST_MOVE_ANALYSIS_TIME_LIMIT = 1
NUM_BEST_MOVES_TO_SHOW = 5
EVAL_CLAMP_LIMIT = 800
MATE_SCORE_PLOT_VALUE = EVAL_CLAMP_LIMIT * 1.5

# --- Asset Paths ---
PIECE_IMAGE_PATH = "pieces"
BADGE_IMAGE_PATH = "badges"
SOUND_PATH = "sounds"

# --- Animation ---
ANIMATION_DURATION = 0.20 # seconds for piece move animation
animating_piece_info = None # Stores {'piece_key', 'img', 'start_pos', 'end_pos', 'start_time', 'from_sq', 'to_sq'}

# --- Asset Loading ---
PIECE_IMAGES = {}
BADGE_IMAGES = {}
SOUNDS = {}
all_badges_loaded = True
all_sounds_loaded = True
scaled_promotion_images = {}
scaled_tree_piece_images = {} # Cache for piece images scaled for the tree

# --- Game Tree Drawing State ---
drawn_tree_nodes = {} # Maps Node object -> screen Rect for click detection
tree_scroll_x = 0
tree_scroll_y = 0
max_drawn_tree_x = 0 # Tracks the max width needed for the tree content
max_drawn_tree_y = 0 # Tracks the max height needed for the tree content
tree_render_surface = None # The large surface the whole tree is drawn onto
temp_san_board = chess.Board(chess960=True) # Reusable board for SAN calculation
tree_needs_redraw = True # Controls full tree content redraw

# --- Font Loading ---
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

# --- Asset Loading Functions ---
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
            loaded_images[piece + "_orig"] = img
            scaled_img = pygame.transform.smoothscale(img, (sq_size, sq_size))
            loaded_images[piece] = scaled_img
        except pygame.error as e: print(f"Error loading piece image '{file_path}': {e}"); all_loaded = False
    if not all_loaded: print("Please ensure all 12 piece PNG files exist in the 'pieces' directory."); return None
    print(f"Loaded {len(loaded_images)//2} piece types.")
    return loaded_images

def get_scaled_promotion_image(piece_key, target_size):
    global PIECE_IMAGES, scaled_promotion_images
    if PIECE_IMAGES is None: return None
    cache_key = (piece_key, target_size)
    if cache_key in scaled_promotion_images: return scaled_promotion_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key + "_orig")
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
        for x in range(img.get_width()):
            for y in range(img.get_height()):
                color = img.get_at((x, y))
                if color[3] > 100 and color[0] < 100 and color[1] < 100 and color[2] < 100:
                     img.set_at((x, y), (*target_color, color[3]))
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
                print(f"  - Missing: {quality} ({file_path})"); all_loaded = False; continue
            processed_img = load_and_process_badge_image(file_path, BOARD_BADGE_IMAGE_SIZE, WHITE)
            if processed_img: loaded_badges[quality] = processed_img; print(f"  - Loaded: {quality}")
            else: print(f"  - FAILED processing: {quality}"); all_loaded = False
    if not all_loaded: print("Warning: Some badge images failed to load or process.")
    all_badges_loaded = all_loaded
    return loaded_badges

def load_sounds(path=SOUND_PATH):
    global all_sounds_loaded
    sound_files = {
        "white_move": "move-self.mp3",
        "black_move": "move-opponent.mp3",
        "white_eat": "capture.mp3",
        "black_eat": "capture.mp3",
        "check": "move-check.mp3",
        "checkmate": "move-check.mp3",
        "stalemate": "illegal.mp3",
    }
    loaded_sounds = {}
    all_loaded = True
    print(f"Loading sound effects from '{path}'...")
    if not os.path.isdir(path):
        print(f"Error: Sound directory '{path}' not found. Sounds disabled."); all_loaded = False
    else:
        for key, filename in sound_files.items():
            file_path = os.path.join(path, filename)
            if not os.path.exists(file_path):
                print(f"  - Missing: {key} ({file_path})"); all_loaded = False; continue
            try:
                sound = pygame.mixer.Sound(file_path)
                loaded_sounds[key] = sound; print(f"  - Loaded: {key}")
            except pygame.error as e: print(f"  - FAILED loading {key} ({file_path}): {e}"); all_loaded = False
    if not all_loaded: print("Warning: Some sound effects failed to load.")
    all_sounds_loaded = all_loaded
    return loaded_sounds

def play_sound(sound_key):
    if all_sounds_loaded and sound_key in SOUNDS:
        try: SOUNDS[sound_key].play()
        except pygame.error as e: print(f"Error playing sound '{sound_key}': {e}")

# --- Helper Functions ---
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
    if white_percentage_before is None or white_percentage_after is None: return None
    if turn_before_move == chess.WHITE:
        eval_before_pov = white_percentage_before / 100.0; eval_after_pov = white_percentage_after / 100.0
    else:
        eval_before_pov = (100.0 - white_percentage_before) / 100.0; eval_after_pov = (100.0 - white_percentage_after) / 100.0
    eval_drop = max(0.0, eval_before_pov - eval_after_pov); eval_drop_percent = eval_drop * 100
    if eval_drop_percent <= 2: return "Best"
    elif eval_drop_percent <= 5: return "Excellent"
    elif eval_drop_percent <= 10: return "Good"
    elif eval_drop_percent <= 20: return "Inaccuracy"
    elif eval_drop_percent <= 35: return "Mistake"
    else: return "Blunder"

def format_score(score, turn):
    if score is None: return "N/A"
    pov_score = score.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate(); return f"Mate in {abs(mate_in)}" if mate_in is not None else "Mate"
    else:
        cp = pov_score.score(mate_score=MATE_SCORE_PLOT_VALUE * 2); return f"{cp / 100.0:+.2f}" if cp is not None else "N/A (No CP)"

def get_sound_key_for_move(board_before, board_after, move):
    if not move: return None
    turn_before_move = board_before.turn
    is_capture_before = board_before.is_capture(move) or board_before.is_en_passant(move)
    if board_after.is_checkmate(): return "checkmate"
    elif board_after.is_stalemate(): return "stalemate"
    elif board_after.is_check(): return "check"
    elif is_capture_before: return "white_eat" if turn_before_move == chess.WHITE else "black_eat"
    else: return "white_move" if turn_before_move == chess.WHITE else "black_move"

def square_to_screen_coords(square):
    rank = chess.square_rank(square); file = chess.square_file(square)
    screen_x = BOARD_X + file * SQ_SIZE + SQ_SIZE // 2
    screen_y = BOARD_Y + (7 - rank) * SQ_SIZE + SQ_SIZE // 2
    return screen_x, screen_y

def screen_to_square(pos):
    x, y = pos
    if x < BOARD_X or x >= BOARD_X + BOARD_SIZE or y < BOARD_Y or y >= BOARD_Y + BOARD_SIZE: return None
    file = (x - BOARD_X) // SQ_SIZE; rank = 7 - ((y - BOARD_Y) // SQ_SIZE)
    if 0 <= file <= 7 and 0 <= rank <= 7: return chess.square(file, rank)
    else: return None

def ease_in_out_sine(t):
    return 0.5 * (1 - math.cos(math.pi * t))

# --- Game Tree Node ---
class GameNode:
    def __init__(self, fen, move=None, parent=None, raw_score=None):
        self.fen = fen; self.move = move; self.parent = parent
        self.children = []; self.raw_score = raw_score
        self.white_percentage = score_to_white_percentage(raw_score, EVAL_CLAMP_LIMIT)
        self._san_cache = None; self.x = 0; self.y = 0
        self.screen_rect = None; self.move_quality = None

    def add_child(self, child_node): self.children.append(child_node)
    def get_ply(self): 
        count = 0; node = self; 
        while node.parent: 
            count += 1; 
            node = node.parent; 
        return count

    def get_san(self, board_at_parent):
        if self._san_cache is not None: return self._san_cache
        if not self.move or not self.parent: return "root"
        try: 
            san = board_at_parent.san(self.move); 
            self._san_cache = san; 
            return san
        except Exception: 
            return self.move.uci()

    def calculate_and_set_move_quality(self):
        if not self.parent or self.parent.white_percentage is None or self.white_percentage is None: 
            self.move_quality = None; 
            return
        wp_before = self.parent.white_percentage; wp_after = self.white_percentage
        try: 
            parent_board = chess.Board(self.parent.fen, chess960=True); 
            turn_before_move = parent_board.turn
        except ValueError: 
            self.move_quality = None; print(f"Warning: Could not create board from parent FEN: {self.parent.fen}"); return
        self.move_quality = classify_move_quality(wp_before, wp_after, turn_before_move)

    def __str__(self):
        move_str = self.move.uci() if self.move else "root"
        score_str = format_score(self.raw_score, chess.WHITE if self.get_ply() % 2 == 0 else chess.BLACK)
        return f"Node(Ply:{self.get_ply()}, Move:{move_str}, Score:{score_str}, Children:{len(self.children)})"

# --- Asynchronous Analysis Manager (Unchanged from previous correct version) ---
class AnalysisManager:
    def __init__(self, engine):
        self.engine = engine; self._lock = threading.Lock(); self._target_fen = None
        self._request_type = 'idle'; self._best_moves_params = (NUM_BEST_MOVES_TO_SHOW, BEST_MOVE_ANALYSIS_TIME_LIMIT)
        self._results_queue = queue.Queue(); self._stop_event = threading.Event()
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._last_continuous_result = None
    def start(self):
        if not self.engine: print("Analysis Manager: No engine provided."); return
        print("Analysis Manager: Starting analysis thread."); self._analysis_thread.start()
    def stop(self):
        print("Analysis Manager: Signalling analysis thread to stop."); self._stop_event.set()
        self._analysis_thread.join(timeout=1.0)
        if self._analysis_thread.is_alive(): print("Analysis Manager: Warning - Analysis thread did not stop.")
        else: print("Analysis Manager: Analysis thread stopped.")
    def set_target(self, fen):
        if not self.engine: return
        with self._lock:
            if self._target_fen != fen or self._request_type != 'continuous':
                self._target_fen = fen; self._request_type = 'continuous'
                if self._target_fen != fen: self._last_continuous_result = None
    def request_best_moves(self, fen, num_moves=NUM_BEST_MOVES_TO_SHOW, time_limit=BEST_MOVE_ANALYSIS_TIME_LIMIT):
        if not self.engine: return
        with self._lock:
            print(f"Analysis Manager: Requesting best {num_moves} moves for FEN: {fen} (Time: {time_limit}s)")
            self._target_fen = fen; self._request_type = 'best_moves'
            self._best_moves_params = (num_moves, time_limit); self._last_continuous_result = None
    def get_latest_result(self):
        try:
            result = self._results_queue.get_nowait()
            if result and result.get('type') == 'continuous': self._last_continuous_result = result
            return result
        except queue.Empty: return None
    def get_last_continuous_score(self, fen):
         if self._last_continuous_result and self._last_continuous_result.get('fen') == fen: return self._last_continuous_result.get('score')
         return None
    def _analysis_loop(self):
        last_analyzed_fen_continuous = None; last_analyzed_fen_best_moves = None
        last_request_type = None; last_best_move_params = None; best_moves_analysis_handle = None
        while not self._stop_event.is_set():
            target_fen = None; request_type = 'idle'; best_moves_params = None
            with self._lock: target_fen = self._target_fen; request_type = self._request_type; best_moves_params = self._best_moves_params
            if best_moves_analysis_handle and \
               (request_type != 'best_moves' or target_fen != last_analyzed_fen_best_moves or best_moves_params != last_best_move_params):
                try: best_moves_analysis_handle.stop()
                except Exception: pass
                best_moves_analysis_handle = None; last_analyzed_fen_best_moves = None
            if not target_fen or request_type == 'idle': last_analyzed_fen_continuous = None; time.sleep(0.1); continue
            try:
                board = chess.Board(target_fen, chess960=True)
                if request_type == 'continuous':
                    if target_fen != last_analyzed_fen_continuous:
                        limit = chess.engine.Limit(time=CONTINUOUS_ANALYSIS_TIME_LIMIT)
                        try:
                            info = self.engine.analyse(board, limit, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
                            score = info.get("score"); pv = info.get("pv"); best_move = pv[0] if pv else None
                            result = {'type': 'continuous', 'fen': target_fen, 'score': score, 'best_move': best_move, 'error': None}
                            self._results_queue.put(result); last_analyzed_fen_continuous = target_fen
                        except (chess.engine.EngineError, BrokenPipeError, AttributeError, ValueError, chess.engine.EngineTerminatedError) as e:
                            print(f"Analysis Manager: Continuous analysis error: {e}")
                            self._results_queue.put({'type': 'continuous', 'fen': target_fen, 'score': None, 'best_move': None, 'error': str(e)})
                            last_analyzed_fen_continuous = None; time.sleep(0.5)
                    else: time.sleep(0.05)
                elif request_type == 'best_moves':
                    if not best_moves_analysis_handle or target_fen != last_analyzed_fen_best_moves or best_moves_params != last_best_move_params:
                        limit = chess.engine.Limit(time=best_moves_params[1]); num_moves = best_moves_params[0]
                        try:
                            best_moves_analysis_handle = self.engine.analysis(board, limit, multipv=num_moves, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
                            last_analyzed_fen_best_moves = target_fen; last_request_type = request_type; last_best_move_params = best_moves_params
                            moves_info = []
                            for info in best_moves_analysis_handle:
                                if self._stop_event.is_set(): break
                                if "pv" in info and info["pv"] and "score" in info:
                                    move = info["pv"][0]; score = info.get("score"); score_str = format_score(score, board.turn)
                                    try: move_san = board.san(move)
                                    except Exception: move_san = move.uci()
                                    if not any(m['move'] == move for m in moves_info):
                                        moves_info.append({"move": move, "san": move_san, "score_str": score_str, "score_obj": score})
                            if not self._stop_event.is_set():
                                self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': moves_info, 'error': None})
                        except (chess.engine.EngineError, BrokenPipeError, AttributeError, ValueError, chess.engine.EngineTerminatedError) as e:
                            print(f"Analysis Manager: Best moves analysis error: {e}")
                            self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': [], 'error': str(e)})
                        except Exception as e:
                              print(f"Analysis Manager: Unexpected best moves error: {e}")
                              self._results_queue.put({'type': 'best_moves', 'fen': target_fen, 'moves': [], 'error': f"Unexpected: {e}"})
                        finally:
                            best_moves_analysis_handle = None; last_analyzed_fen_best_moves = None
                            with self._lock:
                                if self._target_fen == target_fen and self._request_type == 'best_moves':
                                    self._request_type = 'continuous'; last_analyzed_fen_continuous = None
            except ValueError as e:
                print(f"Analysis Manager: Invalid FEN: {target_fen} - Error: {e}")
                self._results_queue.put({'type': request_type, 'fen': target_fen, 'score': None, 'best_move': None, 'moves': [], 'error': f"Invalid FEN: {e}"})
                with self._lock: self._target_fen = None; self._request_type = 'idle'
                last_analyzed_fen_continuous = None; last_analyzed_fen_best_moves = None; time.sleep(0.5)
            except Exception as e:
                 print(f"Analysis Manager: Unexpected error in analysis loop: {e}"); import traceback; traceback.print_exc()
                 with self._lock: self._request_type = 'idle'
                 last_analyzed_fen_continuous = None; last_analyzed_fen_best_moves = None
                 if best_moves_analysis_handle: 
                     try: 
                         best_moves_analysis_handle.stop(); 
                     except: 
                         pass
                 best_moves_analysis_handle = None; time.sleep(1)
            time.sleep(0.01)
        if best_moves_analysis_handle: 
            print("Analysis Manager: Stopping final analysis."); 
            try: 
                best_moves_analysis_handle.stop(); 
            except Exception: 
                pass
        print("Analysis Manager: Analysis loop finished.")

# --- Drawing Functions ---
def get_path_to_node(node): 
    path = []; current = node; 
    while current: 
        path.append(current); current = current.parent; 
    return path[::-1]

def draw_board(surface):
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ_COLOR if (r + c) % 2 == 0 else DARK_SQ_COLOR
            pygame.draw.rect(surface, color, (BOARD_X + c * SQ_SIZE, BOARD_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_coordinates(surface, font):
    for i in range(8): text = chr(ord('a') + i); text_surf = font.render(text, True, COORD_COLOR); text_rect = text_surf.get_rect(center=(BOARD_X + i * SQ_SIZE + SQ_SIZE // 2, BOARD_Y + BOARD_SIZE + COORD_PADDING // 2)); surface.blit(text_surf, text_rect)
    for i in range(8): text = str(8 - i); text_surf = font.render(text, True, COORD_COLOR); text_rect = text_surf.get_rect(center=(BOARD_X - COORD_PADDING // 2, BOARD_Y + i * SQ_SIZE + SQ_SIZE // 2)); surface.blit(text_surf, text_rect)

def draw_pieces(surface, board, piece_images, dragging_piece_info, promotion_pending=False, animating_info=None):
    if piece_images is None: return
    animating_from_sq = animating_info['from_sq'] if animating_info else None
    animating_to_sq = animating_info['to_sq'] if animating_info else None

    for square in chess.SQUARES:
        # Skip drawing the piece at its *start* square during animation
        if animating_from_sq is not None and square == animating_from_sq: continue
        # Skip drawing the piece at its *end* square during animation (it's drawn by draw_animating_piece)
        if animating_to_sq is not None and square == animating_to_sq: continue

        piece = board.piece_at(square)
        if piece:
            if dragging_piece_info and square == dragging_piece_info['square']: continue
            if promotion_pending and pending_move and square == pending_move['from_sq']: continue

            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
            img = piece_images.get(piece_key)
            if img:
                rank = chess.square_rank(square); file = chess.square_file(square)
                screen_x = BOARD_X + file * SQ_SIZE; screen_y = BOARD_Y + (7 - rank) * SQ_SIZE
                surface.blit(img, (screen_x, screen_y))

    if dragging_piece_info and dragging_piece_info['img']:
        img = dragging_piece_info['img']; img_rect = img.get_rect(center=dragging_piece_info['pos']); surface.blit(img, img_rect)

def draw_animating_piece(surface, animating_info):
    global animating_piece_info
    if not animating_info: return
    current_time = time.time(); elapsed_time = current_time - animating_info['start_time']
    progress = min(1.0, elapsed_time / ANIMATION_DURATION); eased_progress = ease_in_out_sine(progress)
    start_x, start_y = animating_info['start_pos']; end_x, end_y = animating_info['end_pos']
    current_x = start_x + (end_x - start_x) * eased_progress; current_y = start_y + (end_y - start_y) * eased_progress
    img = animating_info['img']; img_rect = img.get_rect(center=(int(current_x), int(current_y)))
    surface.blit(img, img_rect.topleft)
    if progress >= 1.0: animating_piece_info = None

def draw_highlights(surface, board, selected_square, legal_moves_for_selected, last_move, engine_move_to_highlight):
    def highlight_squares(squares, color):
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA); s.fill(color)
        for sq in squares: rank = chess.square_rank(sq); file = chess.square_file(sq); highlight_rect = pygame.Rect(BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE); surface.blit(s, highlight_rect.topleft)
    def highlight_legal_moves(moves, color):
        for move in moves:
            dest_sq = move.to_square; rank = chess.square_rank(dest_sq); file = chess.square_file(dest_sq); s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            is_capture = board.is_capture(move) or board.is_en_passant(move); center_x, center_y = SQ_SIZE // 2, SQ_SIZE // 2; radius = SQ_SIZE // 6
            if is_capture: pygame.draw.circle(s, color, (center_x, center_y), radius + 3, 3)
            else: pygame.draw.circle(s, color, (center_x, center_y), radius)
            surface.blit(s, (BOARD_X + file * SQ_SIZE, BOARD_Y + (7 - rank) * SQ_SIZE))
    if engine_move_to_highlight: highlight_squares([engine_move_to_highlight.from_square, engine_move_to_highlight.to_square], ENGINE_MOVE_HIGHLIGHT_COLOR)
    if last_move: highlight_squares([last_move.from_square, last_move.to_square], LAST_MOVE_HIGHLIGHT_COLOR)
    if selected_square is not None: highlight_squares([selected_square], HIGHLIGHT_COLOR)
    if legal_moves_for_selected: highlight_legal_moves(legal_moves_for_selected, POSSIBLE_MOVE_COLOR)

def draw_board_badge(surface, square, quality):
    if not all_badges_loaded: return
    badge_image = BADGE_IMAGES.get(quality); badge_color = TREE_MOVE_QUALITY_COLORS.get(quality)
    if quality is None or badge_image is None or badge_color is None: return
    rank = chess.square_rank(square); file = chess.square_file(square); square_base_x = BOARD_X + file * SQ_SIZE; square_base_y = BOARD_Y + (7 - rank) * SQ_SIZE
    center_x = square_base_x + BOARD_BADGE_OFFSET_X; center_y = square_base_y + BOARD_BADGE_OFFSET_Y
    pygame.draw.circle(surface, badge_color, (center_x, center_y), BOARD_BADGE_RADIUS); pygame.draw.circle(surface, BOARD_BADGE_OUTLINE_COLOR, (center_x, center_y), BOARD_BADGE_RADIUS, 1)
    badge_rect = badge_image.get_rect(center=(center_x, center_y)); surface.blit(badge_image, badge_rect.topleft)

def draw_eval_bar(surface, white_percentage):
    bar_height = BOARD_SIZE; bar_x = BOARD_X + BOARD_SIZE + (SIDE_PANEL_WIDTH - EVAL_BAR_WIDTH_PX) // 2; bar_y = BOARD_Y
    white_percentage = 50.0 if white_percentage is None else white_percentage; white_height = int(bar_height * (white_percentage / 100.0)); black_height = bar_height - white_height
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, black_height)); pygame.draw.rect(surface, WHITE, (bar_x, bar_y + black_height, EVAL_BAR_WIDTH_PX, white_height))
    pygame.draw.rect(surface, GREY, (bar_x, bar_y, EVAL_BAR_WIDTH_PX, bar_height), 1)

def create_eval_plot_surface(node_path, plot_width_px, plot_height_px):
    if not node_path or len(node_path) < 1:
        placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY)
        pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    plies = [node.get_ply() for node in node_path]
    percentages = [(node.white_percentage if node.white_percentage is not None else 50.0) for node in node_path]
    dark_grey_mpl=tuple(c/255.0 for c in DARK_GREY); orange_mpl=tuple(c/255.0 for c in ORANGE); grey_mpl=tuple(c/255.0 for c in GREY)
    white_mpl=tuple(c/255.0 for c in WHITE); black_mpl=tuple(c/255.0 for c in BLACK); darker_grey_mpl=tuple(c/255. for c in [30]*3)
    fig, ax = plt.subplots(figsize=(plot_width_px/80, plot_height_px/80), dpi=80)
    fig.patch.set_facecolor(dark_grey_mpl); ax.set_facecolor(darker_grey_mpl)
    if len(plies) > 1: ax.fill_between(plies, percentages, color=white_mpl, alpha=1); ax.plot(plies, percentages, color=white_mpl, marker=None, linestyle='-', linewidth=1.5)
    elif len(plies) == 1: ax.plot(plies[0], percentages[0], color=white_mpl, marker='o', markersize=3)
    ax.axhline(50, color=orange_mpl, linestyle='-', linewidth=1.5)
    ax.set_xlim(0, max(max(plies), 1) if plies else 1); ax.set_ylim(0, 100)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(""); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, linestyle=':', linewidth=0.5, color=grey_mpl, alpha=0.4)
    for spine in ax.spines.values(): spine.set_color(grey_mpl); spine.set_linewidth(0.5)
    plt.tight_layout(pad=0.1); buf = io.BytesIO()
    try: plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, bbox_inches='tight', pad_inches=0.05)
    except Exception as e: print(f"Error saving plot: {e}"); plt.close(fig); buf.close(); placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY); pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally: plt.close(fig)
    buf.seek(0); plot_surface = None
    try: plot_surface = pygame.image.load(buf, 'png').convert()
    except pygame.error as e: print(f"Error loading plot image from buffer: {e}"); placeholder = pygame.Surface((plot_width_px, plot_height_px)); placeholder.fill(DARK_GREY); pygame.draw.rect(placeholder, GREY, placeholder.get_rect(), 1); return placeholder
    finally: buf.close()
    return plot_surface

def get_scaled_tree_image(piece_key, target_size):
    global PIECE_IMAGES, scaled_tree_piece_images
    if PIECE_IMAGES is None: return None
    cache_key = (piece_key, target_size)
    if cache_key in scaled_tree_piece_images: return scaled_tree_piece_images[cache_key]
    original_img = PIECE_IMAGES.get(piece_key + "_orig")
    if original_img:
        try: scaled_img = pygame.transform.smoothscale(original_img, (target_size, target_size)); scaled_tree_piece_images[cache_key] = scaled_img; return scaled_img
        except Exception as e: print(f"Error scaling tree image for {piece_key}: {e}"); return None
    return None

def layout_and_draw_tree_recursive(surface, node, x, y_center, level, font, current_node, helpers_visible):
    global drawn_tree_nodes, max_drawn_tree_x, max_drawn_tree_y, PIECE_IMAGES, temp_san_board
    node.x = x; node.y = y_center; piece_img = None; is_root = not node.parent
    if node.move and node.parent:
        try:
            temp_san_board.set_fen(node.parent.fen); moved_piece = temp_san_board.piece_at(node.move.from_square)
            if moved_piece: piece_key = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper(); piece_img = get_scaled_tree_image(piece_key, TREE_PIECE_SIZE)
        except ValueError: pass
        except Exception as e: print(f"Warning: Error getting piece for tree node: {e}")
    child_y_positions = []; child_subtree_heights = []; total_child_height_estimate = 0; child_x = x + HORIZ_SPACING
    if node.children:
        num_children = len(node.children); total_child_height_estimate = (num_children -1) * VERT_SPACING; current_child_y_start = y_center - total_child_height_estimate / 2; next_child_y = current_child_y_start
        for i, child in enumerate(node.children):
            child_center_y, child_subtree_height = layout_and_draw_tree_recursive(surface, child, child_x, next_child_y, level + 1, font, current_node, helpers_visible)
            child_y_positions.append(child_center_y); child_subtree_heights.append(child_subtree_height); spacing = max(VERT_SPACING, child_subtree_height / 2 + VERT_SPACING / 2 if child_subtree_height > 0 else VERT_SPACING)
            next_child_y = child_center_y + child_subtree_height / 2 + spacing / 2
    node_rect = None
    if piece_img: img_rect = piece_img.get_rect(center=(int(node.x), int(node.y))); surface.blit(piece_img, img_rect.topleft); node_rect = img_rect
    elif is_root: radius = 10; pygame.draw.circle(surface, TREE_NODE_ROOT_COLOR, (int(node.x), int(node.y)), radius); node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    else: radius = 3; pygame.draw.circle(surface, GREY, (int(node.x), int(node.y)), radius); node_rect = pygame.Rect(node.x - radius, node.y - radius, radius*2, radius*2)
    if node_rect: max_drawn_tree_x = max(max_drawn_tree_x, node_rect.right)
    if node == current_node and node_rect: pygame.draw.rect(surface, TREE_NODE_CURRENT_OUTLINE, node_rect.inflate(4, 4), 1)
    if node_rect and node.move_quality and node.move_quality in TREE_MOVE_QUALITY_COLORS and helpers_visible:
        badge_color = TREE_MOVE_QUALITY_COLORS[node.move_quality]; badge_center_x = node_rect.right - TREE_BADGE_RADIUS - 1; badge_center_y = node_rect.bottom - TREE_BADGE_RADIUS - 1
        pygame.draw.circle(surface, badge_color, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS); pygame.draw.circle(surface, DARK_GREY, (badge_center_x, badge_center_y), TREE_BADGE_RADIUS, 1)
    move_text = ""; text_rect = None
    if node.parent:
        try: temp_san_board.set_fen(node.parent.fen); move_text = node.get_san(temp_san_board)
        except ValueError: move_text = node.move.uci() + "?"
        except Exception: move_text = node.move.uci()
    if move_text and node_rect: text_surf = font.render(move_text, True, TREE_TEXT_COLOR); text_rect = text_surf.get_rect(midleft=(node_rect.right + TEXT_OFFSET_X, node_rect.centery + TEXT_OFFSET_Y)); surface.blit(text_surf, text_rect); max_drawn_tree_x = max(max_drawn_tree_x, text_rect.right)
    clickable_rect = node_rect.copy() if node_rect else pygame.Rect(node.x, node.y, 0, 0)
    if text_rect: clickable_rect.width = max(clickable_rect.width, text_rect.right - clickable_rect.left)
    node.screen_rect = clickable_rect; drawn_tree_nodes[node] = node.screen_rect
    if node_rect and node.children:
        for i, child in enumerate(node.children):
            if hasattr(child, 'x') and hasattr(child, 'y'):
               child_visual_rect = drawn_tree_nodes.get(child, pygame.Rect(child.x-1, child.y-1,2,2)); start_pos = (node_rect.right, node_rect.centery); end_pos = (child_visual_rect.left, child_visual_rect.centery); pygame.draw.line(surface, TREE_LINE_COLOR, start_pos, end_pos, 1)
    my_height = node_rect.height if node_rect else TREE_PIECE_SIZE; subtree_total_height = 0
    if child_y_positions: min_child_y_center = min(child_y_positions); max_child_y_center = max(child_y_positions); est_top = min_child_y_center - (max(child_subtree_heights)/2 if child_subtree_heights else 0); est_bottom = max_child_y_center + (max(child_subtree_heights)/2 if child_subtree_heights else 0); subtree_total_height = max(0, est_bottom - est_top)
    node_bottom_extent = node.y + max(my_height, subtree_total_height) / 2; max_drawn_tree_y = max(max_drawn_tree_y, node_bottom_extent)
    return node.y, max(my_height, subtree_total_height)

def draw_game_tree(surface, root_node, current_node, font, helpers_visible):
    global drawn_tree_nodes, tree_scroll_x, tree_scroll_y, max_drawn_tree_x, max_drawn_tree_y, tree_render_surface, tree_needs_redraw
    plot_panel_y = BOARD_Y + BOARD_SIZE; tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT; tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)
    min_width = max(INITIAL_TREE_SURFACE_WIDTH, tree_panel_rect.width); min_height = max(INITIAL_TREE_SURFACE_HEIGHT, tree_panel_rect.height)
    estimated_required_width = max(min_width, int(max_drawn_tree_x + 2 * HORIZ_SPACING)); estimated_required_height = max(min_height, int(max_drawn_tree_y + 2 * VERT_SPACING))
    if tree_render_surface is None or tree_render_surface.get_width() < estimated_required_width or tree_render_surface.get_height() < estimated_required_height:
        new_width = estimated_required_width; new_height = estimated_required_height
        try: tree_render_surface = pygame.Surface((new_width, new_height)); tree_needs_redraw = True # Force redraw after resize
        except (pygame.error, MemoryError, ValueError) as e:
            print(f"Error creating/resizing tree surface ({new_width}x{new_height}): {e}")
            if tree_render_surface: tree_needs_redraw = False # Keep old surf, don't force redraw
            else: tree_render_surface = None
            pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect); pygame.draw.rect(surface, GREY, tree_panel_rect, 1); error_msg = f"Tree too large ({new_width}x{new_height})! Err: {e}"; error_surf = font.render(error_msg[:80], True, ORANGE); surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center)); return
    if tree_render_surface is None: pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect); pygame.draw.rect(surface, GREY, tree_panel_rect, 1); error_surf = font.render("Tree Surface Error", True, ORANGE); surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center)); return

    if tree_needs_redraw:
        drawn_tree_nodes.clear(); max_drawn_tree_x = 0; max_drawn_tree_y = 0; tree_render_surface.fill(TREE_BG_COLOR)
        if root_node: start_x = 15 + TREE_PIECE_SIZE // 2; start_y = tree_render_surface.get_height() // 2; layout_and_draw_tree_recursive(tree_render_surface, root_node, start_x, start_y, 0, font, current_node, helpers_visible)
        tree_needs_redraw = False
    total_tree_width = max_drawn_tree_x + HORIZ_SPACING; total_tree_height = max_drawn_tree_y + VERT_SPACING * 2
    max_scroll_x = max(0, total_tree_width - tree_panel_rect.width); max_scroll_y = max(0, total_tree_height - tree_panel_rect.height)
    scroll_margin_x = HORIZ_SPACING * 1.5; scroll_margin_y = VERT_SPACING * 3
    if current_node and current_node.screen_rect:
        node_rect_on_surface = current_node.screen_rect
        if node_rect_on_surface.right > tree_scroll_x + tree_panel_rect.width - scroll_margin_x: tree_scroll_x = node_rect_on_surface.right - tree_panel_rect.width + scroll_margin_x
        elif node_rect_on_surface.left < tree_scroll_x + scroll_margin_x: tree_scroll_x = node_rect_on_surface.left - scroll_margin_x
        if node_rect_on_surface.bottom > tree_scroll_y + tree_panel_rect.height - scroll_margin_y: tree_scroll_y = node_rect_on_surface.bottom - tree_panel_rect.height + scroll_margin_y
        elif node_rect_on_surface.top < tree_scroll_y + scroll_margin_y: tree_scroll_y = node_rect_on_surface.top - scroll_margin_y
    tree_scroll_x = max(0, min(tree_scroll_x, max_scroll_x)); tree_scroll_y = max(0, min(tree_scroll_y, max_scroll_y))
    blit_width = min(tree_panel_rect.width, total_tree_width - tree_scroll_x); blit_height = min(tree_panel_rect.height, total_tree_height - tree_scroll_y)
    if blit_width > 0 and blit_height > 0:
        source_rect = pygame.Rect(tree_scroll_x, tree_scroll_y, blit_width, blit_height)
        try: surface.blit(tree_render_surface, tree_panel_rect.topleft, area=source_rect)
        except pygame.error as e: print(f"Error blitting tree view: {e}"); pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect); pygame.draw.rect(surface, GREY, tree_panel_rect, 1); error_surf = font.render("Tree Blit Error", True, ORANGE); surface.blit(error_surf, error_surf.get_rect(center=tree_panel_rect.center))
    else: pygame.draw.rect(surface, TREE_BG_COLOR, tree_panel_rect)
    scrollbar_thickness = 7
    if total_tree_width > tree_panel_rect.width:
        ratio_visible = tree_panel_rect.width / total_tree_width; scrollbar_width = max(15, tree_panel_rect.width * ratio_visible); scrollbar_x_ratio = tree_scroll_x / max_scroll_x if max_scroll_x > 0 else 0; scrollbar_x = tree_panel_rect.left + (tree_panel_rect.width - scrollbar_width) * scrollbar_x_ratio; scrollbar_rect = pygame.Rect(scrollbar_x, tree_panel_rect.bottom - scrollbar_thickness - 1, scrollbar_width, scrollbar_thickness); pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)
    if total_tree_height > tree_panel_rect.height:
        ratio_visible = tree_panel_rect.height / total_tree_height; scrollbar_height = max(15, tree_panel_rect.height * ratio_visible); scrollbar_y_ratio = tree_scroll_y / max_scroll_y if max_scroll_y > 0 else 0; scrollbar_y = tree_panel_rect.top + (tree_panel_rect.height - scrollbar_height) * scrollbar_y_ratio; scrollbar_rect = pygame.Rect(tree_panel_rect.right - scrollbar_thickness - 1, scrollbar_y, scrollbar_thickness, scrollbar_height); pygame.draw.rect(surface, GREY, scrollbar_rect, border_radius=3)

def draw_setup_panel(surface, ui_elements, button_font, best_move_button_text, helpers_visible):
    panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SETUP_PANEL_HEIGHT); pygame.draw.rect(surface, DARK_GREY, panel_rect)
    buttons = [ ("show_best_move_button_rect", best_move_button_text, False), ("toggle_helpers_button_rect", f"Helpers: {'ON' if helpers_visible else 'OFF'}", helpers_visible), ("save_game_button_rect", "Save Game", False), ]
    for rect_key, text, highlighted in buttons:
        button_rect = ui_elements[rect_key]; bg_color = ORANGE if highlighted else BUTTON_COLOR; pygame.draw.rect(surface, bg_color, button_rect, border_radius=3); pygame.draw.rect(surface, GREY, button_rect, 1, border_radius=3)
        text_surf = button_font.render(text, True, BUTTON_TEXT_COLOR); text_rect = text_surf.get_rect(center=button_rect.center); surface.blit(text_surf, text_rect)

def save_game_history(current_node, start_time, start_board_number):
    if not current_node or not current_node.parent: 
        print("No game history to save."); 
        return
    games_dir = "./games"; 
    try: 
        os.makedirs(games_dir, exist_ok=True)
    except OSError as e: 
        print(f"Error creating save directory '{games_dir}': {e}"); 
        return
    timestamp = start_time.strftime("%Y%m%d_%H%M%S"); filename = os.path.join(games_dir, f"game_{timestamp}_pos{start_board_number}.pgn")
    node_path = get_path_to_node(current_node); moves_san = []; move_num = 1; is_white_move = True; temp_board_for_san = chess.Board(chess960=True)
    try:
        initial_fen = node_path[0].fen; temp_board_for_san.set_fen(initial_fen); is_white_move = (temp_board_for_san.turn == chess.WHITE)
        for i in range(len(node_path) - 1):
            parent_node = node_path[i]; child_node = node_path[i+1]; move = child_node.move
            if move:
                temp_board_for_san.set_fen(parent_node.fen); san = temp_board_for_san.san(move); move_prefix = ""
                if is_white_move: move_prefix = f"{move_num}. "
                elif i == 0 and not is_white_move: move_prefix = f"{move_num}... "
                moves_san.append(move_prefix + san)
                if not is_white_move: move_num += 1
                is_white_move = not is_white_move
            else: print(f"Warning: Node at index {i+1} has no move.")
    except (ValueError, IndexError, AttributeError) as e: print(f"Error generating SAN for PGN: {e}."); return
    start_fen_for_header = node_path[0].fen if node_path else chess.STARTING_BOARD_FEN
    pgn_header = f"""[Event "Chess960 Analysis"]\n[Site "Local"]\n[Date "{start_time.strftime('%Y.%m.%d')}"]\n[Round "-"]\n[White "Player"]\n[Black "Player"]\n[Result "*"]\n[FEN "{start_fen_for_header}"]\n[SetUp "1"]\n[Variant "Chess960"]\n\n"""
    pgn_moves = ""; ply_count = 0; temp_board_check_turn = chess.Board(start_fen_for_header, chess960=True); pgn_start_turn_is_white = (temp_board_check_turn.turn == chess.WHITE)
    for san_entry in moves_san: 
        pgn_moves += san_entry + " "; 
        ply_count += 1; 
        if ply_count % 10 == 0 : 
            pgn_moves += "\n"
    try:
        with open(filename, "w") as file: 
            file.write(pgn_header); file.write(pgn_moves.strip()); file.write(" *"); print(f"Game history saved to {filename}")
    except IOError as e: 
        print(f"Error writing PGN to file '{filename}': {e}")

def draw_promotion_popup(surface, pos, color_to_promote, mouse_pos):
    global promotion_popup_rect_cache, promotion_choice_rects_cache
    popup_x = pos[0] - PROMOTION_POPUP_WIDTH // 2; popup_y = pos[1] - PROMOTION_POPUP_HEIGHT // 2; popup_x = max(0, min(popup_x, SCREEN_WIDTH - PROMOTION_POPUP_WIDTH)); popup_y = max(0, min(popup_y, SCREEN_HEIGHT - PROMOTION_POPUP_HEIGHT))
    overall_rect = pygame.Rect(popup_x, popup_y, PROMOTION_POPUP_WIDTH, PROMOTION_POPUP_HEIGHT); popup_surf = pygame.Surface(overall_rect.size, pygame.SRCALPHA); popup_surf.fill(PROMOTION_POPUP_BG); surface.blit(popup_surf, overall_rect.topleft); pygame.draw.rect(surface, PROMOTION_POPUP_BORDER, overall_rect, 1)
    choice_rects = {}; color_prefix = 'w' if color_to_promote == chess.WHITE else 'b'
    for i, piece_type in enumerate(PROMOTION_PIECES):
        piece_key = color_prefix + chess.piece_symbol(piece_type).upper(); img = get_scaled_promotion_image(piece_key, PROMOTION_PIECE_SIZE)
        if img:
            choice_y = overall_rect.top + i * SQ_SIZE; choice_rect = pygame.Rect(overall_rect.left, choice_y, SQ_SIZE, SQ_SIZE); choice_rects[piece_type] = choice_rect; img_rect = img.get_rect(center=choice_rect.center)
            if choice_rect.collidepoint(mouse_pos): highlight_surf = pygame.Surface(choice_rect.size, pygame.SRCALPHA); highlight_surf.fill(PROMOTION_HIGHLIGHT_BG); surface.blit(highlight_surf, choice_rect.topleft)
            surface.blit(img, img_rect.topleft)
    promotion_popup_rect_cache = overall_rect; promotion_choice_rects_cache = choice_rects
    return overall_rect, choice_rects

# Global state for promotion popup
promotion_pending = False; pending_move = None; promotion_popup_pos = None; promotion_popup_color = None
promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}

# --- Main Game Loop ---
def play_chess960_pygame(start_board_number):
    global PIECE_IMAGES, BADGE_IMAGES, SOUNDS, drawn_tree_nodes, temp_san_board, scaled_tree_piece_images
    global tree_scroll_x, tree_scroll_y, all_badges_loaded, all_sounds_loaded
    global promotion_pending, pending_move, promotion_popup_pos, promotion_popup_color
    global promotion_popup_rect_cache, promotion_choice_rects_cache
    global animating_piece_info, tree_needs_redraw

    # --- Initialize Game State ---
    board = chess.Board(chess960=True); engine = None; analysis_manager = None; message = "Loading..."
    game_root = None; current_node = None; live_raw_score = None; live_best_move = None
    best_moves_result = None; analysis_error_message = None; meter_visible = True
    plot_surface = None; needs_redraw = True; selected_square = None
    dragging_piece_info = None; legal_moves_for_selected = []; last_move_displayed = None
    highlighted_engine_move = None; current_best_move_index = -1; helpers_visible = True
    start_time = datetime.now()

    setup_panel_ui_elements = { "button_width": 140, "button_height": SETUP_PANEL_HEIGHT - 10, "button_y": 5, "button_spacing": 10, "left_start_pos": COORD_PADDING + 5, }
    btn_x = setup_panel_ui_elements["left_start_pos"]
    setup_panel_ui_elements["show_best_move_button_rect"] = pygame.Rect(btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"])
    btn_x += setup_panel_ui_elements["button_width"] + setup_panel_ui_elements["button_spacing"]
    setup_panel_ui_elements["toggle_helpers_button_rect"] = pygame.Rect(btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"])
    btn_x += setup_panel_ui_elements["button_width"] + setup_panel_ui_elements["button_spacing"]
    setup_panel_ui_elements["save_game_button_rect"] = pygame.Rect(btn_x, setup_panel_ui_elements["button_y"], setup_panel_ui_elements["button_width"], setup_panel_ui_elements["button_height"])

    # --- Load Assets ---
    PIECE_IMAGES = load_piece_images()
    if PIECE_IMAGES is None: pygame.quit(); print("Exiting: Missing piece images."); return
    BADGE_IMAGES = load_badges()
    SOUNDS = load_sounds()

    # --- Initialize Engine ---
    try:
        if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            try:
                try: engine.configure({"UCI_Chess960": "true"})
                except Exception as cfg_err: print(f"Note: Could not configure UCI_Chess960: {cfg_err}")
                engine.ping(); print(f"Stockfish engine loaded.")
                analysis_manager = AnalysisManager(engine); analysis_manager.start()
            except (chess.engine.EngineError, BrokenPipeError, Exception) as config_err: print(f"Warning: Engine error during setup: {config_err}"); message = "Engine error."; engine.quit(); engine = None
        else: message = f"Engine not found: '{STOCKFISH_PATH}'."; print(message)
    except (FileNotFoundError, OSError, Exception) as e: message = f"Error initializing engine: {e}."; print(message); engine = None

    # --- Helper to Reset Transient State ---
    def reset_transient_state(clear_message=True):
        nonlocal selected_square, dragging_piece_info, legal_moves_for_selected, needs_redraw, message
        # Don't reset animating_piece_info here
        selected_square = None; dragging_piece_info = None; legal_moves_for_selected = []
        if clear_message: message = ""

    # --- Game Setup / Reset Function ---
    def reset_game(start_pos_num):
        nonlocal board, game_root, current_node, live_raw_score, message, start_time
        nonlocal best_moves_result, highlighted_engine_move, analysis_error_message, plot_surface, needs_redraw
        global tree_scroll_x, tree_scroll_y, promotion_pending

        try:
            board = chess.Board(chess960=True); board.set_chess960_pos(start_pos_num); start_fen = board.fen()
            print(f"Resetting to Chess960 Position {start_pos_num} (FEN: {start_fen})")
            game_root = None; current_node = None; live_raw_score = None; best_moves_result = None; highlighted_engine_move = None
            analysis_error_message = None; drawn_tree_nodes.clear(); tree_scroll_x = 0; tree_scroll_y = 0
            reset_transient_state(clear_message=False); plot_surface = None; promotion_pending = False; animating_piece_info = None
            tree_needs_redraw = True # Force tree content redraw
            game_root = GameNode(fen=start_fen, raw_score=None); current_node = game_root; message = f"Position {start_pos_num} set."
            if analysis_manager: analysis_manager.set_target(start_fen); message += " Analyzing..."
            needs_redraw = True; start_time = datetime.now(); pygame.display.set_caption(f"Chess960 Analysis - Position {start_board_number}")
        except ValueError as ve:
             print(f"Error setting Chess960 pos {start_pos_num}: {ve}"); message = f"Invalid pos: {start_pos_num}."
             if game_root is None: board = chess.Board(chess960=True); game_root = GameNode(fen=board.fen()); current_node = game_root
             tree_scroll_x = 0; tree_scroll_y = 0; promotion_pending = False; animating_piece_info = None; tree_needs_redraw = True; needs_redraw = True
        except Exception as e:
            print(f"Error resetting game: {e}"); message = f"Error setting pos {start_pos_num}."
            if game_root is None: board = chess.Board(chess960=True); game_root = GameNode(fen=board.fen()); current_node = game_root
            tree_scroll_x = 0; tree_scroll_y = 0; promotion_pending = False; animating_piece_info = None; tree_needs_redraw = True; needs_redraw = True

    # --- Helper Function to Make Move and Update State ---
    def make_move_and_update(move_to_make):
        nonlocal current_node, board, live_raw_score, message, needs_redraw
        nonlocal best_moves_result, highlighted_engine_move, current_best_move_index

        parent_node = current_node; board_before_move = deepcopy(board)
        existing_child = None
        for child in parent_node.children:
            if child.move == move_to_make: existing_child = child; break

        if existing_child:
            current_node = existing_child
            try: board.set_fen(current_node.fen); live_raw_score = current_node.raw_score; temp_san_board.set_fen(parent_node.fen); san = current_node.get_san(temp_san_board); message = f"Played {san} (existing)"
            except ValueError: message = f"Played {move_to_make.uci()} (invalid FEN?)"; board = chess.Board(current_node.fen, chess960=True)
        else:
            parent_fen = parent_node.fen; temp_board = chess.Board(parent_fen, chess960=True); temp_board.push(move_to_make); new_fen = temp_board.fen()
            new_node = GameNode(fen=new_fen, move=move_to_make, parent=parent_node, raw_score=None)
            try: temp_san_board.set_fen(parent_fen); san = new_node.get_san(temp_san_board); message = f"Played {san}"
            except ValueError: san = move_to_make.uci()+"?"; message = f"Played {san} (FEN Error?)"
            parent_node.add_child(new_node); current_node = new_node; board.set_fen(current_node.fen); live_raw_score = None

        sound_key = get_sound_key_for_move(board_before_move, board, move_to_make)
        if sound_key: play_sound(sound_key)

        reset_transient_state(clear_message=False); best_moves_result = None; highlighted_engine_move = None; current_best_move_index = -1
        if analysis_manager: 
            analysis_manager.set_target(current_node.fen); 
            if "Played" in message: 
                message += " - Analyzing..."
        tree_needs_redraw = True # New node added or navigated to existing
        needs_redraw = True


    reset_game(start_board_number) # Initial setup

    # --- Main Loop ---
    running = True; clock = pygame.time.Clock(); tree_scroll_speed = 30
    dragging_tree = False; drag_start_pos = None

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        plot_panel_y = BOARD_Y + BOARD_SIZE; tree_panel_y = plot_panel_y + PLOT_PANEL_HEIGHT
        tree_panel_rect = pygame.Rect(0, tree_panel_y, SCREEN_WIDTH, TREE_PANEL_HEIGHT)

        # Update Animation State & Check Completion
        if animating_piece_info and time.time() - animating_piece_info['start_time'] > ANIMATION_DURATION:
            animating_piece_info = None; needs_redraw = True # Final redraw

        # Process Async Analysis Results
        if analysis_manager:
            analysis_result = analysis_manager.get_latest_result()
            if analysis_result:
                needs_redraw = True; result_fen = analysis_result.get('fen'); result_type = analysis_result.get('type'); error = analysis_result.get('error')
                if error: analysis_error_message = f"Analysis Err: {error}"; print(f"Analysis error: {error}")
                elif result_type == 'continuous' and current_node and result_fen == current_node.fen:
                    new_score = analysis_result.get('score'); new_best_move = analysis_result.get('best_move');
                    if new_score is not None:
                         live_raw_score = new_score
                         if current_node.raw_score is None: current_node.raw_score = new_score; current_node.white_percentage = score_to_white_percentage(new_score, EVAL_CLAMP_LIMIT); current_node.calculate_and_set_move_quality(); tree_needs_redraw = True # Quality might affect badge
                    if new_best_move is not None: live_best_move = new_best_move
                    analysis_error_message = None
                elif result_type == 'best_moves' and current_node and result_fen == current_node.fen:
                    best_moves_result = analysis_result; moves_found = analysis_result.get('moves', [])
                    if moves_found: highlighted_engine_move = moves_found[0]['move']; current_best_move_index = 0; num_found = len(moves_found); san = moves_found[0]['san']; score_str = moves_found[0]['score_str']; message = f"Best 1/{num_found}: {san} ({score_str})"
                    else: highlighted_engine_move = None; current_best_move_index = -1; message = "Engine found no moves."
                    analysis_error_message = None

        # Process Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False; break
            if promotion_pending: # Handle promotion exclusively
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if promotion_popup_rect_cache and promotion_popup_rect_cache.collidepoint(pos):
                        chosen_piece = None;
                        for piece_type, rect in promotion_choice_rects_cache.items():
                            if rect.collidepoint(pos): chosen_piece = piece_type; break
                        if chosen_piece:
                            move = chess.Move(pending_move['from_sq'], pending_move['to_sq'], promotion=chosen_piece); temp_board_promo_check = deepcopy(board)
                            if move in temp_board_promo_check.legal_moves: make_move_and_update(move)
                            else: print(f"Error: Illegal promotion move {move.uci()}"); message = "Illegal promotion?"
                            promotion_pending = False; pending_move = None; promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}; needs_redraw = True
                    else: message = "Promotion cancelled."; promotion_pending = False; pending_move = None; promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}; needs_redraw = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: message = "Promotion cancelled."; promotion_pending = False; pending_move = None; promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}; needs_redraw = True
                continue

            # Regular Event Processing
            if event.type == pygame.MOUSEWHEEL:
                if tree_panel_rect.collidepoint(current_mouse_pos): tree_scroll_y -= event.y * tree_scroll_speed; needs_redraw = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if tree_panel_rect.collidepoint(pos): # Tree Click/Drag
                    if event.button == 1:
                        dragging_tree = True; drag_start_pos = pos; animating_piece_info = None # Cancel animation
                        tree_surface_x = pos[0] - tree_panel_rect.left + tree_scroll_x; tree_surface_y = pos[1] - tree_panel_rect.top + tree_scroll_y; clicked_node = None
                        for node, rect in drawn_tree_nodes.items():
                             if rect and rect.collidepoint(tree_surface_x, tree_surface_y): clicked_node = node; break
                        if clicked_node and clicked_node != current_node:
                             current_node = clicked_node
                             try:
                                board.set_fen(current_node.fen); live_raw_score = current_node.raw_score; reset_transient_state(); message = f"Navigated to ply {current_node.get_ply()}"
                                if analysis_manager: analysis_manager.set_target(current_node.fen)
                                best_moves_result = None; highlighted_engine_move = None; current_best_move_index = -1; tree_needs_redraw = True; needs_redraw = True
                             except ValueError: message = "Error: Invalid FEN in node."; tree_needs_redraw = True; needs_redraw = True
                elif BOARD_X <= pos[0] < BOARD_X + BOARD_SIZE and BOARD_Y <= pos[1] < BOARD_Y + BOARD_SIZE: # Board Click/Drag
                    if event.button == 1:
                        reset_transient_state(); animating_piece_info = None # Cancel animation
                        sq = screen_to_square(pos)
                        if sq is not None and not board.is_game_over():
                            piece = board.piece_at(sq)
                            if piece and piece.color == board.turn:
                                selected_square = sq; piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper(); img = PIECE_IMAGES.get(piece_key)
                                if img: dragging_piece_info = {'square': sq, 'piece': piece, 'img': img, 'pos': pos}; legal_moves_for_selected = [m for m in board.legal_moves if m.from_square == sq]
                                else: print(f"Error: Image not found for {piece_key}"); reset_transient_state()
                                needs_redraw = True
                            else: selected_square = None; legal_moves_for_selected = []; needs_redraw = True
                    elif event.button == 3: reset_transient_state(); animating_piece_info = None; needs_redraw = True

                elif setup_panel_ui_elements["show_best_move_button_rect"].collidepoint(pos): # Button Clicks
                     if not analysis_manager or not current_node or board.is_game_over(): message = "Engine unavailable/Game over."; highlighted_engine_move = None; current_best_move_index = -1; best_moves_result = None; needs_redraw = True; continue
                     current_fen = current_node.fen
                     if best_moves_result and best_moves_result.get('fen') == current_fen and best_moves_result.get('moves'):
                         moves = best_moves_result['moves']; num_moves = len(moves)
                         if num_moves > 0: current_best_move_index = (current_best_move_index + 1) % num_moves; highlighted_engine_move = moves[current_best_move_index]['move']; san = moves[current_best_move_index]['san']; score_str = moves[current_best_move_index]['score_str']; message = f"Best {current_best_move_index + 1}/{num_moves}: {san} ({score_str})"
                         else: highlighted_engine_move = None; current_best_move_index = -1; message = "No moves found."
                     else: message = "Analyzing for best moves..."; highlighted_engine_move = None; current_best_move_index = -1; best_moves_result = None; pygame.display.flip(); analysis_manager.request_best_moves(current_fen)
                     needs_redraw = True
                elif setup_panel_ui_elements["toggle_helpers_button_rect"].collidepoint(pos): helpers_visible = not helpers_visible; message = f"Helpers {'ON' if helpers_visible else 'OFF'}"; tree_needs_redraw = True; needs_redraw = True # Tree redraw needed for badges
                elif setup_panel_ui_elements["save_game_button_rect"].collidepoint(pos):
                    if current_node: save_game_history(current_node, start_time, start_board_number); message = "Game history saved."
                    else: message = "No game to save."; needs_redraw = True

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 and dragging_piece_info: # Finish dragging piece
                    pos = event.pos; to_sq = screen_to_square(pos); from_sq = dragging_piece_info['square']; piece = dragging_piece_info['piece']
                    dragging_piece_info = None; move_made_or_pending = False
                    if to_sq is not None and from_sq != to_sq:
                        is_promotion = (piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7])
                        if is_promotion:
                            potential_move = chess.Move(from_sq, to_sq); is_legal_target = False
                            for legal_move in board.legal_moves:
                                if legal_move.from_square == from_sq and legal_move.to_square == to_sq: is_legal_target = True; break
                            if is_legal_target: promotion_pending = True; pending_move = {'from_sq': from_sq, 'to_sq': to_sq}; promotion_popup_pos = square_to_screen_coords(to_sq); promotion_popup_color = board.turn; promotion_popup_rect_cache = None; promotion_choice_rects_cache = {}; message = "Select promotion piece"; move_made_or_pending = True
                            else: message = "Illegal move"
                        else: # Regular move
                            move = chess.Move(from_sq, to_sq)
                            if move in board.legal_moves: make_move_and_update(move); move_made_or_pending = True
                            else: message = f"Illegal move"
                    if not move_made_or_pending: selected_square = None; legal_moves_for_selected = []
                    needs_redraw = True
                 if event.button == 1 and dragging_tree: dragging_tree = False; needs_redraw = True

            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece_info: dragging_piece_info['pos'] = event.pos; needs_redraw = True
                if dragging_tree: dx = event.pos[0] - drag_start_pos[0]; dy = event.pos[1] - drag_start_pos[1]; tree_scroll_x -= dx; tree_scroll_y -= dy; drag_start_pos = event.pos; needs_redraw = True
                if promotion_pending: needs_redraw = True

            elif event.type == pygame.KEYDOWN:
                node_changed = False; previous_node = current_node; move_for_animation = None; is_forward_animation = False
                if event.key == pygame.K_LEFT:
                    if current_node and current_node.parent: move_for_animation = previous_node.move; is_forward_animation = False; current_node = current_node.parent; node_changed = True; message = f"Back (Ply {current_node.get_ply()})"
                    else: message = "At start"
                elif event.key == pygame.K_RIGHT:
                    if current_node and current_node.children: move_for_animation = current_node.children[0].move; is_forward_animation = True; current_node = current_node.children[0]; node_changed = True; message = f"Forward (Ply {current_node.get_ply()})"
                    else: message = "End of line"
                elif event.key == pygame.K_UP:
                    if current_node and current_node.parent:
                        parent = current_node.parent; siblings = parent.children; current_index = siblings.index(current_node)
                        if current_index > 0: move_for_animation = siblings[current_index - 1].move; is_forward_animation = True; current_node = siblings[current_index - 1]; node_changed = True; message = f"Sibling Up (Ply {current_node.get_ply()})"
                        else: message = "First sibling"
                    else: message = "No siblings"
                elif event.key == pygame.K_DOWN:
                    if current_node and current_node.parent:
                        parent = current_node.parent; siblings = parent.children; current_index = siblings.index(current_node)
                        if current_index < len(siblings) - 1: move_for_animation = siblings[current_index + 1].move; is_forward_animation = True; current_node = siblings[current_index + 1]; node_changed = True; message = f"Sibling Down (Ply {current_node.get_ply()})"
                        else: message = "Last sibling"
                    else: message = "No siblings"

                if node_changed:
                    the_move_traversed = move_for_animation
                    try:
                        board.set_fen(current_node.fen); live_raw_score = current_node.raw_score; reset_transient_state(clear_message=False)
                        if analysis_manager: analysis_manager.set_target(current_node.fen)
                        best_moves_result = None; highlighted_engine_move = None; current_best_move_index = -1
                        tree_needs_redraw = True # Node changed, tree highlight must update

                        if the_move_traversed:
                            board_before_the_move = chess.Board(chess960=True); board_after_the_move = chess.Board(chess960=True)
                            if is_forward_animation: board_before_the_move.set_fen(previous_node.fen); board_after_the_move.set_fen(current_node.fen)
                            else: board_before_the_move.set_fen(current_node.fen); board_after_the_move.set_fen(previous_node.fen)

                            sound_key = get_sound_key_for_move(board_before_the_move, board_after_the_move, the_move_traversed)
                            if sound_key: play_sound(sound_key)

                            animating_piece_info = None; from_sq = the_move_traversed.from_square; to_sq = the_move_traversed.to_square
                            piece = board_before_the_move.piece_at(from_sq)
                            if piece:
                                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper(); img = PIECE_IMAGES.get(piece_key)
                                if img:
                                    screen_pos_from = square_to_screen_coords(from_sq); screen_pos_to = square_to_screen_coords(to_sq)
                                    start_pos, end_pos = (screen_pos_from, screen_pos_to) if is_forward_animation else (screen_pos_to, screen_pos_from)
                                    animating_piece_info = {'piece_key': piece_key, 'img': img, 'from_sq': from_sq, 'to_sq': to_sq, 'start_pos': start_pos, 'end_pos': end_pos, 'start_time': time.time()}
                                else: print(f"Animation Error: Image not found: {piece_key}")
                            else: print(f"Animation Error: No piece at {chess.square_name(from_sq)}")
                        needs_redraw = True # Always redraw screen after node change
                    except ValueError:
                        message = "Error: Invalid FEN."; print(f"ERROR: Invalid FEN navigation")
                        if previous_node: current_node = previous_node
                        try: board.set_fen(current_node.fen)
                        except: pass
                        animating_piece_info = None; tree_needs_redraw = True; needs_redraw = True

                elif event.key == pygame.K_a: # Other keys
                     if analysis_manager and current_node: message = "Analyzing..."; highlighted_engine_move = None; current_best_move_index = -1; best_moves_result = None; pygame.display.flip(); analysis_manager.request_best_moves(current_node.fen)
                     else: message = "Engine unavailable."; needs_redraw = True
                elif event.key == pygame.K_m: meter_visible = not meter_visible; message = f"Eval Plot {'ON' if meter_visible else 'OFF'}"; needs_redraw = True
                elif event.key == pygame.K_h: helpers_visible = not helpers_visible; message = f"Helpers {'ON' if helpers_visible else 'OFF'}"; tree_needs_redraw = True; needs_redraw = True
                elif event.key == pygame.K_s:
                    if current_node: save_game_history(current_node, start_time, start_board_number); message = "Game saved."
                    else: message = "No game to save."; needs_redraw = True
                elif event.key == pygame.K_ESCAPE: running = False; break

        if not running: continue

        # Safety check
        if not promotion_pending and not animating_piece_info and current_node and board.fen() != current_node.fen:
             try: print(f"Warning: Board desync. Forcing FEN: {current_node.fen}"); board.set_fen(current_node.fen); live_raw_score = current_node.raw_score; reset_transient_state(clear_message=False); tree_needs_redraw=True; needs_redraw = True
             except ValueError: print(f"CRITICAL ERROR: Invalid FEN: {current_node.fen}"); message = "CRITICAL FEN ERROR!"; needs_redraw = True

        # Get last move for highlight
        last_move_displayed = current_node.move if current_node and current_node.parent else None
        current_last_move_highlight = last_move_displayed if not animating_piece_info else None # Hide during animation

        # Update dynamic UI text
        best_move_button_text = "Show Best"
        if best_moves_result and current_node and best_moves_result.get('fen') == current_node.fen:
            moves = best_moves_result.get('moves'); error = best_moves_result.get('error')
            if error: best_move_button_text = "Analysis Failed"
            elif moves: num_found = len(moves); best_move_button_text = f"Showing {current_best_move_index + 1}/{num_found}"
            else: best_move_button_text = "No moves found"
        elif analysis_manager and current_node and board.is_game_over(): best_move_button_text = "Game Over"
        elif not analysis_manager: best_move_button_text = "Engine Off"

        # Redraw Screen
        if animating_piece_info: needs_redraw = True

        if needs_redraw:
            screen.fill(DARK_GREY)
            draw_setup_panel(screen, setup_panel_ui_elements, BUTTON_FONT, best_move_button_text, helpers_visible)
            draw_coordinates(screen, COORD_FONT)
            draw_board(screen)
            if not promotion_pending: draw_highlights(screen, board, selected_square, legal_moves_for_selected, current_last_move_highlight, highlighted_engine_move)
            draw_pieces(screen, board, PIECE_IMAGES, dragging_piece_info, promotion_pending, animating_piece_info)

            if helpers_visible and not promotion_pending: # Draw helpers unless promoting
                if current_node and current_node.move and current_node.parent and current_node.move_quality:
                    draw_board_badge(screen, current_node.move.to_square, current_node.move_quality)
                current_wp = score_to_white_percentage(live_raw_score, EVAL_CLAMP_LIMIT) if current_node else 50.0
                draw_eval_bar(screen, current_wp)
                if meter_visible and current_node:
                    path = get_path_to_node(current_node); plot_surface = create_eval_plot_surface(path, BOARD_SIZE, PLOT_PANEL_HEIGHT)
                    if plot_surface: plot_rect = plot_surface.get_rect(topleft=(BOARD_X, plot_panel_y)); screen.blit(plot_surface, plot_rect)
                    else: plot_rect = pygame.Rect(BOARD_X, plot_panel_y, BOARD_SIZE, PLOT_PANEL_HEIGHT); pygame.draw.rect(screen, DARK_GREY, plot_rect); pygame.draw.rect(screen, GREY, plot_rect, 1); error_surf = TREE_FONT.render("Plot Error", True, ORANGE); screen.blit(error_surf, error_surf.get_rect(center=plot_rect.center))

            draw_game_tree(screen, game_root, current_node, TREE_FONT, helpers_visible) # Uses internal tree_needs_redraw flag

            if animating_piece_info: draw_animating_piece(screen, animating_piece_info)
            if promotion_pending: draw_promotion_popup(screen, promotion_popup_pos, promotion_popup_color, current_mouse_pos)

            status_y_offset = SCREEN_HEIGHT - 10
            if analysis_error_message: error_surf = TREE_FONT.render(analysis_error_message, True, ORANGE); error_rect = error_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset)); screen.blit(error_surf, error_rect); status_y_offset -= error_rect.height + 2
            if message: status_surf = TREE_FONT.render(message, True, WHITE); status_rect = status_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, status_y_offset)); screen.blit(status_surf, status_rect)

            pygame.display.flip()
            needs_redraw = False

        clock.tick(60)

    # --- Cleanup ---
    print("\nExiting Pygame...")
    if analysis_manager: analysis_manager.stop()
    if engine: 
        try: 
            time.sleep(0.1); 
            engine.quit(); 
            print("Stockfish engine closed.")
        except (AttributeError, BrokenPipeError, Exception) as e: 
            print(f"Error closing engine: {e}")
    plt.close('all'); pygame.quit(); sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chess960 Analysis Board")
    parser.add_argument("board_number", type=int, nargs='?', help="Chess960 start position (0-959). Random if omitted.")
    args = parser.parse_args()
    if args.board_number is None: args.board_number = random.randint(0, 959); print(f"Using random board number: {args.board_number}")
    elif not (0 <= args.board_number <= 959): print(f"Error: Board number must be 0-959. Got: {args.board_number}"); sys.exit(1)
    play_chess960_pygame(args.board_number)