#!/bin/bash

# --- Configuration ---
PROJECT_DIR="pgn-analyzer-extension"
LIBS_DIR="$PROJECT_DIR/libs"
ICONS_DIR="$PROJECT_DIR/icons"
IMG_DIR="$LIBS_DIR/img/chesspieces/wikipedia"

# URLs for Libraries (Check for latest versions if needed)
JQUERY_URL="https://code.jquery.com/jquery-3.7.1.min.js"
JQUERY_FILENAME="jquery-3.7.1.min.js"

CHESSBOARD_JS_URL="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"
CHESSBOARD_JS_FILENAME="chessboard-1.0.0.min.js"
CHESSBOARD_CSS_URL="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"
CHESSBOARD_CSS_FILENAME="chessboard-1.0.0.min.css"
CHESSBOARD_IMG_WP_URL="https://raw.githubusercontent.com/oakmac/chessboardjs/master/website/img/chesspieces/wikipedia/wP.png" # Placeholder piece

CHESSJS_URL="https://raw.githubusercontent.com/jhlywa/chess.js/master/dist/chess.min.js"
CHESSJS_FILENAME="chess.js" # Rename for consistency with example

CHARTJS_URL="https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.js"
CHARTJS_FILENAME="chart.umd.js"

STOCKFISH_JS_URL="https://raw.githubusercontent.com/nmrugg/stockfish.js/master/stockfish.js"
STOCKFISH_JS_FILENAME="stockfish.js"
STOCKFISH_WASM_URL="https://raw.githubusercontent.com/nmrugg/stockfish.js/master/stockfish.wasm"
STOCKFISH_WASM_FILENAME="stockfish.wasm"

# --- Helper Function ---
download_file() {
  local url="$1"
  local target_path="$2"
  echo "Downloading $(basename "$target_path")..."
  # Use -f to fail silently on server errors, -L to follow redirects, -sS for silent progress but show errors
  if curl -f -sSL "$url" -o "$target_path"; then
    echo " -> Saved to $target_path"
  else
    echo " !! FAILED to download $url"
    # Optionally exit on failure: exit 1
  fi
}

# --- Main Script ---
echo "Setting up PGN Analyzer Extension directory: $PROJECT_DIR"

# Create Directories
echo "Creating directories..."
mkdir -p "$PROJECT_DIR"
mkdir -p "$LIBS_DIR"
mkdir -p "$ICONS_DIR"
mkdir -p "$IMG_DIR"
echo " -> Directories created."

# Download Libraries
echo "Downloading libraries..."
download_file "$JQUERY_URL" "$LIBS_DIR/$JQUERY_FILENAME"
download_file "$CHESSBOARD_JS_URL" "$LIBS_DIR/$CHESSBOARD_JS_FILENAME"
download_file "$CHESSBOARD_CSS_URL" "$LIBS_DIR/$CHESSBOARD_CSS_FILENAME"
download_file "$CHESSJS_URL" "$LIBS_DIR/$CHESSJS_FILENAME"
download_file "$CHARTJS_URL" "$LIBS_DIR/$CHARTJS_FILENAME"
download_file "$STOCKFISH_JS_URL" "$LIBS_DIR/$STOCKFISH_JS_FILENAME"
download_file "$STOCKFISH_WASM_URL" "$LIBS_DIR/$STOCKFISH_WASM_FILENAME"
echo " -> Library downloads attempted."

# Download Placeholder Chess Piece Image
echo "Downloading placeholder chessboard piece (wP.png)..."
download_file "$CHESSBOARD_IMG_WP_URL" "$IMG_DIR/wP.png"
echo "NOTE: You need the *full set* of chessboard images (e.g., wP.png, wN.png, bK.png etc.)"
echo "      in the $IMG_DIR directory."
echo "      Download them from https://chessboardjs.com/ or its GitHub repo."

# Create Placeholder Icons (Optional: Requires ImageMagick)
echo "Attempting to create placeholder icons..."
if command -v convert &> /dev/null; then
  convert -size 16x16 xc:skyblue "$ICONS_DIR/icon16.png"
  convert -size 48x48 xc:steelblue "$ICONS_DIR/icon48.png"
  convert -size 128x128 xc:dodgerblue "$ICONS_DIR/icon128.png"
  echo " -> Placeholder icons created using ImageMagick."
else
  echo " -> ImageMagick 'convert' command not found. Skipping icon creation."
  echo "    Please create the following files manually in the '$ICONS_DIR' directory:"
  echo "    - icon16.png (16x16 pixels)"
  echo "    - icon48.png (48x48 pixels)"
  echo "    - icon128.png (128x128 pixels)"
  # Optionally create empty files as placeholders
  touch "$ICONS_DIR/icon16.png" "$ICONS_DIR/icon48.png" "$ICONS_DIR/icon128.png"
  echo "    (Empty files created as placeholders)"
fi

echo ""
echo "--- Setup Complete ---"
echo "Directory structure and library files created in '$PROJECT_DIR'."
echo "Remember to:"
echo "1. Place your 'manifest.json', 'popup.html', 'popup.css', and 'popup.js' files in '$PROJECT_DIR'."
echo "2. Ensure you have the *complete set* of chessboard images in '$IMG_DIR'."
echo "3. Provide proper icons in '$ICONS_DIR' if placeholders were created or skipped."
echo "4. Verify the 'pieceTheme' path in 'popup.js' matches the image location (currently expects 'libs/img/chesspieces/wikipedia/{piece}.png')."
