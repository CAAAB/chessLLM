// --- DOM Elements ---
const boardContainer = document.getElementById('board-container');
const evalBarWhite = document.getElementById('eval-bar-white');
const evalBarBlack = document.getElementById('eval-bar-black');
const plotCanvas = document.getElementById('eval-plot');
const treePanel = document.getElementById('tree-panel');
const treeSvg = document.getElementById('game-tree-svg');
const treePanelContainer = document.getElementById('tree-panel-container');
const statusMessageEl = document.getElementById('status-message');
const engineStatusMessageEl = document.getElementById('engine-status-message');
const promotionPopup = document.getElementById('promotion-popup');
const blackNameInput = document.getElementById('black-name');
const whiteNameInput = document.getElementById('white-name');
const showBestMoveBtn = document.getElementById('show-best-move-btn');
const explainPlanBtn = document.getElementById('explain-plan-btn');
const toggleHelpersBtn = document.getElementById('toggle-helpers-btn');
const saveGameBtn = document.getElementById('save-game-btn');
const boardNumDisplay = document.getElementById('board-number-display');
const coordContainer = document.getElementById('coordinate-container');

// --- Constants --- (Mirroring Python)
const PIECE_IMAGE_PATH = "pieces";
const BADGE_IMAGE_PATH = "badges";
const SOUND_PATH = "sounds";
const EVAL_CLAMP_LIMIT = 800;
const MATE_SCORE_PLOT_VALUE = EVAL_CLAMP_LIMIT * 1.5;
const NUM_BEST_MOVES_TO_SHOW = 5;
const CONTINUOUS_ANALYSIS_TIME_LIMIT_MS = 500; // In milliseconds for JS
const BEST_MOVE_ANALYSIS_TIME_LIMIT_MS = 1000;
const SELECTED_PIECE_MOVE_ANALYSIS_TIME_LIMIT_MS = 750; // For selected piece move evals
const PLAYER_PIECES_ANALYSIS_TIME_LIMIT_MS = 1000; // New: For all current player piece best moves
const ANIMATION_DURATION_MS = 200;
// Tree Layout
const TREE_PIECE_SIZE = 35;
const NODE_DIAMETER = TREE_PIECE_SIZE;
const HORIZ_SPACING = 40 + TREE_PIECE_SIZE;
const VERT_SPACING = 5 + TREE_PIECE_SIZE;
const TEXT_OFFSET_X = 5;
const TEXT_OFFSET_Y = TREE_PIECE_SIZE / 2 + 10;
const TREE_BADGE_RADIUS = TREE_PIECE_SIZE / 5;
const BOARD_BADGE_RADIUS_PX = 10;

// Tree Badge Colors (hex)
const TREE_MOVE_QUALITY_COLORS = {
    "Best": "#00b400", "Excellent": "#32cd32", "Good": "#006400",
    "Inaccuracy": "#f0e68c", "Mistake": "#ffa500", "Blunder": "#c80000"
};
const BOARD_BADGE_IMAGE_SIZE = { width: 16, height: 16 };

// Selected Piece Move Evaluation Colors
const SELECTED_PIECE_MOVE_EVAL_CLAMP = 150; // Adjusted as per subtask
const SELECTED_PIECE_MOVE_MAX_ALPHA = 0.65;

// --- Game State ---
let game = new Chess();
let board = null;
let engine = null;
let currentBoardNumber = -1;
let gameRoot = null;
let currentNode = null;
let selectedSquare = null;
let draggingPieceElement = null;
let dragStartX = 0, dragStartY = 0;
let dragStartSquare = null;
let legalMovesForSelected = [];
let lastMoveDisplayed = null;
let highlightedEngineMove = null;
let currentBestMovesResult = null;
let selectedPieceEvaluations = null; // { fen: string, pieceSq: string, evaluations: Map<string(toSq), { score_obj, uci }> }
let currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; // New: Map<pieceSq, { bestToSq, score_obj, uci }>
let currentBestMoveIndex = -1;
let liveRawScore = null; // { score: cp | mate, depth: d } (White's POV)
let analysisErrorMessage = null;
let helpersVisible = true;
let plotVisible = true;
let currentEvalPlot = null;
let promotionState = {
    pending: false,
    fromSq: null,
    toSq: null,
    callback: null
};
let animatingPieceInfo = null;
let displayedWhitePercentage = 50.0;
let targetWhitePercentage = 50.0;
let analysisManager = {
    isProcessing: false,
    requestQueue: [], // { type: 'continuous' | 'best_moves' | 'selected_piece_moves' | 'player_pieces_best_moves', fen: '...', ... }
    currentAnalysisType: null,
    lastContinuousFen: null,
    lastBestMovesFen: null,
    lastSelectedPieceFen: null,
    lastSelectedPieceSq: null,
    lastPlayerPiecesBestMovesFen: null, // New
    lastScore: null, // White's POV
    lastBestMove: null
};
let startTime = new Date();
let sounds = {};
let badgeImages = {};

// Tree Drawing State
let treeNodeElements = new Map();
let treeLayout = {
    nodePositions: new Map(),
    maxX: 0,
    maxY: 0,
    needsRedraw: true
};
let treeDragging = {
    active: false,
    startX: 0,
    startY: 0,
    startScrollX: 0,
    startScrollY: 0
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initEngine();
    initBoard();
    initPlot();
    initSounds();
    setupEventListeners();

    const urlParams = new URLSearchParams(window.location.search);
    let startPos = parseInt(urlParams.get('pos'), 10);
    if (isNaN(startPos) || startPos < 0 || startPos > 959) {
        startPos = Math.floor(Math.random() * 960);
        console.log(`No valid position specified, using random: ${startPos}`);
    }
    resetGame(startPos);
    requestAnimationFrame(gameLoop);
});

// --- Engine Communication ---
function initEngine() {
    statusMessage("Initializing Stockfish...");
    try {
        var wasmSupported = typeof WebAssembly === 'object' && WebAssembly.validate(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
        var stockfish = new Worker(wasmSupported ? 'stockfish.wasm.js' : 'stockfish.js');
        stockfish.addEventListener('message', function (e) { handleEngineMessage(e.data); });
        stockfish.postMessage('uci');
        engine = stockfish;
        statusMessage("Stockfish Loaded. Configuring...");
    } catch (error) {
        console.error("Error during Stockfish initialization:", error);
        engineStatusMessage("Stockfish init error.");
        statusMessage("Engine unavailable.");
    }
}

function setEngineOptions() {
    if (!engine) return;
    try {
        engine.postMessage('setoption name UCI_Chess960 value true');
        engine.postMessage('setoption name Threads value 2');
        engine.postMessage('setoption name Hash value 32');
        engine.postMessage('isready');
    } catch (e) {
        console.error("Error posting message within setEngineOptions:", e);
        engineStatusMessage("Error setting engine options.");
    }
}

function handleEngineMessage(message) {
    if (message === 'uciok') {
        setEngineOptions();
    } else if (message === 'readyok') {
        statusMessage("Engine Ready.");
        processAnalysisQueue();
    } else if (message.startsWith('info')) {
        parseInfoLine(message);
    } else if (message.startsWith('bestmove')) {
        parseBestmoveLine(message);
    } else if (message.startsWith('option name')) {
        // Ignore
    } else {
        console.log("Unhandled engine message:", message);
    }
}

function requestAnalysis(fen, type = 'continuous', options = {}) {
    if (!engine) return;
    let newRequest = { type: type, fen: fen, ...options };

    if (type === 'best_moves') {
        newRequest.multipv = options.multipv || NUM_BEST_MOVES_TO_SHOW;
        analysisManager.requestQueue = [newRequest];
    } else if (type === 'selected_piece_moves') {
        if (!options.searchMovesUci || !options.pieceSq) {
            console.error("Missing searchMovesUci or pieceSq for selected_piece_moves analysis."); return;
        }
        analysisManager.requestQueue = [newRequest];
    } else if (type === 'player_pieces_best_moves') {
        if (!options.searchMovesUci) {
            console.error("Missing searchMovesUci for player_pieces_best_moves analysis."); return;
        }
        analysisManager.requestQueue = [newRequest];
    } else if (type === 'continuous') {
        if (analysisManager.requestQueue.length === 0 || analysisManager.requestQueue.every(r => r.type === 'continuous')) {
            analysisManager.requestQueue = [newRequest];
        } else { return; }
    }
    processAnalysisQueue();
}

function processAnalysisQueue() {
    if (analysisManager.isProcessing || analysisManager.requestQueue.length === 0 || !engine) return;

    const request = analysisManager.requestQueue.shift();
    analysisManager.isProcessing = true;
    analysisManager.currentAnalysisType = request.type;
    analysisManager.activeRequestBaseScoreWhitePov = null; // Reset
    console.log(`Starting analysis: ${request.type} for FEN: ${request.fen.substring(0, 20)}...`);
    engineStatusMessage("");

    if (request.type === 'selected_piece_moves' || request.type === 'player_pieces_best_moves') {
        if (analysisManager.lastScore) { // Ensure lastScore is available from a previous continuous analysis
            analysisManager.activeRequestBaseScoreWhitePov = JSON.parse(JSON.stringify(analysisManager.lastScore)); // Deep copy
            console.log(`Stored base score for delta calculation:`, analysisManager.activeRequestBaseScoreWhitePov);
        } else {
            console.warn(`No lastScore available for delta calculation for ${request.type}. Deltas will be relative to zero or not calculable if base is needed.`);
            // Fallback: if no base score, deltas might be misleading or absolute scores might be used implicitly if base is null.
            // For now, activeRequestBaseScoreWhitePov will remain null, and delta calculation needs to handle this.
        }
    }

    engine.postMessage('stop');
    engine.postMessage(`position fen ${request.fen}`);

    if (request.type === 'continuous') {
        analysisManager.lastContinuousFen = request.fen;
        engine.postMessage(`go movetime ${CONTINUOUS_ANALYSIS_TIME_LIMIT_MS}`);
    } else if (request.type === 'best_moves') {
        analysisManager.lastBestMovesFen = request.fen;
        engine.postMessage(`setoption name MultiPV value ${request.multipv}`);
        engine.postMessage(`go movetime ${BEST_MOVE_ANALYSIS_TIME_LIMIT_MS}`);
        currentBestMovesResult = { fen: request.fen, moves: [], error: null };
    } else if (request.type === 'selected_piece_moves') {
        analysisManager.lastSelectedPieceFen = request.fen;
        analysisManager.lastSelectedPieceSq = request.pieceSq;
        engine.postMessage(`setoption name MultiPV value ${request.searchMovesUci.length}`);
        engine.postMessage(`go searchmoves ${request.searchMovesUci.join(' ')} movetime ${SELECTED_PIECE_MOVE_ANALYSIS_TIME_LIMIT_MS}`);
        selectedPieceEvaluations = { fen: request.fen, pieceSq: request.pieceSq, evaluations: new Map() };
        statusMessage(`Evaluating moves for ${request.pieceSq}...`);
    } else if (request.type === 'player_pieces_best_moves') {
        analysisManager.lastPlayerPiecesBestMovesFen = request.fen;
        if (currentPlayerPieceBestEvals.fen !== request.fen) {
            currentPlayerPieceBestEvals = { fen: request.fen, evals: new Map() };
        }
        engine.postMessage(`setoption name MultiPV value ${Math.min(request.searchMovesUci.length, 250)}`); // Cap MultiPV due to engine limits
        engine.postMessage(`go searchmoves ${request.searchMovesUci.join(' ')} movetime ${PLAYER_PIECES_ANALYSIS_TIME_LIMIT_MS}`);
        // Status message is handled by the calling function (requestPlayerPiecesBestMoves)
    }
}

function parseInfoLine(line) {
    const parts = line.split(' ');
    let parsed_score_obj_white_pov = null;
    let pv = [];
    let depth = null;
    let multipvRank = null;
    let score_current_for_delta_calc = null;

    if (analysisManager.currentAnalysisType === 'selected_piece_moves' || analysisManager.currentAnalysisType === 'player_pieces_best_moves') {
        score_current_for_delta_calc = analysisManager.activeRequestBaseScoreWhitePov;
        // console.log("Retrieved base score in parseInfoLine:", score_current_for_delta_calc);
    }

    for (let i = 0; i < parts.length; i++) {
        if (parts[i] === 'score') {
            let cp_val = null, mate_val = null;
            if (parts[i + 1] === 'mate') {
                mate_val = parseInt(parts[i + 2], 10);
                i += 2;
            } else if (parts[i + 1] === 'cp') {
                cp_val = parseInt(parts[i + 2], 10);
                i += 2;
            }

            const fen_of_analysis = analysisManager.currentAnalysisType === 'continuous' ? analysisManager.lastContinuousFen :
                                   analysisManager.currentAnalysisType === 'best_moves' ? analysisManager.lastBestMovesFen :
                                   analysisManager.currentAnalysisType === 'selected_piece_moves' ? analysisManager.lastSelectedPieceFen :
                                   analysisManager.currentAnalysisType === 'player_pieces_best_moves' ? analysisManager.lastPlayerPiecesBestMovesFen :
                                   null;
            if (fen_of_analysis) {
                const tempGame = new Chess(fen_of_analysis);
                if (tempGame.turn() === 'b') { // UCI score is POV of side to move. Convert to White's POV.
                    if (mate_val !== null) mate_val = -mate_val;
                    if (cp_val !== null) cp_val = -cp_val;
                }
            }
            parsed_score_obj_white_pov = { cp: cp_val, mate: mate_val };
        } else if (parts[i] === 'pv') {
            pv = parts.slice(i + 1);
            break;
        } else if (parts[i] === 'depth') {
            depth = parseInt(parts[i + 1], 10);
            i++;
        } else if (parts[i] === 'multipv') {
            multipvRank = parseInt(parts[i+1], 10);
            i++;
        }
    }

    if (analysisManager.currentAnalysisType === 'continuous' && parsed_score_obj_white_pov) {
        liveRawScore = parsed_score_obj_white_pov;
        targetWhitePercentage = scoreToWhitePercentage(parsed_score_obj_white_pov, EVAL_CLAMP_LIMIT);
        analysisManager.lastScore = parsed_score_obj_white_pov; // Keep this as absolute for the next base score
        if (pv.length > 0) analysisManager.lastBestMove = pv[0];
    } else if (analysisManager.currentAnalysisType === 'best_moves' && parsed_score_obj_white_pov && pv.length > 0 && multipvRank !== null) {
        if (currentBestMovesResult && currentBestMovesResult.fen === analysisManager.lastBestMovesFen) {
            const moveUci = pv[0];
            const moveSan = uciToSan(moveUci, analysisManager.lastBestMovesFen);
            const scoreStr = formatScore(parsed_score_obj_white_pov, new Chess(analysisManager.lastBestMovesFen).turn());
            let existingMove = currentBestMovesResult.moves.find(m => m.rank === multipvRank);
            const pv_sequence = (Array.isArray(pv) && pv.length > 0) ? pv : [];
            const moveData = {
                move: moveUci,
                san: moveSan,
                score_str: scoreStr,
                score_obj: parsed_score_obj_white_pov, // Store absolute score for best_moves display
                depth: depth,
                pv_sequence: pv_sequence
            };
            if (existingMove) {
                Object.assign(existingMove, moveData);
            } else {
                currentBestMovesResult.moves.push({ rank: multipvRank, ...moveData });
            }
            currentBestMovesResult.moves.sort((a, b) => a.rank - b.rank);
        }
    } else if (analysisManager.currentAnalysisType === 'selected_piece_moves' && parsed_score_obj_white_pov && pv.length > 0 && multipvRank !== null) {
        if (selectedPieceEvaluations && selectedPieceEvaluations.fen === analysisManager.lastSelectedPieceFen &&
            selectedPieceEvaluations.pieceSq === analysisManager.lastSelectedPieceSq) {
            const moveUci = pv[0];
            const fromSq = moveUci.substring(0,2);
            const toSq = moveUci.substring(2,4);
            if (fromSq === selectedPieceEvaluations.pieceSq) {
                let eval_delta_white_pov = { cp: null, mate: null };
                const score_after_move_white_pov = parsed_score_obj_white_pov;

                if (score_current_for_delta_calc && score_after_move_white_pov) {
                    // Both are mate scores
                    if (score_current_for_delta_calc.mate !== null && score_after_move_white_pov.mate !== null) {
                        // If current is M5 (good for W), and after_move is M3 (better for W), delta_mate = 3 - 5 = -2 (favors white: fewer moves to mate)
                        // If current is M-5 (bad for W), and after_move is M-3 (less bad for W), delta_mate = -3 - (-5) = 2 (favors white: fewer moves for black to mate)
                        // This needs to be consistent: positive delta = good for white.
                        // Let's define "better mate" as smaller positive number, or larger negative number (closer to 0).
                        // If after_move.mate > 0 and current_move.mate > 0: (e.g. M3 vs M5) current - after = 5-3=2. Positive is good.
                        // If after_move.mate < 0 and current_move.mate < 0: (e.g. M-3 vs M-5) current - after = -5 - (-3) = -2. Positive is good. (No, this is wrong)
                        // Let's use: if positive mate, smaller is better. If negative mate, larger (closer to 0) is better.
                        // Simpler: convert to a "mate score": positive for white mating, negative for black mating.
                        // Mate score: (sign) * (1000 - abs(mate_moves)). Higher is better.
                        const current_mate_score = score_current_for_delta_calc.mate !== 0 ? Math.sign(score_current_for_delta_calc.mate) * (1000 - Math.abs(score_current_for_delta_calc.mate)) : 0;
                        const after_mate_score = score_after_move_white_pov.mate !== 0 ? Math.sign(score_after_move_white_pov.mate) * (1000 - Math.abs(score_after_move_white_pov.mate)) : 0;
                        eval_delta_white_pov.cp = after_mate_score - current_mate_score; // Treat as CP delta for simplicity in downstream use
                    }
                    // Both are CP scores
                    else if (score_current_for_delta_calc.cp !== null && score_after_move_white_pov.cp !== null) {
                        eval_delta_white_pov.cp = score_after_move_white_pov.cp - score_current_for_delta_calc.cp;
                    }
                    // Mixed: after_move is mate, current is CP
                    else if (score_after_move_white_pov.mate !== null && score_current_for_delta_calc.cp !== null) {
                        eval_delta_white_pov.cp = (score_after_move_white_pov.mate > 0 ? 1 : -1) * 20000 - score_current_for_delta_calc.cp; // Large delta favoring mate
                    }
                    // Mixed: current is mate, after_move is CP
                    else if (score_current_for_delta_calc.mate !== null && score_after_move_white_pov.cp !== null) {
                         eval_delta_white_pov.cp = score_after_move_white_pov.cp - ((score_current_for_delta_calc.mate > 0 ? 1 : -1) * 20000); // Large delta, sign depends on who was mating
                    } else { // One or both are null, store absolute as fallback
                        eval_delta_white_pov = score_after_move_white_pov;
                    }
                } else { // If no base score, store absolute score_after_move_white_pov
                    eval_delta_white_pov = score_after_move_white_pov;
                }
                selectedPieceEvaluations.evaluations.set(toSq, { score_obj: eval_delta_white_pov, uci: moveUci });
                requestRedraw();
            }
        }
    } else if (analysisManager.currentAnalysisType === 'player_pieces_best_moves' && parsed_score_obj_white_pov && pv.length > 0) {
        if (currentPlayerPieceBestEvals && currentPlayerPieceBestEvals.fen === analysisManager.lastPlayerPiecesBestMovesFen) {
            const moveUci = pv[0];
            const fromSq = moveUci.substring(0, 2);
            const toSq = moveUci.substring(2, 4);
            const currentTurn = new Chess(analysisManager.lastPlayerPiecesBestMovesFen).turn(); // Turn for isScoreBetter
            let eval_delta_white_pov = { cp: null, mate: null };
            const score_after_move_white_pov = parsed_score_obj_white_pov;

            if (score_current_for_delta_calc && score_after_move_white_pov) {
                 if (score_current_for_delta_calc.mate !== null && score_after_move_white_pov.mate !== null) {
                    const current_mate_score = score_current_for_delta_calc.mate !== 0 ? Math.sign(score_current_for_delta_calc.mate) * (1000 - Math.abs(score_current_for_delta_calc.mate)) : 0;
                    const after_mate_score = score_after_move_white_pov.mate !== 0 ? Math.sign(score_after_move_white_pov.mate) * (1000 - Math.abs(score_after_move_white_pov.mate)) : 0;
                    eval_delta_white_pov.cp = after_mate_score - current_mate_score;
                } else if (score_current_for_delta_calc.cp !== null && score_after_move_white_pov.cp !== null) {
                    eval_delta_white_pov.cp = score_after_move_white_pov.cp - score_current_for_delta_calc.cp;
                } else if (score_after_move_white_pov.mate !== null && score_current_for_delta_calc.cp !== null) {
                    eval_delta_white_pov.cp = (score_after_move_white_pov.mate > 0 ? 1 : -1) * 20000 - score_current_for_delta_calc.cp;
                } else if (score_current_for_delta_calc.mate !== null && score_after_move_white_pov.cp !== null) {
                    eval_delta_white_pov.cp = score_after_move_white_pov.cp - ((score_current_for_delta_calc.mate > 0 ? 1 : -1) * 20000);
                }  else {
                    eval_delta_white_pov = score_after_move_white_pov;
                }
            } else {
                eval_delta_white_pov = score_after_move_white_pov;
            }

            const existingEvalData = currentPlayerPieceBestEvals.evals.get(fromSq);
            // isScoreBetter now compares deltas (or absolute if no base was available)
            // For deltas, a larger positive cp delta is better. Mate deltas are converted to CP-equivalents.
            if (!existingEvalData || isScoreBetter(eval_delta_white_pov, existingEvalData.score_obj, currentTurn)) {
                currentPlayerPieceBestEvals.evals.set(fromSq, { bestToSq: toSq, score_obj: eval_delta_white_pov, uci: moveUci });
                requestRedraw();
            }
        }
    }
}

function parseBestmoveLine(line) {
    const parts = line.split(' ');
    const analysisTypeCompleted = analysisManager.currentAnalysisType;
    analysisManager.isProcessing = false;
    analysisManager.currentAnalysisType = null;

    if (analysisTypeCompleted === 'continuous') {
        if (currentNode && analysisManager.lastContinuousFen === currentNode.fen) {
            if (analysisManager.lastScore && currentNode.raw_score === null) {
                currentNode.raw_score = analysisManager.lastScore; // White's POV
                currentNode.white_percentage = scoreToWhitePercentage(currentNode.raw_score, EVAL_CLAMP_LIMIT);
                currentNode.calculateAndSetMoveQuality();
                treeLayout.needsRedraw = true;
            }
            if (helpersVisible) highlightedEngineMove = uciToMoveObject(analysisManager.lastBestMove || parts[1]);
        }
        analysisManager.lastScore = null; analysisManager.lastBestMove = null;
    } else if (analysisTypeCompleted === 'best_moves') {
        if (currentBestMovesResult && currentBestMovesResult.fen === analysisManager.lastBestMovesFen) {
            if (currentBestMovesResult.moves.length > 0) {
                currentBestMovesResult.moves.sort((a, b) => a.rank - b.rank);
                currentBestMoveIndex = 0;
                highlightedEngineMove = uciToMoveObject(currentBestMovesResult.moves[0].move);
                updateShowBestButtonText();
                statusMessage(`Best ${currentBestMoveIndex + 1}/${currentBestMovesResult.moves.length}: ${currentBestMovesResult.moves[0].san} (${currentBestMovesResult.moves[0].score_str})`);
            } else {
                statusMessage("Engine found no moves for best_moves."); currentBestMoveIndex = -1; highlightedEngineMove = null; updateShowBestButtonText("No moves found");
            }
        }
    } else if (analysisTypeCompleted === 'selected_piece_moves') {
        statusMessage(`Evaluations for ${analysisManager.lastSelectedPieceSq} ready.`);
        requestRedraw();
    } else if (analysisTypeCompleted === 'player_pieces_best_moves') {
        console.log("Player pieces best moves analysis complete:", currentPlayerPieceBestEvals.evals);
        statusMessage("Piece outlines updated.");
        requestRedraw();
    }

    if (analysisManager.requestQueue.length === 0 && currentNode && (analysisTypeCompleted === 'best_moves' || analysisTypeCompleted === 'selected_piece_moves' || analysisTypeCompleted === 'player_pieces_best_moves')) {
        requestAnalysis(currentNode.fen, 'continuous');
    }
    processAnalysisQueue();
}

// --- Board UI ---
function initBoard() {
    boardContainer.innerHTML = ''; coordContainer.innerHTML = '';
    const sqSize = boardContainer.offsetWidth / 8;
    for (let r = 7; r >= 0; r--) {
        const rankCoord = document.createElement('div'); rankCoord.className = 'rank-coord'; rankCoord.textContent = r + 1;
        rankCoord.style.color = (r % 2 !== 0) ? '#b99473' : '#eedcbe';
        rankCoord.style.width = `${sqSize * 0.2}px`; rankCoord.style.height = `${sqSize}px`;
        rankCoord.style.left = `0px`; rankCoord.style.top = `${(7 - r) * sqSize}px`;
        coordContainer.appendChild(rankCoord);
        for (let f = 0; f < 8; f++) {
            const square = document.createElement('div');
            const squareName = String.fromCharCode('a'.charCodeAt(0) + f) + (r + 1);
            square.id = `sq-${squareName}`; square.className = `square ${(r + f) % 2 === 0 ? 'dark' : 'light'}`;
            square.dataset.square = squareName;
            square.addEventListener('dragover', handleDragOver); square.addEventListener('dragleave', handleDragLeave);
            square.addEventListener('drop', handleDrop); square.addEventListener('click', handleSquareClick);
            boardContainer.appendChild(square);
            if (r === 0) {
                const fileCoord = document.createElement('div'); fileCoord.className = 'file-coord'; fileCoord.textContent = String.fromCharCode('a'.charCodeAt(0) + f);
                fileCoord.style.color = (f % 2 !== 0) ? '#eedcbe' : '#b99473';
                fileCoord.style.width = `${sqSize}px`; fileCoord.style.height = `${sqSize * 0.2}px`;
                fileCoord.style.left = `${f * sqSize}px`; fileCoord.style.bottom = `0px`;
                coordContainer.appendChild(fileCoord);
            }
        }
    }
    drawBoard();
}

function drawHighlights() {
    document.querySelectorAll('.highlight-overlay, .legal-move-marker, .board-badge, .selected-piece-move-eval-overlay').forEach(el => el.remove());
    if (lastMoveDisplayed && !animatingPieceInfo) { addHighlight(lastMoveDisplayed.from, 'highlight-last-move'); addHighlight(lastMoveDisplayed.to, 'highlight-last-move'); }
    if (highlightedEngineMove && helpersVisible) { addHighlight(highlightedEngineMove.from, 'highlight-engine-move'); addHighlight(highlightedEngineMove.to, 'highlight-engine-move'); }
    if (helpersVisible && selectedSquare && selectedPieceEvaluations && selectedPieceEvaluations.fen === game.fen() && selectedPieceEvaluations.pieceSq === selectedSquare) {
        selectedPieceEvaluations.evaluations.forEach((evalData, toSqKey) => {
            const squareElement = document.getElementById(`sq-${toSqKey}`);
            if (squareElement) {
                const color = getMoveEvalColor(evalData.score_obj, game.turn());
                const evalOverlay = document.createElement('div'); evalOverlay.className = 'selected-piece-move-eval-overlay';
                evalOverlay.style.backgroundColor = color; squareElement.appendChild(evalOverlay);
            }
        });
    }
    if (selectedSquare) {
        addHighlight(selectedSquare, 'highlight-selected');
        legalMovesForSelected.forEach(move => addLegalMoveMarker(move));
    }
    if (helpersVisible && currentNode && currentNode.move && currentNode.move_quality) { drawBoardBadge(currentNode.move.to, currentNode.move_quality); }
}

function addHighlight(squareName, className) {
    const squareElement = document.getElementById(`sq-${squareName}`);
    if (squareElement) { const h = document.createElement('div'); h.className = `highlight-overlay ${className}`; squareElement.appendChild(h); }
}

function addLegalMoveMarker(move) {
    const toSq = move.to; const sqEl = document.getElementById(`sq-${toSq}`);
    if (sqEl) {
        const c = document.createElement('div'); c.className = 'legal-move-marker';
        const m = document.createElement('div'); m.className = (game.get(toSq) !== null || move.flags.includes('e')) ? 'legal-move-ring' : 'legal-move-dot';
        c.appendChild(m); sqEl.appendChild(c);
    }
}

function drawBoardBadge(squareName, quality) {
    const sqEl = document.getElementById(`sq-${squareName}`); const color = TREE_MOVE_QUALITY_COLORS[quality]; const icon = `${BADGE_IMAGE_PATH}/${quality.toLowerCase()}.png`;
    if (sqEl && color) {
        const b = document.createElement('div'); b.className = `board-badge badge-${quality}`; b.style.backgroundColor = color;
        const i = document.createElement('img'); i.src = icon; i.alt = quality; i.style.width = `${BOARD_BADGE_IMAGE_SIZE.width}px`; i.style.height = `${BOARD_BADGE_IMAGE_SIZE.height}px`;
        b.appendChild(i); sqEl.appendChild(b);
    }
}

// --- Interaction Handlers ---
function handleSquareClick(event) {
    if (promotionState.pending) return;
    const clickedSqEl = event.currentTarget; const clickedSqName = clickedSqEl.dataset.square;
    if (selectedSquare === clickedSqName) { clearSelection(); return; }
    const piece = game.get(clickedSqName);
    if (selectedSquare) { attemptMove(selectedSquare, clickedSqName); }
    else if (piece && piece.color === game.turn()) { selectSquare(clickedSqName); }
    else { clearSelection(); }
}

function handleDragStart(event) {
    if (promotionState.pending) { event.preventDefault(); return; }
    const pieceEl = event.target; const sqName = pieceEl.dataset.square;
    if (sqName !== selectedSquare) { selectSquare(sqName); }
    if (!selectedSquare) { event.preventDefault(); return; }
    draggingPieceElement = pieceEl; dragStartSquare = sqName;
    event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', sqName);
    setTimeout(() => pieceEl.classList.add('dragging'), 0);
}
function handleDragOver(event) { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; event.currentTarget.classList.add('drag-over-highlight'); }
function handleDragLeave(event) { event.currentTarget.classList.remove('drag-over-highlight'); }
function handleDrop(event) {
    event.preventDefault(); if (promotionState.pending) return;
    event.currentTarget.classList.remove('drag-over-highlight');
    const fromSq = event.dataTransfer.getData('text/plain'); const toSq = event.currentTarget.dataset.square;
    if (fromSq && toSq && fromSq !== toSq) { attemptMove(fromSq, toSq); }
    else { clearSelection(); if (draggingPieceElement) draggingPieceElement.classList.remove('dragging'); draggingPieceElement = null; }
}
function handleDragEnd(event) { if (draggingPieceElement) draggingPieceElement.classList.remove('dragging'); draggingPieceElement = null; dragStartSquare = null; }

function selectSquare(squareName) {
    const piece = game.get(squareName); const currentTurn = game.turn();
    clearSelection();
    if (!piece || piece.color !== currentTurn) return;
    selectedSquare = squareName; legalMovesForSelected = game.moves({ square: squareName, verbose: true });
    statusMessage(`${squareName} selected`);
    if (helpersVisible && legalMovesForSelected.length > 0) {
        const fen = game.fen(); const uciMoves = legalMovesForSelected.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
        if (analysisManager.currentAnalysisType === 'selected_piece_moves' && (analysisManager.lastSelectedPieceFen !== fen || analysisManager.lastSelectedPieceSq !== squareName)) {
            engine.postMessage('stop'); analysisManager.isProcessing = false;
        }
        requestAnalysis(fen, 'selected_piece_moves', { searchMovesUci: uciMoves, pieceSq: squareName });
    }
    drawHighlights();
}

function clearSelection() {
    const hadSelection = selectedSquare !== null; selectedSquare = null; legalMovesForSelected = [];
    if (draggingPieceElement) { draggingPieceElement.classList.remove('dragging'); draggingPieceElement = null; }
    if (hadSelection && analysisManager.currentAnalysisType === 'selected_piece_moves') {
        if (engine) engine.postMessage('stop'); analysisManager.isProcessing = false; analysisManager.currentAnalysisType = null;
        if (analysisManager.requestQueue.length === 0 && currentNode && engine) {
            requestAnalysis(currentNode.fen, 'continuous'); processAnalysisQueue();
        }
    }
    selectedPieceEvaluations = null; drawHighlights();
}

function attemptMove(fromSq, toSq) {
    const piece = game.get(fromSq);
    if (!piece || !selectedSquare || fromSq !== selectedSquare) { clearSelection(); return; }

    console.log("Attempting move (original target):", fromSq, toSq, "Piece:", piece.type);
    console.log("FEN before move:", game.fen());
    console.log("White castling rights:", game.getCastlingRights('w'));
    console.log("Black castling rights:", game.getCastlingRights('b'));
    const isKingMove = piece.type === 'k';
    let isPotentialTwoSquareCastle = false;
    if (isKingMove) {
        isPotentialTwoSquareCastle = Math.abs(fromSq.charCodeAt(0) - toSq.charCodeAt(0)) === 2;
    }
    console.log("Castling check: King moving two squares?", isPotentialTwoSquareCastle);

    let moveData = { from: fromSq, to: toSq, promotion: undefined };
    let legalMove = legalMovesForSelected.find(m => m.to === toSq && m.from === fromSq); // Standard move check

    if (!legalMove && isKingMove) { // If standard move not found, and it's a king, check for castling by clicking/dropping on rook
        const targetPiece = game.get(toSq);
        if (targetPiece && targetPiece.type === 'r' && targetPiece.color === piece.color) {
            // King targeting a friendly rook.
            console.log("King targeted friendly rook at", toSq);
            const kingStartFile = fromSq.charCodeAt(0);
            const rookFile = toSq.charCodeAt(0);
            let kingDestSq = null;

            // Determine potential king destination for castling
            // This logic assumes standard castling rules where king moves two squares.
            // Chess960 castling moves in chess.js are represented by king moving to its final square.
            if (rookFile > kingStartFile) { // Potential kingside (rook is to the right of the king)
                kingDestSq = String.fromCharCode(kingStartFile + 2) + fromSq[1];
                console.log("Potential kingside castle, king destination:", kingDestSq);
            } else { // Potential queenside (rook is to the left of the king)
                kingDestSq = String.fromCharCode(kingStartFile - 2) + fromSq[1];
                console.log("Potential queenside castle, king destination:", kingDestSq);
            }

            const castlingMove = legalMovesForSelected.find(m =>
                m.from === fromSq &&
                m.to === kingDestSq &&
                (m.flags.includes('k') || m.flags.includes('q'))
            );

            if (castlingMove) {
                console.log("Found matching castling move in legalMovesForSelected:", castlingMove);
                legalMove = castlingMove; // Found the castling move
                moveData.to = kingDestSq; // Update moveData to king's actual destination
                console.log("Updated moveData.to for castling:", moveData.to);
            } else {
                console.log("No matching castling move found for king dest", kingDestSq, "from", fromSq);
            }
        }
    }

    if (legalMove) {
        // Log the final determined move before promotion check
        console.log("Final determined move for processing:", legalMove, "Target square for commit:", legalMove.to);
        // Check for promotion based on the *actual* 'to' square of the legalMove
        const isPromotion = (legalMove.piece === 'p') &&
                            ((game.turn() === 'w' && legalMove.to[1] === '8') ||
                             (game.turn() === 'b' && legalMove.to[1] === '1'));

        if (isPromotion) {
            console.log("Promotion detected for move to", legalMove.to);
            showPromotionPopup(legalMove.from, legalMove.to, game.turn(), (choice) => {
                if (choice) {
                    console.log("Promotion choice:", choice);
                    commitMove({ from: legalMove.from, to: legalMove.to, promotion: choice });
                } else {
                    console.log("Promotion cancelled");
                    statusMessage("Promotion cancelled."); clearSelection();
                }
            });
        } else {
            console.log("No promotion. Committing move:", { from: legalMove.from, to: legalMove.to, promotion: legalMove.promotion });
            // Use from/to from legalMove for commitMove
            commitMove({ from: legalMove.from, to: legalMove.to, promotion: legalMove.promotion });
        }
    } else {
        console.log("No legal move found for", fromSq, "to", toSq, "(original or derived castle target)");
        statusMessage("Illegal move."); playSound("illegal_move"); clearSelection();
    }
}

function commitMove(moveData) {
    console.log("Committing move with data:", moveData);
    const moveResult = game.move(moveData);
    if (moveResult === null) {
        console.log("Move result: null (illegal move). Castling attempt might have failed.");
        statusMessage("Illegal move."); playSound("illegal_move"); clearSelection(); return;
    }
    console.log("Move result:", moveResult);
    console.log("FEN after move attempt:", game.fen());
    statusMessage(`Played ${moveResult.san}`);
    const parentNode = currentNode; const boardBeforeMoveFen = parentNode.fen;
    let existingChild = parentNode.children.find(c => c.move && c.move.from === moveResult.from && c.move.to === moveResult.to && c.move.promotion === moveResult.promotion);
    let newNode;
    if (existingChild) { newNode = existingChild; if (newNode.fen !== game.fen()) newNode.fen = game.fen(); }
    else { newNode = new GameNode(game.fen(), moveResult, parentNode, null); parentNode.addChild(newNode); treeLayout.needsRedraw = true; }
    const previousNode = currentNode; currentNode = newNode;
    playSoundForMove(boardBeforeMoveFen, game.fen(), moveResult);
    startPieceAnimation(moveResult, previousNode);
    lastMoveDisplayed = { from: moveResult.from, to: moveResult.to };
    highlightedEngineMove = null; currentBestMovesResult = null; selectedPieceEvaluations = null;
    currentBestMoveIndex = -1; updateShowBestButtonText(); clearSelection();
    liveRawScore = null; targetWhitePercentage = 50.0; // Reset live eval
    requestAnalysis(currentNode.fen, 'continuous');
    if (helpersVisible) requestPlayerPiecesBestMoves(currentNode.fen);
    requestRedraw();
}

function drawBoard() {
    document.querySelectorAll('.piece').forEach(p => p.remove());
    const squares = game.board();
    const currentFen = game.fen();
    const turn = game.turn();

    for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
            const squareInfo = squares[r][f];
            if (squareInfo) {
                const squareName = String.fromCharCode('a'.charCodeAt(0) + f) + (8 - r);
                if (animatingPieceInfo && (squareName === animatingPieceInfo.fromSq || squareName === animatingPieceInfo.toSq)) continue;

                const pieceElement = document.createElement('div');
                pieceElement.className = 'piece'; pieceElement.id = `piece-${squareName}`;
                pieceElement.style.backgroundImage = `url('${PIECE_IMAGE_PATH}/${squareInfo.color}${squareInfo.type.toUpperCase()}.png')`;
                pieceElement.dataset.square = squareName;
                pieceElement.style.width = '100%'; pieceElement.style.height = '100%';
                pieceElement.style.backgroundSize = 'contain'; pieceElement.style.backgroundRepeat = 'no-repeat';
                pieceElement.style.backgroundPosition = 'center';

                if (squareInfo.color === turn) {
                    pieceElement.draggable = true; pieceElement.addEventListener('dragstart', handleDragStart);
                    pieceElement.addEventListener('dragend', handleDragEnd); pieceElement.style.cursor = 'grab';
                } else {
                    pieceElement.draggable = false; pieceElement.style.cursor = 'default';
                }

                // Add piece outline logic
                if (helpersVisible && squareInfo.color === turn &&
                    currentPlayerPieceBestEvals && currentPlayerPieceBestEvals.fen === currentFen) {
                    const evalData = currentPlayerPieceBestEvals.evals.get(squareName);
                    if (evalData && evalData.score_obj) {
                        const outlineColor = getMoveEvalColor(evalData.score_obj, turn);
                        const blurRadius = '2.5px';
                        pieceElement.style.filter = `drop-shadow(0 0 ${blurRadius} ${outlineColor}) drop-shadow(0 0 ${blurRadius} ${outlineColor})`; // Double for intensity
                    } else {
                        pieceElement.style.filter = 'none';
                    }
                } else {
                    pieceElement.style.filter = 'none';
                }

                const squareElement = document.getElementById(`sq-${squareName}`);
                if (squareElement) {
                    const existingPiece = squareElement.querySelector('.piece');
                    if (existingPiece) existingPiece.remove();
                    squareElement.appendChild(pieceElement);
                }
            }
        }
    }
    if (animatingPieceInfo && animatingPieceInfo.element) {
        animatingPieceInfo.element.style.width = `${boardContainer.offsetWidth / 8}px`;
        animatingPieceInfo.element.style.height = `${boardContainer.offsetHeight / 8}px`;
        boardContainer.appendChild(animatingPieceInfo.element);
    }
}

// --- Animation ---
function startPieceAnimation(move, previousNode) {
    const fromSq = move.from; const toSq = move.to;
    const fromSqEl = document.getElementById(`sq-${fromSq}`);
    let pieceEl = fromSqEl ? fromSqEl.querySelector('.piece') : document.getElementById(`piece-${fromSq}`);
    if (!pieceEl) { drawBoard(); drawHighlights(); return; }
    const boardRect = boardContainer.getBoundingClientRect(); const sqSize = boardRect.width / 8;
    const startCoords = squareToPixelCoords(fromSq, sqSize, boardRect); const endCoords = squareToPixelCoords(toSq, sqSize, boardRect);
    pieceEl.style.position = 'absolute'; pieceEl.style.left = `${startCoords.x}px`; pieceEl.style.top = `${startCoords.y}px`;
    pieceEl.style.width = `${sqSize}px`; pieceEl.style.height = `${sqSize}px`; pieceEl.style.zIndex = '1000';
    const pieceInfo = game.get(toSq);
    if (pieceInfo) pieceEl.style.backgroundImage = `url('${PIECE_IMAGE_PATH}/${pieceInfo.color}${pieceInfo.type.toUpperCase()}.png')`;
    void pieceEl.offsetWidth; boardContainer.appendChild(pieceEl);
    animatingPieceInfo = {
        element: pieceEl, fromSq: fromSq, toSq: toSq, startTime: performance.now(),
        startX: startCoords.x, startY: startCoords.y, endX: endCoords.x, endY: endCoords.y,
        duration: ANIMATION_DURATION_MS, moveData: move, previousNode: previousNode
    };
    requestAnimationFrame(updateAnimation); drawBoard(); drawHighlights();
}
function updateAnimation(timestamp) {
    if (!animatingPieceInfo) return;
    const progress = Math.min(1.0, (timestamp - animatingPieceInfo.startTime) / animatingPieceInfo.duration);
    const eased = 0.5 * (1 - Math.cos(Math.PI * progress));
    animatingPieceInfo.element.style.left = `${animatingPieceInfo.startX + (animatingPieceInfo.endX - animatingPieceInfo.startX) * eased}px`;
    animatingPieceInfo.element.style.top = `${animatingPieceInfo.startY + (animatingPieceInfo.endY - animatingPieceInfo.startY) * eased}px`;
    if (progress < 1.0) requestAnimationFrame(updateAnimation); else finishAnimation();
}
function finishAnimation() {
    if (!animatingPieceInfo) return;
    const { element: pieceEl, toSq } = animatingPieceInfo;
    const targetSqEl = document.getElementById(`sq-${toSq}`);
    pieceEl.style.position = ''; pieceEl.style.left = ''; pieceEl.style.top = ''; pieceEl.style.zIndex = ''; pieceEl.style.width = ''; pieceEl.style.height = '';
    if (targetSqEl) {
        const existing = targetSqEl.querySelector('.piece');
        if (existing && existing !== pieceEl) existing.remove();
        targetSqEl.appendChild(pieceEl);
    } else pieceEl.remove();
    animatingPieceInfo = null; requestRedraw();
}

// --- Promotion ---
function showPromotionPopup(fromSq, toSq, color, callback) {
    promotionState = { pending: true, fromSq, toSq, callback };
    promotionPopup.innerHTML = ''; promotionPopup.classList.remove('hidden');
    const pieces = ['q', 'r', 'b', 'n']; const sqSize = boardContainer.offsetWidth / 8;
    const targetSqEl = document.getElementById(`sq-${toSq}`);
    if (targetSqEl) {
        const rect = targetSqEl.getBoundingClientRect(); const boardRect = boardContainer.getBoundingClientRect();
        promotionPopup.style.left = `${rect.left - boardRect.left}px`;
        if (toSq[1] === '8') { promotionPopup.style.top = `${rect.top - boardRect.top}px`; promotionPopup.style.bottom = 'auto'; }
        else { promotionPopup.style.bottom = `${boardRect.height - (rect.bottom - boardRect.top)}px`; promotionPopup.style.top = 'auto'; }
        promotionPopup.style.width = `${sqSize}px`;
    }
    pieces.forEach(pType => {
        const choice = document.createElement('div'); choice.className = 'promo-choice';
        choice.style.backgroundImage = `url('${PIECE_IMAGE_PATH}/${color}${pType.toUpperCase()}.png')`;
        choice.dataset.piece = pType; choice.style.width = `${sqSize*0.9}px`; choice.style.height = `${sqSize*0.9}px`;
        choice.addEventListener('click', handlePromotionChoice); promotionPopup.appendChild(choice);
    });
    statusMessage("Select promotion piece"); requestRedraw();
    document.addEventListener('click', handleClickOutside, { once: true, capture: true });
}
function handlePromotionChoice(event) {
    const chosen = event.currentTarget.dataset.piece; hidePromotionPopup();
    if (promotionState.callback) promotionState.callback(chosen);
    promotionState = { pending: false, fromSq: null, toSq: null, callback: null };
    document.removeEventListener('click', handleClickOutside, { capture: true });
}
function hidePromotionPopup() {
    promotionPopup.classList.add('hidden'); promotionState.pending = false;
    document.removeEventListener('click', handleClickOutside, { capture: true });
}

// --- Game Tree ---
class GameNode {
    constructor(fen, move = null, parent = null, raw_score = null) { // raw_score is White's POV
        this.fen = fen; this.move = move; this.parent = parent; this.children = [];
        this.raw_score = raw_score; // White's POV
        this.white_percentage = scoreToWhitePercentage(raw_score, EVAL_CLAMP_LIMIT);
        this.move_quality = null; this.id = `node-${Date.now()}-${Math.random()}`;
    }
    addChild(childNode) { this.children.push(childNode); }
    get_ply() { let c=0; let n=this; while(n.parent){c++; n=n.parent;} return c; }
    get_san() { return this.move ? this.move.san : (this.parent ? "???" : "root"); }
    calculateAndSetMoveQuality() {
        if (!this.parent || this.parent.white_percentage === null || this.white_percentage === null) { this.move_quality = null; return; }
        const wp_before = this.parent.white_percentage; const wp_after = this.white_percentage;
        try {
            const parentBoard = new Chess(this.parent.fen); parentBoard.chess960 = true;
            this.move_quality = classifyMoveQuality(wp_before, wp_after, parentBoard.turn());
        } catch(e) { this.move_quality = null; }
    }
}

function layoutAndDrawTree() {
    if (!gameRoot) return;
    treeSvg.innerHTML = ''; treeNodeElements.clear(); treeLayout.nodePositions.clear();
    treeLayout.maxX = 0; treeLayout.maxY = 0;
    const treeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    treeGroup.id = 'tree-content-group'; treeSvg.appendChild(treeGroup);
    function calculateLayout(node, x, level, siblingIndex, parentY) {
        node.layoutX = x;
        let estY = (level === 0) ? VERT_SPACING * 1.5 : parentY + (siblingIndex - (node.parent.children.length -1) / 2) * VERT_SPACING * 1.2;
        if (level > 0 && node.parent.children.length > 1) {
            const prevSib = node.parent.children[siblingIndex - 1];
            if (prevSib && prevSib.layoutY !== undefined) estY = Math.max(estY, prevSib.layoutY + VERT_SPACING * 0.8);
        }
        node.layoutY = estY;
        treeLayout.nodePositions.set(node, { x: node.layoutX, y: node.layoutY });
        treeLayout.maxX = Math.max(treeLayout.maxX, node.layoutX + HORIZ_SPACING);
        treeLayout.maxY = Math.max(treeLayout.maxY, node.layoutY + VERT_SPACING);
        node.children.forEach((child, i) => calculateLayout(child, x + HORIZ_SPACING, level + 1, i, node.layoutY));
    }
    function drawNodeRecursive(node) { // Renamed to avoid conflict with an assumed outer `drawNode`
        const pos = treeLayout.nodePositions.get(node); if (!pos) return;
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.classList.add('tree-node'); g.dataset.nodeId = node.id;
        if (node === currentNode) g.classList.add('current'); if (node === gameRoot) g.classList.add('root');
        g.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);
        g.addEventListener('click', handleTreeNodeClick);
        let nodeEl, textX = 0; const piece = getMovedPieceInfo(node);
        if (piece) {
            nodeEl = document.createElementNS("http://www.w3.org/2000/svg", "image");
            nodeEl.setAttributeNS('http://www.w3.org/1999/xlink', 'href', `${PIECE_IMAGE_PATH}/${piece.color}${piece.type.toUpperCase()}.png`);
            nodeEl.setAttribute('width', TREE_PIECE_SIZE); nodeEl.setAttribute('height', TREE_PIECE_SIZE);
            nodeEl.setAttribute('x', -TREE_PIECE_SIZE/2); nodeEl.setAttribute('y', -TREE_PIECE_SIZE/2);
            textX = TREE_PIECE_SIZE/2 + TEXT_OFFSET_X;
        } else {
            nodeEl = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            const r = (node === gameRoot) ? 6 : 3;
            nodeEl.setAttribute('r', r); nodeEl.setAttribute('cx',0); nodeEl.setAttribute('cy',0); textX = r + TEXT_OFFSET_X;
        }
        g.appendChild(nodeEl);
        if (node.move) {
            const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
            t.textContent = node.get_san(); t.setAttribute('x', textX); t.setAttribute('y', TEXT_OFFSET_Y - TREE_PIECE_SIZE/2 + 4);
            g.appendChild(t);
        }
        if (helpersVisible && node.move_quality && TREE_MOVE_QUALITY_COLORS[node.move_quality]) {
            const b = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            b.classList.add('tree-badge'); b.setAttribute('r', TREE_BADGE_RADIUS);
            b.setAttribute('cx', TREE_PIECE_SIZE/2 - TREE_BADGE_RADIUS); b.setAttribute('cy', TREE_PIECE_SIZE/2 - TREE_BADGE_RADIUS);
            b.setAttribute('fill', TREE_MOVE_QUALITY_COLORS[node.move_quality]); g.appendChild(b);
        }
        treeNodeElements.set(node.id, { group: g, node: node }); treeGroup.appendChild(g);
        node.children.forEach(child => {
            const childPos = treeLayout.nodePositions.get(child);
            if (childPos) {
                const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
                l.classList.add('tree-line'); l.setAttribute('x1',pos.x); l.setAttribute('y1',pos.y); l.setAttribute('x2',childPos.x); l.setAttribute('y2',childPos.y);
                treeGroup.insertBefore(l, g); drawNodeRecursive(child);
            }
        });
    }
    calculateLayout(gameRoot, HORIZ_SPACING / 2, 0, 0, 0);
    const padding = 20; treeSvg.style.width = `${treeLayout.maxX + padding}px`; treeSvg.style.height = `${treeLayout.maxY + padding}px`;
    drawNodeRecursive(gameRoot); treeLayout.needsRedraw = false; scrollToNode(currentNode);
}
function getMovedPieceInfo(node) {
    if (!node || !node.move || !node.parent) return null;
    try { const b=new Chess(node.parent.fen); b.chess960=true; const p=b.get(node.move.from); return p ? {color:p.color,type:p.type}:null; } catch(e){return null;}
}
function handleTreeNodeClick(event) {
    const nodeInfo = treeNodeElements.get(event.currentTarget.dataset.nodeId);
    if (nodeInfo && nodeInfo.node && nodeInfo.node !== currentNode) {
        const prevNode = currentNode; currentNode = nodeInfo.node;
        try {
            game = new Chess(currentNode.fen); game.chess960 = true;
            statusMessage(`Navigated to ply ${currentNode.get_ply()}`);
            liveRawScore = currentNode.raw_score; // White's POV
            targetWhitePercentage = scoreToWhitePercentage(liveRawScore, EVAL_CLAMP_LIMIT);
            displayedWhitePercentage = targetWhitePercentage; // Snap eval bar
            clearSelection();
            lastMoveDisplayed = currentNode.move ? { from: currentNode.move.from, to: currentNode.move.to } : null;
            highlightedEngineMove = null; currentBestMovesResult = null; currentBestMoveIndex = -1; updateShowBestButtonText();
            requestAnalysis(currentNode.fen, 'continuous');
            if (helpersVisible) requestPlayerPiecesBestMoves(currentNode.fen);
            drawBoard(); drawHighlights(); drawEvalBar(); updatePlot();
            document.querySelectorAll('.tree-node.current').forEach(el => el.classList.remove('current'));
            event.currentTarget.classList.add('current');
            playSoundForMove(currentNode.parent?.fen, currentNode.fen, currentNode.move);
            animatingPieceInfo = null;
        } catch (e) { console.error("Error navigating tree node:", e); statusMessage("Error: Invalid FEN."); currentNode = prevNode; }
    }
}
function scrollToNode(node) {
    if (!node) return; const elInfo = Array.from(treeNodeElements.values()).find(i => i.node === node);
    if (!elInfo || !elInfo.group) return; const r = elInfo.group.getBBox();
    const x = r.x + r.width/2; const y = r.y + r.height/2;
    const cw = treePanelContainer.clientWidth; const ch = treePanelContainer.clientHeight;
    treePanelContainer.scrollTo({ left: Math.max(0, Math.min(x-cw/2, treePanelContainer.scrollWidth-cw)), top: Math.max(0, Math.min(y-ch/2, treePanelContainer.scrollHeight-ch)), behavior: 'smooth' });
}
function setupTreeDrag() {
    let dragging=false, lastX, lastY;
    treeSvg.addEventListener('mousedown', (e) => { if(e.button!==0 || !(e.target===treeSvg || e.target.id==='tree-content-group')) return; dragging=true; treeSvg.classList.add('grabbing'); lastX=e.clientX; lastY=e.clientY; e.preventDefault(); });
    document.addEventListener('mousemove', (e) => { if(!dragging) return; treePanelContainer.scrollLeft-=(e.clientX-lastX); treePanelContainer.scrollTop-=(e.clientY-lastY); lastX=e.clientX; lastY=e.clientY; });
    document.addEventListener('mouseup', (e) => { if(dragging && e.button===0){dragging=false; treeSvg.classList.remove('grabbing');} });
}

// --- Eval Plot ---
function initPlot() {
    const ctx = plotCanvas.getContext('2d');
    currentEvalPlot = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [{ label: 'White Advantage %', data: [], borderColor: 'rgba(255,255,255,0.8)', backgroundColor: 'rgba(255,255,255,0.3)', borderWidth: 1.5, fill: true, tension: 0.1, pointRadius: 0, pointHoverRadius: 3 }] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false, min: 0, max: 100, ticks: { display: false }, grid: { color: 'rgba(128,128,128,0.4)', borderDash: [2,2]}}, x: { ticks: { display: false }, grid: { display: false }}}, plugins: { legend: { display: false }, tooltip: { enabled: false }}, animation: { duration: 150 }, layout: { padding: 1 }}});
}
function updatePlot() {
    if (!currentEvalPlot || !currentNode || !plotVisible) return;
    const path = getNodePath(currentNode);
    const labels = path.map(n => n.get_ply()); const data = path.map(n => n.white_percentage ?? 50.0);
    currentEvalPlot.data.labels = labels; currentEvalPlot.data.datasets[0].data = data;
    if (currentEvalPlot.data.datasets.length === 1) currentEvalPlot.data.datasets.push({ label: 'Centerline', data: labels.map(()=>50), borderColor: 'rgba(255,165,0,0.7)', borderWidth:1.5, fill:false, pointRadius:0, borderDash:[5,5], order:0 });
    else if (currentEvalPlot.data.datasets.length > 1) currentEvalPlot.data.datasets[1].data = labels.map(()=>50);
    currentEvalPlot.update();
}
function getNodePath(node) { const p=[]; let c=node; while(c){p.push(c);c=c.parent;} return p.reverse(); }

// --- UI Updates & Helpers ---
function statusMessage(msg) { statusMessageEl.textContent = msg; }
function engineStatusMessage(msg) { engineStatusMessageEl.textContent = msg; }
function requestRedraw() { drawHighlights(); drawEvalBar(); updatePlot(); if(treeLayout.needsRedraw) layoutAndDrawTree(); drawBoard(); /* drawBoard for piece outlines */ }
function drawEvalBar() { evalBarWhite.style.height = `${displayedWhitePercentage}%`; evalBarBlack.style.height = `${100.0 - displayedWhitePercentage}%`; }
function updateEvalBarAnimation() {
    const diff = targetWhitePercentage - displayedWhitePercentage;
    if (Math.abs(diff) > 0.1) { displayedWhitePercentage += diff * 0.1; drawEvalBar(); }
    else if (displayedWhitePercentage !== targetWhitePercentage) { displayedWhitePercentage = targetWhitePercentage; drawEvalBar(); }
}
function updateShowBestButtonText(customText = null) {
    if (customText) { showBestMoveBtn.textContent = customText; return; }
    let txt = "Show Best";
    if (analysisManager.currentAnalysisType === 'best_moves' && analysisManager.isProcessing) txt = "Analyzing...";
    else if (currentBestMovesResult && currentBestMovesResult.fen === currentNode?.fen) {
        if (currentBestMovesResult.error) txt = "Analysis Failed";
        else if (currentBestMovesResult.moves?.length > 0 && currentBestMoveIndex !== -1) txt = `Showing ${currentBestMoveIndex+1}/${currentBestMovesResult.moves.length}`;
        else if (currentBestMovesResult.moves?.length > 0) txt = "Moves Found";
        else if (currentBestMovesResult.moves?.length === 0) txt = "No Moves Found";
    } else if (!engine) txt = "Engine Off"; else if (game.game_over()) txt = "Game Over";
    showBestMoveBtn.textContent = txt;
}

// --- Sounds ---
function initSounds() {
    const sFiles = {"white_move":"move-self.mp3", "black_move":"move-opponent.mp3", "capture":"capture.mp3", "check":"move-check.mp3", "checkmate":"game-end.mp3", "stalemate":"game-end.mp3", "illegal_move":"illegal.mp3"};
    for (const k in sFiles) { try { sounds[k] = new Audio(`${SOUND_PATH}/${sFiles[k]}`); sounds[k].load(); } catch(e){} }
}
function playSound(key) { if(sounds[key]){ sounds[key].currentTime=0; sounds[key].play().catch(e=>{});}}
function playSoundForMove(fenBefore, fenAfter, move) {
    if (!move) return; try {
        const bAfter = new Chess(fenAfter); bAfter.chess960=true; const turnBefore = new Chess(fenBefore).turn(); let key=null;
        if(bAfter.in_checkmate()) key="checkmate"; else if(bAfter.in_stalemate()||bAfter.in_draw()) key="stalemate";
        else if(bAfter.in_check()) key="check"; else if(move.flags.includes('c')||move.flags.includes('e')) key="capture";
        else key = (turnBefore === 'w') ? "white_move" : "black_move";
        if(key) playSound(key);
    } catch(e){}
}

// --- Helpers ---
function scoreToWhitePercentage(score_white_pov, clamp = EVAL_CLAMP_LIMIT) { // Expects White's POV score
    if (!score_white_pov) return 50.0;
    if (score_white_pov.mate !== null) return score_white_pov.mate > 0 ? 100.0 : 0.0;
    if (score_white_pov.cp !== null) return (Math.max(-clamp, Math.min(clamp, score_white_pov.cp)) + clamp) / (2 * clamp) * 100.0;
    return 50.0;
}
function classifyMoveQuality(wpBefore, wpAfter, turn_before_move) {
    if (wpBefore === null || wpAfter === null) return null;
    let drop = (turn_before_move === 'w') ? (wpBefore - wpAfter) : ((100-wpBefore) - (100-wpAfter)); drop = Math.max(0, drop);
    if (drop <= 2) return "Best"; if (drop <= 5) return "Excellent"; if (drop <= 10) return "Good";
    if (drop <= 20) return "Inaccuracy"; if (drop <= 35) return "Mistake"; return "Blunder";
}
function formatScore(score_white_pov, turn_for_pov) { // Expects White's POV score
    if (!score_white_pov) return "N/A";
    let povScore = { cp: score_white_pov.cp, mate: score_white_pov.mate };
    if (turn_for_pov === 'b') { povScore.cp = score_white_pov.cp !== null ? -score_white_pov.cp : null; povScore.mate = score_white_pov.mate !== null ? -score_white_pov.mate : null; }
    if (povScore.mate !== null) return `Mate in ${Math.abs(povScore.mate)}`;
    if (povScore.cp !== null) return `${(povScore.cp / 100.0).toFixed(2)}`;
    return "N/A";
}
function squareToPixelCoords(sq, sz, r) { const f=sq.charCodeAt(0)-'a'.charCodeAt(0); const R=parseInt(sq[1],10)-1; return {x:f*sz, y:(7-R)*sz}; }
function uciToMoveObject(uci) { if(!uci||uci.length<4)return null; return {from:uci.substring(0,2),to:uci.substring(2,4),promotion:uci.length===5?uci.substring(4):undefined}; }
function uciToSan(uci, fen) { try{const b=new Chess(fen);b.chess960=true;const m=b.move(uci,{sloppy:true});return m?m.san:uci;}catch(e){return uci;} }

function isScoreBetter(newScore_wp, oldScore_wp, playerTurn) {
    // newScore_wp and oldScore_wp can be absolute scores (White's POV) or delta scores (White's POV).
    // The delta calculation in parseInfoLine now stores mate-vs-mate or mate-vs-cp differences in the .cp field of the delta.
    // Therefore, we primarily compare .cp values here.
    // playerTurn is still relevant if we need to consider the perspective for absolute scores,
    // but for deltas (which are already White's POV), a larger positive CP delta is always better for White.

    const getComparable = (score_obj_wp, turn_for_pov_if_absolute) => {
        // If score_obj_wp is a delta, its .cp already reflects the change.
        // If it's an absolute score, we convert to player's POV.
        // The delta calculation already put mate differences into .cp field with large magnitudes.

        if (score_obj_wp.cp !== null) {
            // If this is a delta, it's already White's POV.
            // If this is an absolute score, and playerTurn is 'b', we'd flip it.
            // However, since deltas are now used for selected_piece and player_pieces,
            // and they are calculated from White's POV, direct comparison of .cp is fine.
            // The `turn_for_pov_if_absolute` is mostly for clarity if we ever mix raw absolute scores here.
            // For player_pieces_best_moves, the eval_delta_white_pov is passed, which is White's POV.
            // A higher CP value (whether absolute for White or a positive delta for White) is better for White.
            return score_obj_wp.cp;
        }
        // Fallback for rare cases or if only mate is present (though delta logic tries to put into .cp)
        if (score_obj_wp.mate !== null) {
            // This part is less likely to be hit if delta logic correctly populates .cp for mate differences
            let mateVal = score_obj_wp.mate; // White's POV mate
            // If it's a delta with mate (e.g. eval_delta_white_pov.mate), it's already from White's POV perspective change.
            // If it's an absolute score, then playerTurn matters.
            // Let's assume for now that if .mate is here, it's an absolute score that needs POV conversion.
            if (turn_for_pov_if_absolute === 'b') mateVal = -mateVal;
            return mateVal > 0 ? (100000 - mateVal) : (-100000 - mateVal);
        }
        // If turn_for_pov_if_absolute is 'w', worse score is -200000. If 'b', better score for white is -200000 (meaning +200000 for black).
        // This needs to be consistent: higher is better for the player whose turn it is, *or* for White if comparing White POV deltas.
        // Since deltas are White POV, higher is better.
        return -200000; // Default for non-comparable or error
    };

    // When comparing deltas (which are White's POV), playerTurn isn't strictly needed for getComparable,
    // as we just want to see if new_delta.cp > old_delta.cp.
    // If these are absolute scores, playerTurn is used by getComparable to flip to current player's POV.
    // The current implementation of delta calculation stores results in .cp field for comparison.
    const newComparable = getComparable(newScore_wp, playerTurn);
    const oldComparable = getComparable(oldScore_wp, playerTurn);

    return newComparable > oldComparable;
}

function getMoveEvalColor(score_obj_white_pov, turn_to_move) { // Expects White's POV DELTA score
    let eval_points;

    if (score_obj_white_pov && score_obj_white_pov.cp !== null) {
        eval_points = score_obj_white_pov.cp;
    } else if (score_obj_white_pov && score_obj_white_pov.mate !== null) {
        // This is a fallback if a mate delta wasn't converted to CP equivalent in parseInfoLine.
        // A positive mate delta means "better for White", negative means "worse for White".
        console.warn("getMoveEvalColor received score_obj with mate property. Expected .cp for deltas.", score_obj_white_pov);
        eval_points = score_obj_white_pov.mate > 0 ? SELECTED_PIECE_MOVE_EVAL_CLAMP : -SELECTED_PIECE_MOVE_EVAL_CLAMP;
    } else {
        // No valid score information or effectively a zero delta
        return `rgba(128, 128, 128, ${SELECTED_PIECE_MOVE_MAX_ALPHA * 0.1})`; // Reduced alpha for neutral/zero delta
    }

    // Convert White's POV delta to the current player's POV delta for coloring
    // If it's White's turn, a positive delta is good (green).
    // If it's Black's turn, a positive White POV delta is bad for Black (red).
    // So, if Black's turn, we flip the sign of eval_points.
    if (turn_to_move === 'b') {
        eval_points = -eval_points;
    }

    const norm = Math.max(-1, Math.min(1, eval_points / SELECTED_PIECE_MOVE_EVAL_CLAMP));
    let r, g, b;

    // Colors: Green for positive (good for current player), Red for negative (bad for current player)
    if (norm <= 0) { // Negative or zero (bad or neutral for current player) -> Red or transitioning to Yellow/Neutral
        const t = 1 + norm; // t goes from 0 (max negative) to 1 (zero)
        r = 255;
        g = Math.round(255 * t); // From 0 (red) to 255 (yellow)
        b = 0;
    } else { // Positive (good for current player) -> Green or transitioning from Yellow/Neutral
        const t = norm; // t goes from 0 (zero) to 1 (max positive)
        r = Math.round(255 * (1-t)); // From 255 (yellow) to 0 (green)
        g = 255;
        b = 0;
    }
    const alpha = SELECTED_PIECE_MOVE_MAX_ALPHA * Math.abs(norm);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// --- Main Game Loop ---
function gameLoop(timestamp) { updateEvalBarAnimation(); requestAnimationFrame(gameLoop); }

// --- Game Reset and Setup ---
function resetGame(startPosNum) {
    currentBoardNumber = startPosNum; boardNumDisplay.textContent = `Pos: ${startPosNum}`; startTime = new Date();
    try {
        game = new Chess(); game.chess960 = true;
        game.load(getFenForChess960Pos(startPosNum));
        const startFen = game.fen();
        gameRoot = new GameNode(startFen); currentNode = gameRoot;
        selectedSquare = null; legalMovesForSelected = []; lastMoveDisplayed = null;
        highlightedEngineMove = null; currentBestMovesResult = null; selectedPieceEvaluations = null;
        currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; // Reset this
        currentBestMoveIndex = -1; liveRawScore = null; targetWhitePercentage = 50.0;
        displayedWhitePercentage = 50.0; analysisErrorMessage = null; promotionState.pending = false;
        animatingPieceInfo = null; treeLayout.needsRedraw = true; treeNodeElements.clear(); treeLayout.nodePositions.clear();
        if (engine) engine.postMessage('stop');
        analysisManager = { isProcessing: false, requestQueue: [], currentAnalysisType: null, lastContinuousFen: null, lastBestMovesFen: null, lastSelectedPieceFen: null, lastSelectedPieceSq: null, lastPlayerPiecesBestMovesFen: null, lastScore: null, lastBestMove: null };
        initBoard(); // Redraws board, which calls drawBoard()
        drawHighlights(); drawEvalBar(); updatePlot(); layoutAndDrawTree();
        requestAnalysis(startFen, 'continuous');
        if (helpersVisible) requestPlayerPiecesBestMoves(startFen);
        statusMessage(`Position ${startPosNum} set. Analyzing...`);
    } catch (error) {
        statusMessage(`Error pos ${startPosNum}.`);
        game = new Chess(); game.chess960=true; const startFen = game.fen();
        gameRoot = new GameNode(startFen); currentNode = gameRoot;
        initBoard(); drawHighlights(); drawEvalBar(); updatePlot(); layoutAndDrawTree();
        requestAnalysis(startFen, 'continuous');
        if (helpersVisible) requestPlayerPiecesBestMoves(startFen);
    }
}

const chess960Fens = [
    "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1",
    "bbqnrnkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNRNKR w KQkq - 0 1",
    "bbqnrknr/pppppppp/8/8/8/8/PPPPPPPP/BBQNRKNR w KQkq - 0 1",
    "bbqnrkrn/pppppppp/8/8/8/8/PPPPPPPP/BBQNRKRN w KQkq - 0 1",
    "bbqrnnkr/pppppppp/8/8/8/8/PPPPPPPP/BBQRNNKR w KQkq - 0 1",
    "bbqrnknr/pppppppp/8/8/8/8/PPPPPPPP/BBQRNKNR w KQkq - 0 1",
    "bbqrnkrn/pppppppp/8/8/8/8/PPPPPPPP/BBQRNKRN w KQkq - 0 1",
    "bbqrknnr/pppppppp/8/8/8/8/PPPPPPPP/BBQRKNNR w KQkq - 0 1",
    "bbqrknrn/pppppppp/8/8/8/8/PPPPPPPP/BBQRKNRN w KQkq - 0 1",
    "bbqrkrnn/pppppppp/8/8/8/8/PPPPPPPP/BBQRKRNN w KQkq - 0 1",
    "bbnqnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBNQNRKR w KQkq - 0 1",
    "bbnqrnkr/pppppppp/8/8/8/8/PPPPPPPP/BBNQRNKR w KQkq - 0 1",
    "bbnqrknr/pppppppp/8/8/8/8/PPPPPPPP/BBNQRKNR w KQkq - 0 1",
    "bbnqrkrn/pppppppp/8/8/8/8/PPPPPPPP/BBNQRKRN w KQkq - 0 1",
    "bbrqnnkr/pppppppp/8/8/8/8/PPPPPPPP/BBRQNNKR w KQkq - 0 1",
    "bbrqnknr/pppppppp/8/8/8/8/PPPPPPPP/BBRQNKNR w KQkq - 0 1",
    "bbrqnkrn/pppppppp/8/8/8/8/PPPPPPPP/BBRQNKRN w KQkq - 0 1",
    "bbrqknnr/pppppppp/8/8/8/8/PPPPPPPP/BBRQKNNR w KQkq - 0 1",
    "bbrqknrn/pppppppp/8/8/8/8/PPPPPPPP/BBRQKNRN w KQkq - 0 1",
    "bbrqkrnn/pppppppp/8/8/8/8/PPPPPPPP/BBRQKRNN w KQkq - 0 1",
    "bbnnqrkr/pppppppp/8/8/8/8/PPPPPPPP/BBNNQRKR w KQkq - 0 1",
    "bbnrqnkr/pppppppp/8/8/8/8/PPPPPPPP/BBNRQNKR w KQkq - 0 1",
    "bbnrqknr/pppppppp/8/8/8/8/PPPPPPPP/BBNRQKNR w KQkq - 0 1",
    "bbnrqkrn/pppppppp/8/8/8/8/PPPPPPPP/BBNRQKRN w KQkq - 0 1",
    "bbrnqnkr/pppppppp/8/8/8/8/PPPPPPPP/BBRNQNKR w KQkq - 0 1",
    "bbrnqknr/pppppppp/8/8/8/8/PPPPPPPP/BBRNQKNR w KQkq - 0 1",
    "bbrnqkrn/pppppppp/8/8/8/8/PPPPPPPP/BBRNQKRN w KQkq - 0 1",
    "bbrkqnnr/pppppppp/8/8/8/8/PPPPPPPP/BBRKQNNR w KQkq - 0 1",
    "bbrkqnrn/pppppppp/8/8/8/8/PPPPPPPP/BBRKQNRN w KQkq - 0 1",
    "bbrkqrnn/pppppppp/8/8/8/8/PPPPPPPP/BBRKQRNN w KQkq - 0 1",
    "bbnnrqkr/pppppppp/8/8/8/8/PPPPPPPP/BBNNRQKR w KQkq - 0 1",
    "bbnrnqkr/pppppppp/8/8/8/8/PPPPPPPP/BBNRNQKR w KQkq - 0 1",
    "bbnrkqnr/pppppppp/8/8/8/8/PPPPPPPP/BBNRKQNR w KQkq - 0 1",
    "bbnrkqrn/pppppppp/8/8/8/8/PPPPPPPP/BBNRKQRN w KQkq - 0 1",
    "bbrnnqkr/pppppppp/8/8/8/8/PPPPPPPP/BBRNNQKR w KQkq - 0 1",
    "bbrnkqnr/pppppppp/8/8/8/8/PPPPPPPP/BBRNKQNR w KQkq - 0 1",
    "bbrnkqrn/pppppppp/8/8/8/8/PPPPPPPP/BBRNKQRN w KQkq - 0 1",
    "bbrknqnr/pppppppp/8/8/8/8/PPPPPPPP/BBRKNQNR w KQkq - 0 1",
    "bbrknqrn/pppppppp/8/8/8/8/PPPPPPPP/BBRKNQRN w KQkq - 0 1",
    "bbrkrqnn/pppppppp/8/8/8/8/PPPPPPPP/BBRKRQNN w KQkq - 0 1",
    "bbnnrkqr/pppppppp/8/8/8/8/PPPPPPPP/BBNNRKQR w KQkq - 0 1",
    "bbnrnkqr/pppppppp/8/8/8/8/PPPPPPPP/BBNRNKQR w KQkq - 0 1",
    "bbnrknqr/pppppppp/8/8/8/8/PPPPPPPP/BBNRKNQR w KQkq - 0 1",
    "bbnrkrqn/pppppppp/8/8/8/8/PPPPPPPP/BBNRKRQN w KQkq - 0 1",
    "bbrnnkqr/pppppppp/8/8/8/8/PPPPPPPP/BBRNNKQR w KQkq - 0 1",
    "bbrnknqr/pppppppp/8/8/8/8/PPPPPPPP/BBRNKNQR w KQkq - 0 1",
    "bbrnkrqn/pppppppp/8/8/8/8/PPPPPPPP/BBRNKRQN w KQkq - 0 1",
    "bbrknnqr/pppppppp/8/8/8/8/PPPPPPPP/BBRKNNQR w KQkq - 0 1",
    "bbrknrqn/pppppppp/8/8/8/8/PPPPPPPP/BBRKNRQN w KQkq - 0 1",
    "bbrkrnqn/pppppppp/8/8/8/8/PPPPPPPP/BBRKRNQN w KQkq - 0 1",
    "bbnnrkrq/pppppppp/8/8/8/8/PPPPPPPP/BBNNRKRQ w KQkq - 0 1",
    "bbnrnkrq/pppppppp/8/8/8/8/PPPPPPPP/BBNRNKRQ w KQkq - 0 1",
    "bbnrknrq/pppppppp/8/8/8/8/PPPPPPPP/BBNRKNRQ w KQkq - 0 1",
    "bbnrkrnq/pppppppp/8/8/8/8/PPPPPPPP/BBNRKRNQ w KQkq - 0 1",
    "bbrnnkrq/pppppppp/8/8/8/8/PPPPPPPP/BBRNNKRQ w KQkq - 0 1",
    "bbrnknrq/pppppppp/8/8/8/8/PPPPPPPP/BBRNKNRQ w KQkq - 0 1",
    "bbrnkrnq/pppppppp/8/8/8/8/PPPPPPPP/BBRNKRNQ w KQkq - 0 1",
    "bbrknnrq/pppppppp/8/8/8/8/PPPPPPPP/BBRKNNRQ w KQkq - 0 1",
    "bbrknrnq/pppppppp/8/8/8/8/PPPPPPPP/BBRKNRNQ w KQkq - 0 1",
    "bbrkrnnq/pppppppp/8/8/8/8/PPPPPPPP/BBRKRNNQ w KQkq - 0 1",
    "bqnbnrkr/pppppppp/8/8/8/8/PPPPPPPP/BQNBNRKR w KQkq - 0 1",
    "bqnbrnkr/pppppppp/8/8/8/8/PPPPPPPP/BQNBRNKR w KQkq - 0 1",
    "bqnbrknr/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKNR w KQkq - 0 1",
    "bqnbrkrn/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKRN w KQkq - 0 1",
    "bqrbnnkr/pppppppp/8/8/8/8/PPPPPPPP/BQRBNNKR w KQkq - 0 1",
    "bqrbnknr/pppppppp/8/8/8/8/PPPPPPPP/BQRBNKNR w KQkq - 0 1",
    "bqrbnkrn/pppppppp/8/8/8/8/PPPPPPPP/BQRBNKRN w KQkq - 0 1",
    "bqrbknnr/pppppppp/8/8/8/8/PPPPPPPP/BQRBKNNR w KQkq - 0 1",
    "bqrbknrn/pppppppp/8/8/8/8/PPPPPPPP/BQRBKNRN w KQkq - 0 1",
    "bqrbkrnn/pppppppp/8/8/8/8/PPPPPPPP/BQRBKRNN w KQkq - 0 1",
    "bnqbnrkr/pppppppp/8/8/8/8/PPPPPPPP/BNQBNRKR w KQkq - 0 1",
    "bnqbrnkr/pppppppp/8/8/8/8/PPPPPPPP/BNQBRNKR w KQkq - 0 1",
    "bnqbrknr/pppppppp/8/8/8/8/PPPPPPPP/BNQBRKNR w KQkq - 0 1",
    "bnqbrkrn/pppppppp/8/8/8/8/PPPPPPPP/BNQBRKRN w KQkq - 0 1",
    "brqbnnkr/pppppppp/8/8/8/8/PPPPPPPP/BRQBNNKR w KQkq - 0 1",
    "brqbnknr/pppppppp/8/8/8/8/PPPPPPPP/BRQBNKNR w KQkq - 0 1",
    "brqbnkrn/pppppppp/8/8/8/8/PPPPPPPP/BRQBNKRN w KQkq - 0 1",
    "brqbknnr/pppppppp/8/8/8/8/PPPPPPPP/BRQBKNNR w KQkq - 0 1",
    "brqbknrn/pppppppp/8/8/8/8/PPPPPPPP/BRQBKNRN w KQkq - 0 1",
    "brqbkrnn/pppppppp/8/8/8/8/PPPPPPPP/BRQBKRNN w KQkq - 0 1",
    "bnnbqrkr/pppppppp/8/8/8/8/PPPPPPPP/BNNBQRKR w KQkq - 0 1",
    "bnrbqnkr/pppppppp/8/8/8/8/PPPPPPPP/BNRBQNKR w KQkq - 0 1",
    "bnrbqknr/pppppppp/8/8/8/8/PPPPPPPP/BNRBQKNR w KQkq - 0 1",
    "bnrbqkrn/pppppppp/8/8/8/8/PPPPPPPP/BNRBQKRN w KQkq - 0 1",
    "brnbqnkr/pppppppp/8/8/8/8/PPPPPPPP/BRNBQNKR w KQkq - 0 1",
    "brnbqknr/pppppppp/8/8/8/8/PPPPPPPP/BRNBQKNR w KQkq - 0 1",
    "brnbqkrn/pppppppp/8/8/8/8/PPPPPPPP/BRNBQKRN w KQkq - 0 1",
    "brkbqnnr/pppppppp/8/8/8/8/PPPPPPPP/BRKBQNNR w KQkq - 0 1",
    "brkbqnrn/pppppppp/8/8/8/8/PPPPPPPP/BRKBQNRN w KQkq - 0 1",
    "brkbqrnn/pppppppp/8/8/8/8/PPPPPPPP/BRKBQRNN w KQkq - 0 1",
    "bnnbrqkr/pppppppp/8/8/8/8/PPPPPPPP/BNNBRQKR w KQkq - 0 1",
    "bnrbnqkr/pppppppp/8/8/8/8/PPPPPPPP/BNRBNQKR w KQkq - 0 1",
    "bnrbkqnr/pppppppp/8/8/8/8/PPPPPPPP/BNRBKQNR w KQkq - 0 1",
    "bnrbkqrn/pppppppp/8/8/8/8/PPPPPPPP/BNRBKQRN w KQkq - 0 1",
    "brnbnqkr/pppppppp/8/8/8/8/PPPPPPPP/BRNBNQKR w KQkq - 0 1",
    "brnbkqnr/pppppppp/8/8/8/8/PPPPPPPP/BRNBKQNR w KQkq - 0 1",
    "brnbkqrn/pppppppp/8/8/8/8/PPPPPPPP/BRNBKQRN w KQkq - 0 1",
    "brkbnqnr/pppppppp/8/8/8/8/PPPPPPPP/BRKBNQNR w KQkq - 0 1",
    "brkbnqrn/pppppppp/8/8/8/8/PPPPPPPP/BRKBNQRN w KQkq - 0 1",
    "brkbrqnn/pppppppp/8/8/8/8/PPPPPPPP/BRKBRQNN w KQkq - 0 1",
    "bnnbrkqr/pppppppp/8/8/8/8/PPPPPPPP/BNNBRKQR w KQkq - 0 1",
    "bnrbnkqr/pppppppp/8/8/8/8/PPPPPPPP/BNRBNKQR w KQkq - 0 1",
    "bnrbknqr/pppppppp/8/8/8/8/PPPPPPPP/BNRBKNQR w KQkq - 0 1",
    "bnrbkrqn/pppppppp/8/8/8/8/PPPPPPPP/BNRBKRQN w KQkq - 0 1",
    "brnbnkqr/pppppppp/8/8/8/8/PPPPPPPP/BRNBNKQR w KQkq - 0 1",
    "brnbknqr/pppppppp/8/8/8/8/PPPPPPPP/BRNBKNQR w KQkq - 0 1",
    "brnbkrqn/pppppppp/8/8/8/8/PPPPPPPP/BRNBKRQN w KQkq - 0 1",
    "brkbnnqr/pppppppp/8/8/8/8/PPPPPPPP/BRKBNNQR w KQkq - 0 1",
    "brkbnrqn/pppppppp/8/8/8/8/PPPPPPPP/BRKBNRQN w KQkq - 0 1",
    "brkbrnqn/pppppppp/8/8/8/8/PPPPPPPP/BRKBRNQN w KQkq - 0 1",
    "bnnbrkrq/pppppppp/8/8/8/8/PPPPPPPP/BNNBRKRQ w KQkq - 0 1",
    "bnrbnkrq/pppppppp/8/8/8/8/PPPPPPPP/BNRBNKRQ w KQkq - 0 1",
    "bnrbknrq/pppppppp/8/8/8/8/PPPPPPPP/BNRBKNRQ w KQkq - 0 1",
    "bnrbkrnq/pppppppp/8/8/8/8/PPPPPPPP/BNRBKRNQ w KQkq - 0 1",
    "brnbnkrq/pppppppp/8/8/8/8/PPPPPPPP/BRNBNKRQ w KQkq - 0 1",
    "brnbknrq/pppppppp/8/8/8/8/PPPPPPPP/BRNBKNRQ w KQkq - 0 1",
    "brnbkrnq/pppppppp/8/8/8/8/PPPPPPPP/BRNBKRNQ w KQkq - 0 1",
    "brkbnnrq/pppppppp/8/8/8/8/PPPPPPPP/BRKBNNRQ w KQkq - 0 1",
    "brkbnrnq/pppppppp/8/8/8/8/PPPPPPPP/BRKBNRNQ w KQkq - 0 1",
    "brkbnrnq/pppppppp/8/8/8/8/PPPPPPPP/BRKBNRNQ w KQkq - 0 1",
    "bqnnrbkr/pppppppp/8/8/8/8/PPPPPPPP/BQNNRBKR w KQkq - 0 1",
    "bqnrnbkr/pppppppp/8/8/8/8/PPPPPPPP/BQNRNBKR w KQkq - 0 1",
    "bqnrkbnr/pppppppp/8/8/8/8/PPPPPPPP/BQNRKBNR w KQkq - 0 1",
    "bqnrkbrn/pppppppp/8/8/8/8/PPPPPPPP/BQNRKBRN w KQkq - 0 1",
    "bqrnnbkr/pppppppp/8/8/8/8/PPPPPPPP/BQRNNBKR w KQkq - 0 1",
    "bqrnkbnr/pppppppp/8/8/8/8/PPPPPPPP/BQRNKBNR w KQkq - 0 1",
    "bqrnkbrn/pppppppp/8/8/8/8/PPPPPPPP/BQRNKBRN w KQkq - 0 1",
    "bqrknbnr/pppppppp/8/8/8/8/PPPPPPPP/BQRKNBNR w KQkq - 0 1",
    "bqrknbrn/pppppppp/8/8/8/8/PPPPPPPP/BQRKNBRN w KQkq - 0 1",
    "bqrkrbnn/pppppppp/8/8/8/8/PPPPPPPP/BQRKRBNN w KQkq - 0 1",
    "bnqnrbkr/pppppppp/8/8/8/8/PPPPPPPP/BNQNRBKR w KQkq - 0 1",
    "bnqrnbkr/pppppppp/8/8/8/8/PPPPPPPP/BNQRNBKR w KQkq - 0 1",
    "bnqrkbnr/pppppppp/8/8/8/8/PPPPPPPP/BNQRKBNR w KQkq - 0 1",
    "bnqrkbrn/pppppppp/8/8/8/8/PPPPPPPP/BNQRKBRN w KQkq - 0 1",
    "brqnnbkr/pppppppp/8/8/8/8/PPPPPPPP/BRQNNBKR w KQkq - 0 1",
    "brqnkbnr/pppppppp/8/8/8/8/PPPPPPPP/BRQNKBNR w KQkq - 0 1",
    "brqnkbrn/pppppppp/8/8/8/8/PPPPPPPP/BRQNKBRN w KQkq - 0 1",
    "brqknbnr/pppppppp/8/8/8/8/PPPPPPPP/BRQKNBNR w KQkq - 0 1",
    "brqknbrn/pppppppp/8/8/8/8/PPPPPPPP/BRQKNBRN w KQkq - 0 1",
    "brqkrbnn/pppppppp/8/8/8/8/PPPPPPPP/BRQKRBNN w KQkq - 0 1",
    "bnnqrbkr/pppppppp/8/8/8/8/PPPPPPPP/BNNQRBKR w KQkq - 0 1",
    "bnrqnbkr/pppppppp/8/8/8/8/PPPPPPPP/BNRQNBKR w KQkq - 0 1",
    "bnrqkbnr/pppppppp/8/8/8/8/PPPPPPPP/BNRQKBNR w KQkq - 0 1",
    "bnrqkbrn/pppppppp/8/8/8/8/PPPPPPPP/BNRQKBRN w KQkq - 0 1",
    "brnqnbkr/pppppppp/8/8/8/8/PPPPPPPP/BRNQNBKR w KQkq - 0 1",
    "brnqkbnr/pppppppp/8/8/8/8/PPPPPPPP/BRNQKBNR w KQkq - 0 1",
    "brnqkbrn/pppppppp/8/8/8/8/PPPPPPPP/BRNQKBRN w KQkq - 0 1",
    "brkqnbnr/pppppppp/8/8/8/8/PPPPPPPP/BRKQNBNR w KQkq - 0 1",
    "brkqnbrn/pppppppp/8/8/8/8/PPPPPPPP/BRKQNBRN w KQkq - 0 1",
    "brkqrbnn/pppppppp/8/8/8/8/PPPPPPPP/BRKQRBNN w KQkq - 0 1",
    "bnnrqbkr/pppppppp/8/8/8/8/PPPPPPPP/BNNRQBKR w KQkq - 0 1",
    "bnrnqbkr/pppppppp/8/8/8/8/PPPPPPPP/BNRNQBKR w KQkq - 0 1",
    "bnrkqbnr/pppppppp/8/8/8/8/PPPPPPPP/BNRKQBNR w KQkq - 0 1",
    "bnrkqbrn/pppppppp/8/8/8/8/PPPPPPPP/BNRKQBRN w KQkq - 0 1",
    "brnnqbkr/pppppppp/8/8/8/8/PPPPPPPP/BRNNQBKR w KQkq - 0 1",
    "brnkqbnr/pppppppp/8/8/8/8/PPPPPPPP/BRNKQBNR w KQkq - 0 1",
    "brnkqbrn/pppppppp/8/8/8/8/PPPPPPPP/BRNKQBRN w KQkq - 0 1",
    "brknqbnr/pppppppp/8/8/8/8/PPPPPPPP/BRKNQBNR w KQkq - 0 1",
    "brknqbrn/pppppppp/8/8/8/8/PPPPPPPP/BRKNQBRN w KQkq - 0 1",
    "brkrqbnn/pppppppp/8/8/8/8/PPPPPPPP/BRKRQBNN w KQkq - 0 1",
    "bnnrkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBQR w KQkq - 0 1",
    "bnrnkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNRNKBQR w KQkq - 0 1",
    "bnrknbqr/pppppppp/8/8/8/8/PPPPPPPP/BNRKNBQR w KQkq - 0 1",
    "bnrkrbqn/pppppppp/8/8/8/8/PPPPPPPP/BNRKRBQN w KQkq - 0 1",
    "brnnkbqr/pppppppp/8/8/8/8/PPPPPPPP/BRNNKBQR w KQkq - 0 1",
    "brnknbqr/pppppppp/8/8/8/8/PPPPPPPP/BRNKNBQR w KQkq - 0 1",
    "brnkrbqn/pppppppp/8/8/8/8/PPPPPPPP/BRNKRBQN w KQkq - 0 1",
    "brknnbqr/pppppppp/8/8/8/8/PPPPPPPP/BRKNNBQR w KQkq - 0 1",
    "brknrbqn/pppppppp/8/8/8/8/PPPPPPPP/BRKNRBQN w KQkq - 0 1",
    "brkrnbqn/pppppppp/8/8/8/8/PPPPPPPP/BRKRNBQN w KQkq - 0 1",
    "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBRQ w KQkq - 0 1",
    "bnrnkbrq/pppppppp/8/8/8/8/PPPPPPPP/BNRNKBRQ w KQkq - 0 1",
    "bnrknbrq/pppppppp/8/8/8/8/PPPPPPPP/BNRKNBRQ w KQkq - 0 1",
    "bnrkrbnq/pppppppp/8/8/8/8/PPPPPPPP/BNRKRBNQ w KQkq - 0 1",
    "brnnkbrq/pppppppp/8/8/8/8/PPPPPPPP/BRNNKBRQ w KQkq - 0 1",
    "brnknbrq/pppppppp/8/8/8/8/PPPPPPPP/BRNKNBRQ w KQkq - 0 1",
    "brnkrbnq/pppppppp/8/8/8/8/PPPPPPPP/BRNKRBNQ w KQkq - 0 1",
    "brknnbrq/pppppppp/8/8/8/8/PPPPPPPP/BRKNNBRQ w KQkq - 0 1",
    "brknrbnq/pppppppp/8/8/8/8/PPPPPPPP/BRKNRBNQ w KQkq - 0 1",
    "brkrnbnq/pppppppp/8/8/8/8/PPPPPPPP/BRKRNBNQ w KQkq - 0 1",
    "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1",
    "bqnrnkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNRNKRB w KQkq - 0 1",
    "bqnrknrb/pppppppp/8/8/8/8/PPPPPPPP/BQNRKNRB w KQkq - 0 1",
    "bqnrkrnb/pppppppp/8/8/8/8/PPPPPPPP/BQNRKRNB w KQkq - 0 1",
    "bqrnnkrb/pppppppp/8/8/8/8/PPPPPPPP/BQRNNKRB w KQkq - 0 1",
    "bqrnknrb/pppppppp/8/8/8/8/PPPPPPPP/BQRNKNRB w KQkq - 0 1",
    "bqrnkrnb/pppppppp/8/8/8/8/PPPPPPPP/BQRNKRNB w KQkq - 0 1",
    "bqrknnrb/pppppppp/8/8/8/8/PPPPPPPP/BQRKNNRB w KQkq - 0 1",
    "bqrknrnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKNRNB w KQkq - 0 1",
    "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1",
    "bnqnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BNQNRKRB w KQkq - 0 1",
    "bnqrnkrb/pppppppp/8/8/8/8/PPPPPPPP/BNQRNKRB w KQkq - 0 1",
    "bnqrknrb/pppppppp/8/8/8/8/PPPPPPPP/BNQRKNRB w KQkq - 0 1",
    "bnqrkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNQRKRNB w KQkq - 0 1",
    "brqnnkrb/pppppppp/8/8/8/8/PPPPPPPP/BRQNNKRB w KQkq - 0 1",
    "brqnknrb/pppppppp/8/8/8/8/PPPPPPPP/BRQNKNRB w KQkq - 0 1",
    "brqnkrnb/pppppppp/8/8/8/8/PPPPPPPP/BRQNKRNB w KQkq - 0 1",
    "brqknnrb/pppppppp/8/8/8/8/PPPPPPPP/BRQKNNRB w KQkq - 0 1",
    "brqknrnb/pppppppp/8/8/8/8/PPPPPPPP/BRQKNRNB w KQkq - 0 1",
    "brqkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BRQKRNNB w KQkq - 0 1",
    "bnnqrkrb/pppppppp/8/8/8/8/PPPPPPPP/BNNQRKRB w KQkq - 0 1",
    "bnrqnkrb/pppppppp/8/8/8/8/PPPPPPPP/BNRQNKRB w KQkq - 0 1",
    "bnrqknrb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKNRB w KQkq - 0 1",
    "bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1",
    "brnqnkrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQNKRB w KQkq - 0 1",
    "brnqknrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKNRB w KQkq - 0 1",
    "brnqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKRNB w KQkq - 0 1",
    "brkqnnrb/pppppppp/8/8/8/8/PPPPPPPP/BRKQNNRB w KQkq - 0 1",
    "brkqnrnb/pppppppp/8/8/8/8/PPPPPPPP/BRKQNRNB w KQkq - 0 1",
    "brkqrnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKQRNNB w KQkq - 0 1",
    "bnnrqkrb/pppppppp/8/8/8/8/PPPPPPPP/BNNRQKRB w KQkq - 0 1",
    "bnrnqkrb/pppppppp/8/8/8/8/PPPPPPPP/BNRNQKRB w KQkq - 0 1",
    "bnrkqnrb/pppppppp/8/8/8/8/PPPPPPPP/BNRKQNRB w KQkq - 0 1",
    "bnrkqrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRKQRNB w KQkq - 0 1",
    "brnnqkrb/pppppppp/8/8/8/8/PPPPPPPP/BRNNQKRB w KQkq - 0 1",
    "brnkqnrb/pppppppp/8/8/8/8/PPPPPPPP/BRNKQNRB w KQkq - 0 1",
    "brnkqrnb/pppppppp/8/8/8/8/PPPPPPPP/BRNKQRNB w KQkq - 0 1",
    "brknqnrb/pppppppp/8/8/8/8/PPPPPPPP/BRKNQNRB w KQkq - 0 1",
    "brknqrnb/pppppppp/8/8/8/8/PPPPPPPP/BRKNQRNB w KQkq - 0 1",
    "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1",
    "bnnrkqrb/pppppppp/8/8/8/8/PPPPPPPP/BNNRKQRB w KQkq - 0 1",
    "bnrnkqrb/pppppppp/8/8/8/8/PPPPPPPP/BNRNKQRB w KQkq - 0 1",
    "bnrknqrb/pppppppp/8/8/8/8/PPPPPPPP/BNRKNQRB w KQkq - 0 1",
    "bnrkrqnb/pppppppp/8/8/8/8/PPPPPPPP/BNRKRQNB w KQkq - 0 1",
    "brnnkqrb/pppppppp/8/8/8/8/PPPPPPPP/BRNNKQRB w KQkq - 0 1",
    "brnknqrb/pppppppp/8/8/8/8/PPPPPPPP/BRNKNQRB w KQkq - 0 1",
    "brnkrqnb/pppppppp/8/8/8/8/PPPPPPPP/BRNKRQNB w KQkq - 0 1",
    "brknnqrb/pppppppp/8/8/8/8/PPPPPPPP/BRKNNQRB w KQkq - 0 1",
    "brknrqnb/pppppppp/8/8/8/8/PPPPPPPP/BRKNRQNB w KQkq - 0 1",
    "brkrnqnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRNQNB w KQkq - 0 1",
    "bnnrkrqb/pppppppp/8/8/8/8/PPPPPPPP/BNNRKRQB w KQkq - 0 1",
    "bnrnkrqb/pppppppp/8/8/8/8/PPPPPPPP/BNRNKRQB w KQkq - 0 1",
    "bnrknrqb/pppppppp/8/8/8/8/PPPPPPPP/BNRKNRQB w KQkq - 0 1",
    "bnrkrnqb/pppppppp/8/8/8/8/PPPPPPPP/BNRKRNQB w KQkq - 0 1",
    "brnnkrqb/pppppppp/8/8/8/8/PPPPPPPP/BRNNKRQB w KQkq - 0 1",
    "brnknrqb/pppppppp/8/8/8/8/PPPPPPPP/BRNKNRQB w KQkq - 0 1",
    "brnkrnqb/pppppppp/8/8/8/8/PPPPPPPP/BRNKRNQB w KQkq - 0 1",
    "brknnrqb/pppppppp/8/8/8/8/PPPPPPPP/BRKNNRQB w KQkq - 0 1",
    "brknrnqb/pppppppp/8/8/8/8/PPPPPPPP/BRKNRNQB w KQkq - 0 1",
    "brkrnnqb/pppppppp/8/8/8/8/PPPPPPPP/BRKRNNQB w KQkq - 0 1",
    "qbbnnrkr/pppppppp/8/8/8/8/PPPPPPPP/QBBNNRKR w KQkq - 0 1",
    "qbbnrnkr/pppppppp/8/8/8/8/PPPPPPPP/QBBNRNKR w KQkq - 0 1",
    "qbbnrknr/pppppppp/8/8/8/8/PPPPPPPP/QBBNRKNR w KQkq - 0 1",
    "qbbnrkrn/pppppppp/8/8/8/8/PPPPPPPP/QBBNRKRN w KQkq - 0 1",
    "qbbrnnkr/pppppppp/8/8/8/8/PPPPPPPP/QBBRNNKR w KQkq - 0 1",
    "qbbrnknr/pppppppp/8/8/8/8/PPPPPPPP/QBBRNKNR w KQkq - 0 1",
    "qbbrnkrn/pppppppp/8/8/8/8/PPPPPPPP/QBBRNKRN w KQkq - 0 1",
    "qbbrknnr/pppppppp/8/8/8/8/PPPPPPPP/QBBRKNNR w KQkq - 0 1",
    "qbbrknrn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKNRN w KQkq - 0 1",
    "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1",
    "nbbqnrkr/pppppppp/8/8/8/8/PPPPPPPP/NBBQNRKR w KQkq - 0 1",
    "nbbqrnkr/pppppppp/8/8/8/8/PPPPPPPP/NBBQRNKR w KQkq - 0 1",
    "nbbqrknr/pppppppp/8/8/8/8/PPPPPPPP/NBBQRKNR w KQkq - 0 1",
    "nbbqrkrn/pppppppp/8/8/8/8/PPPPPPPP/NBBQRKRN w KQkq - 0 1",
    "rbbqnnkr/pppppppp/8/8/8/8/PPPPPPPP/RBBQNNKR w KQkq - 0 1",
    "rbbqnknr/pppppppp/8/8/8/8/PPPPPPPP/RBBQNKNR w KQkq - 0 1",
    "rbbqnkrn/pppppppp/8/8/8/8/PPPPPPPP/RBBQNKRN w KQkq - 0 1",
    "rbbqknnr/pppppppp/8/8/8/8/PPPPPPPP/RBBQKNNR w KQkq - 0 1",
    "rbbqknrn/pppppppp/8/8/8/8/PPPPPPPP/RBBQKNRN w KQkq - 0 1",
    "rbbqkrnn/pppppppp/8/8/8/8/PPPPPPPP/RBBQKRNN w KQkq - 0 1",
    "nbbnqrkr/pppppppp/8/8/8/8/PPPPPPPP/NBBNQRKR w KQkq - 0 1",
    "nbbrqnkr/pppppppp/8/8/8/8/PPPPPPPP/NBBRQNKR w KQkq - 0 1",
    "nbbrqknr/pppppppp/8/8/8/8/PPPPPPPP/NBBRQKNR w KQkq - 0 1",
    "nbbrqkrn/pppppppp/8/8/8/8/PPPPPPPP/NBBRQKRN w KQkq - 0 1",
    "rbbnqnkr/pppppppp/8/8/8/8/PPPPPPPP/RBBNQNKR w KQkq - 0 1",
    "rbbnqknr/pppppppp/8/8/8/8/PPPPPPPP/RBBNQKNR w KQkq - 0 1",
    "rbbnqkrn/pppppppp/8/8/8/8/PPPPPPPP/RBBNQKRN w KQkq - 0 1",
    "rbbkqnnr/pppppppp/8/8/8/8/PPPPPPPP/RBBKQNNR w KQkq - 0 1",
    "rbbkqnrn/pppppppp/8/8/8/8/PPPPPPPP/RBBKQNRN w KQkq - 0 1",
    "rbbkqrnn/pppppppp/8/8/8/8/PPPPPPPP/RBBKQRNN w KQkq - 0 1",
    "nbbnrqkr/pppppppp/8/8/8/8/PPPPPPPP/NBBNRQKR w KQkq - 0 1",
    "nbbrnqkr/pppppppp/8/8/8/8/PPPPPPPP/NBBRNQKR w KQkq - 0 1",
    "nbbrkqnr/pppppppp/8/8/8/8/PPPPPPPP/NBBRKQNR w KQkq - 0 1",
    "nbbrkqrn/pppppppp/8/8/8/8/PPPPPPPP/NBBRKQRN w KQkq - 0 1",
    "rbbnnqkr/pppppppp/8/8/8/8/PPPPPPPP/RBBNNQKR w KQkq - 0 1",
    "rbbnkqnr/pppppppp/8/8/8/8/PPPPPPPP/RBBNKQNR w KQkq - 0 1",
    "rbbnkqrn/pppppppp/8/8/8/8/PPPPPPPP/RBBNKQRN w KQkq - 0 1",
    "rbbknqnr/pppppppp/8/8/8/8/PPPPPPPP/RBBKNQNR w KQkq - 0 1",
    "rbbknqrn/pppppppp/8/8/8/8/PPPPPPPP/RBBKNQRN w KQkq - 0 1",
    "rbbkrqnn/pppppppp/8/8/8/8/PPPPPPPP/RBBKRQNN w KQkq - 0 1",
    "nbbnrkqr/pppppppp/8/8/8/8/PPPPPPPP/NBBNRKQR w KQkq - 0 1",
    "nbbrnkqr/pppppppp/8/8/8/8/PPPPPPPP/NBBRNKQR w KQkq - 0 1",
    "nbbrknqr/pppppppp/8/8/8/8/PPPPPPPP/NBBRKNQR w KQkq - 0 1",
    "nbbrkrqn/pppppppp/8/8/8/8/PPPPPPPP/NBBRKRQN w KQkq - 0 1",
    "rbbnnkqr/pppppppp/8/8/8/8/PPPPPPPP/RBBNNKQR w KQkq - 0 1",
    "rbbnknqr/pppppppp/8/8/8/8/PPPPPPPP/RBBNKNQR w KQkq - 0 1",
    "rbbnkrqn/pppppppp/8/8/8/8/PPPPPPPP/RBBNKRQN w KQkq - 0 1",
    "rbbknnqr/pppppppp/8/8/8/8/PPPPPPPP/RBBKNNQR w KQkq - 0 1",
    "rbbknrqn/pppppppp/8/8/8/8/PPPPPPPP/RBBKNRQN w KQkq - 0 1",
    "rbbkrnqn/pppppppp/8/8/8/8/PPPPPPPP/RBBKRNQN w KQkq - 0 1",
    "nbbnrkrq/pppppppp/8/8/8/8/PPPPPPPP/NBBNRKRQ w KQkq - 0 1",
    "nbbrnkrq/pppppppp/8/8/8/8/PPPPPPPP/NBBRNKRQ w KQkq - 0 1",
    "nbbrknrq/pppppppp/8/8/8/8/PPPPPPPP/NBBRKNRQ w KQkq - 0 1",
    "nbbrkrnq/pppppppp/8/8/8/8/PPPPPPPP/NBBRKRNQ w KQkq - 0 1",
    "rbbnnkrq/pppppppp/8/8/8/8/PPPPPPPP/RBBNNKRQ w KQkq - 0 1",
    "rbbnknrq/pppppppp/8/8/8/8/PPPPPPPP/RBBNKNRQ w KQkq - 0 1",
    "rbbnkrnq/pppppppp/8/8/8/8/PPPPPPPP/RBBNKRNQ w KQkq - 0 1",
    "rbbknnrq/pppppppp/8/8/8/8/PPPPPPPP/RBBKNNRQ w KQkq - 0 1",
    "rbbknrnq/pppppppp/8/8/8/8/PPPPPPPP/RBBKNRNQ w KQkq - 0 1",
    "rbbkrnnq/pppppppp/8/8/8/8/PPPPPPPP/RBBKRNNQ w KQkq - 0 1",
    "qnbbnrkr/pppppppp/8/8/8/8/PPPPPPPP/QNBBNRKR w KQkq - 0 1",
    "qnbbrnkr/pppppppp/8/8/8/8/PPPPPPPP/QNBBRNKR w KQkq - 0 1",
    "qnbbrknr/pppppppp/8/8/8/8/PPPPPPPP/QNBBRKNR w KQkq - 0 1",
    "qnbbrkrn/pppppppp/8/8/8/8/PPPPPPPP/QNBBRKRN w KQkq - 0 1",
    "qrbbnnkr/pppppppp/8/8/8/8/PPPPPPPP/QRBBNNKR w KQkq - 0 1",
    "qrbbnknr/pppppppp/8/8/8/8/PPPPPPPP/QRBBNKNR w KQkq - 0 1",
    "qrbbnkrn/pppppppp/8/8/8/8/PPPPPPPP/QRBBNKRN w KQkq - 0 1",
    "qrbbknnr/pppppppp/8/8/8/8/PPPPPPPP/QRBBKNNR w KQkq - 0 1",
    "qrbbknrn/pppppppp/8/8/8/8/PPPPPPPP/QRBBKNRN w KQkq - 0 1",
    "qrbbkrnn/pppppppp/8/8/8/8/PPPPPPPP/QRBBKRNN w KQkq - 0 1",
    "nqbbnrkr/pppppppp/8/8/8/8/PPPPPPPP/NQBBNRKR w KQkq - 0 1",
    "nqbbrnkr/pppppppp/8/8/8/8/PPPPPPPP/NQBBRNKR w KQkq - 0 1",
    "nqbbrknr/pppppppp/8/8/8/8/PPPPPPPP/NQBBRKNR w KQkq - 0 1",
    "nqbbrkrn/pppppppp/8/8/8/8/PPPPPPPP/NQBBRKRN w KQkq - 0 1",
    "rqbbnnkr/pppppppp/8/8/8/8/PPPPPPPP/RQBBNNKR w KQkq - 0 1",
    "rqbbnknr/pppppppp/8/8/8/8/PPPPPPPP/RQBBNKNR w KQkq - 0 1",
    "rqbbnkrn/pppppppp/8/8/8/8/PPPPPPPP/RQBBNKRN w KQkq - 0 1",
    "rqbbknnr/pppppppp/8/8/8/8/PPPPPPPP/RQBBKNNR w KQkq - 0 1",
    "rqbbknrn/pppppppp/8/8/8/8/PPPPPPPP/RQBBKNRN w KQkq - 0 1",
    "rqbbkrnn/pppppppp/8/8/8/8/PPPPPPPP/RQBBKRNN w KQkq - 0 1",
    "nnbbqrkr/pppppppp/8/8/8/8/PPPPPPPP/NNBBQRKR w KQkq - 0 1",
    "nrbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/NRBBQNKR w KQkq - 0 1",
    "nrbbqknr/pppppppp/8/8/8/8/PPPPPPPP/NRBBQKNR w KQkq - 0 1",
    "nrbbqkrn/pppppppp/8/8/8/8/PPPPPPPP/NRBBQKRN w KQkq - 0 1",
    "rnbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/RNBBQNKR w KQkq - 0 1",
    "rnbbqknr/pppppppp/8/8/8/8/PPPPPPPP/RNBBQKNR w KQkq - 0 1",
    "rnbbqkrn/pppppppp/8/8/8/8/PPPPPPPP/RNBBQKRN w KQkq - 0 1",
    "rkbbqnnr/pppppppp/8/8/8/8/PPPPPPPP/RKBBQNNR w KQkq - 0 1",
    "rkbbqnrn/pppppppp/8/8/8/8/PPPPPPPP/RKBBQNRN w KQkq - 0 1",
    "rkbbqrnn/pppppppp/8/8/8/8/PPPPPPPP/RKBBQRNN w KQkq - 0 1",
    "nnbbrqkr/pppppppp/8/8/8/8/PPPPPPPP/NNBBRQKR w KQkq - 0 1",
    "nrbbnqkr/pppppppp/8/8/8/8/PPPPPPPP/NRBBNQKR w KQkq - 0 1",
    "nrbbkqnr/pppppppp/8/8/8/8/PPPPPPPP/NRBBKQNR w KQkq - 0 1",
    "nrbbkqrn/pppppppp/8/8/8/8/PPPPPPPP/NRBBKQRN w KQkq - 0 1",
    "rnbbnqkr/pppppppp/8/8/8/8/PPPPPPPP/RNBBNQKR w KQkq - 0 1",
    "rnbbkqnr/pppppppp/8/8/8/8/PPPPPPPP/RNBBKQNR w KQkq - 0 1",
    "rnbbkqrn/pppppppp/8/8/8/8/PPPPPPPP/RNBBKQRN w KQkq - 0 1",
    "rkbbnqnr/pppppppp/8/8/8/8/PPPPPPPP/RKBBNQNR w KQkq - 0 1",
    "rkbbnqrn/pppppppp/8/8/8/8/PPPPPPPP/RKBBNQRN w KQkq - 0 1",
    "rkbbrqnn/pppppppp/8/8/8/8/PPPPPPPP/RKBBRQNN w KQkq - 0 1",
    "nnbbrkqr/pppppppp/8/8/8/8/PPPPPPPP/NNBBRKQR w KQkq - 0 1",
    "nrbbnkqr/pppppppp/8/8/8/8/PPPPPPPP/NRBBNKQR w KQkq - 0 1",
    "nrbbknqr/pppppppp/8/8/8/8/PPPPPPPP/NRBBKNQR w KQkq - 0 1",
    "nrbbkrqn/pppppppp/8/8/8/8/PPPPPPPP/NRBBKRQN w KQkq - 0 1",
    "rnbbnkqr/pppppppp/8/8/8/8/PPPPPPPP/RNBBNKQR w KQkq - 0 1",
    "rnbbknqr/pppppppp/8/8/8/8/PPPPPPPP/RNBBKNQR w KQkq - 0 1",
    "rnbbkrqn/pppppppp/8/8/8/8/PPPPPPPP/RNBBKRQN w KQkq - 0 1",
    "rkbbnnqr/pppppppp/8/8/8/8/PPPPPPPP/RKBBNNQR w KQkq - 0 1",
    "rkbbnrqn/pppppppp/8/8/8/8/PPPPPPPP/RKBBNRQN w KQkq - 0 1",
    "rkbbrnqn/pppppppp/8/8/8/8/PPPPPPPP/RKBBRNQN w KQkq - 0 1",
    "nnbbrkrq/pppppppp/8/8/8/8/PPPPPPPP/NNBBRKRQ w KQkq - 0 1",
    "nrbbnkrq/pppppppp/8/8/8/8/PPPPPPPP/NRBBNKRQ w KQkq - 0 1",
    "nrbbknrq/pppppppp/8/8/8/8/PPPPPPPP/NRBBKNRQ w KQkq - 0 1",
    "nrbbkrnq/pppppppp/8/8/8/8/PPPPPPPP/NRBBKRNQ w KQkq - 0 1",
    "rnbbnkrq/pppppppp/8/8/8/8/PPPPPPPP/RNBBNKRQ w KQkq - 0 1",
    "rnbbknrq/pppppppp/8/8/8/8/PPPPPPPP/RNBBKNRQ w KQkq - 0 1",
    "rnbbkrnq/pppppppp/8/8/8/8/PPPPPPPP/RNBBKRNQ w KQkq - 0 1",
    "rkbbnnrq/pppppppp/8/8/8/8/PPPPPPPP/RKBBNNRQ w KQkq - 0 1",
    "rkbbnrnq/pppppppp/8/8/8/8/PPPPPPPP/RKBBNRNQ w KQkq - 0 1",
    "rkbbrnnq/pppppppp/8/8/8/8/PPPPPPPP/RKBBRNNQ w KQkq - 0 1",
    "qnbnrbkr/pppppppp/8/8/8/8/PPPPPPPP/QNBNRBKR w KQkq - 0 1",
    "qnbrnbkr/pppppppp/8/8/8/8/PPPPPPPP/QNBRNBKR w KQkq - 0 1",
    "qnbrkbnr/pppppppp/8/8/8/8/PPPPPPPP/QNBRKBNR w KQkq - 0 1",
    "qnbrkbrn/pppppppp/8/8/8/8/PPPPPPPP/QNBRKBRN w KQkq - 0 1",
    "qrbnnbkr/pppppppp/8/8/8/8/PPPPPPPP/QRBNNBKR w KQkq - 0 1",
    "qrbnkbnr/pppppppp/8/8/8/8/PPPPPPPP/QRBNKBNR w KQkq - 0 1",
    "qrbnkbrn/pppppppp/8/8/8/8/PPPPPPPP/QRBNKBRN w KQkq - 0 1",
    "qrbknbnr/pppppppp/8/8/8/8/PPPPPPPP/QRBKNBNR w KQkq - 0 1",
    "qrbknbrn/pppppppp/8/8/8/8/PPPPPPPP/QRBKNBRN w KQkq - 0 1",
    "qrbkrbnn/pppppppp/8/8/8/8/PPPPPPPP/QRBKRBNN w KQkq - 0 1",
    "nqbnrbkr/pppppppp/8/8/8/8/PPPPPPPP/NQBNRBKR w KQkq - 0 1",
    "nqbrnbkr/pppppppp/8/8/8/8/PPPPPPPP/NQBRNBKR w KQkq - 0 1",
    "nqbrkbnr/pppppppp/8/8/8/8/PPPPPPPP/NQBRKBNR w KQkq - 0 1",
    "nqbrkbrn/pppppppp/8/8/8/8/PPPPPPPP/NQBRKBRN w KQkq - 0 1",
    "rqbnnbkr/pppppppp/8/8/8/8/PPPPPPPP/RQBNNBKR w KQkq - 0 1",
    "rqbnkbnr/pppppppp/8/8/8/8/PPPPPPPP/RQBNKBNR w KQkq - 0 1",
    "rqbnkbrn/pppppppp/8/8/8/8/PPPPPPPP/RQBNKBRN w KQkq - 0 1",
    "rqbknbnr/pppppppp/8/8/8/8/PPPPPPPP/RQBKNBNR w KQkq - 0 1",
    "rqbknbrn/pppppppp/8/8/8/8/PPPPPPPP/RQBKNBRN w KQkq - 0 1",
    "rqbkrbnn/pppppppp/8/8/8/8/PPPPPPPP/RQBKRBNN w KQkq - 0 1",
    "nnbqrbkr/pppppppp/8/8/8/8/PPPPPPPP/NNBQRBKR w KQkq - 0 1",
    "nrbqnbkr/pppppppp/8/8/8/8/PPPPPPPP/NRBQNBKR w KQkq - 0 1",
    "nrbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/NRBQKBNR w KQkq - 0 1",
    "nrbqkbrn/pppppppp/8/8/8/8/PPPPPPPP/NRBQKBRN w KQkq - 0 1",
    "rnbqnbkr/pppppppp/8/8/8/8/PPPPPPPP/RNBQNBKR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbrn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBRN w KQkq - 0 1",
    "rkbqnbnr/pppppppp/8/8/8/8/PPPPPPPP/RKBQNBNR w KQkq - 0 1",
    "rkbqnbrn/pppppppp/8/8/8/8/PPPPPPPP/RKBQNBRN w KQkq - 0 1",
    "rkbqrbnn/pppppppp/8/8/8/8/PPPPPPPP/RKBQRBNN w KQkq - 0 1",
    "nnbrqbkr/pppppppp/8/8/8/8/PPPPPPPP/NNBRQBKR w KQkq - 0 1",
    "nrbnqbkr/pppppppp/8/8/8/8/PPPPPPPP/NRBNQBKR w KQkq - 0 1",
    "nrbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/NRBKQBNR w KQkq - 0 1",
    "nrbkqbrn/pppppppp/8/8/8/8/PPPPPPPP/NRBKQBRN w KQkq - 0 1",
    "rnbnqbkr/pppppppp/8/8/8/8/PPPPPPPP/RNBNQBKR w KQkq - 0 1",
    "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1",
    "rnbkqbrn/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBRN w KQkq - 0 1",
    "rkbnqbnr/pppppppp/8/8/8/8/PPPPPPPP/RKBNQBNR w KQkq - 0 1",
    "rkbnqbrn/pppppppp/8/8/8/8/PPPPPPPP/RKBNQBRN w KQkq - 0 1",
    "rkbrqbnn/pppppppp/8/8/8/8/PPPPPPPP/RKBRQBNN w KQkq - 0 1",
    "nnbrkbqr/pppppppp/8/8/8/8/PPPPPPPP/NNBRKBQR w KQkq - 0 1",
    "nrbnkbqr/pppppppp/8/8/8/8/PPPPPPPP/NRBNKBQR w KQkq - 0 1",
    "nrbknbqr/pppppppp/8/8/8/8/PPPPPPPP/NRBKNBQR w KQkq - 0 1",
    "nrbkrbqn/pppppppp/8/8/8/8/PPPPPPPP/NRBKRBQN w KQkq - 0 1",
    "rnbnkbqr/pppppppp/8/8/8/8/PPPPPPPP/RNBNKBQR w KQkq - 0 1",
    "rnbknbqr/pppppppp/8/8/8/8/PPPPPPPP/RNBKNBQR w KQkq - 0 1",
    "rnbkrbqn/pppppppp/8/8/8/8/PPPPPPPP/RNBKRBQN w KQkq - 0 1",
    "rkbnnbqr/pppppppp/8/8/8/8/PPPPPPPP/RKBNNBQR w KQkq - 0 1",
    "rkbnrbqn/pppppppp/8/8/8/8/PPPPPPPP/RKBNRBQN w KQkq - 0 1",
    "rkbrnbqn/pppppppp/8/8/8/8/PPPPPPPP/RKBRNBQN w KQkq - 0 1",
    "nnbrkbrq/pppppppp/8/8/8/8/PPPPPPPP/NNBRKBRQ w KQkq - 0 1",
    "nrbnkbrq/pppppppp/8/8/8/8/PPPPPPPP/NRBNKBRQ w KQkq - 0 1",
    "nrbknbrq/pppppppp/8/8/8/8/PPPPPPPP/NRBKNBRQ w KQkq - 0 1",
    "nrbkrbnq/pppppppp/8/8/8/8/PPPPPPPP/NRBKRBNQ w KQkq - 0 1",
    "rnbnkbrq/pppppppp/8/8/8/8/PPPPPPPP/RNBNKBRQ w KQkq - 0 1",
    "rnbknbrq/pppppppp/8/8/8/8/PPPPPPPP/RNBKNBRQ w KQkq - 0 1",
    "rnbkrbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBKRBNQ w KQkq - 0 1",
    "rkbnnbrq/pppppppp/8/8/8/8/PPPPPPPP/RKBNNBRQ w KQkq - 0 1",
    "rkbnrbnq/pppppppp/8/8/8/8/PPPPPPPP/RKBNRBNQ w KQkq - 0 1",
    "rkbrnbnq/pppppppp/8/8/8/8/PPPPPPPP/RKBRNBNQ w KQkq - 0 1",
    "qnbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/QNBNRKRB w KQkq - 0 1",
    "qnbrnkrb/pppppppp/8/8/8/8/PPPPPPPP/QNBRNKRB w KQkq - 0 1",
    "qnbrknrb/pppppppp/8/8/8/8/PPPPPPPP/QNBRKNRB w KQkq - 0 1",
    "qnbrkrnb/pppppppp/8/8/8/8/PPPPPPPP/QNBRKRNB w KQkq - 0 1",
    "qrbnnkrb/pppppppp/8/8/8/8/PPPPPPPP/QRBNNKRB w KQkq - 0 1",
    "qrbnknrb/pppppppp/8/8/8/8/PPPPPPPP/QRBNKNRB w KQkq - 0 1",
    "qrbnkrnb/pppppppp/8/8/8/8/PPPPPPPP/QRBNKRNB w KQkq - 0 1",
    "qrbknnrb/pppppppp/8/8/8/8/PPPPPPPP/QRBKNNRB w KQkq - 0 1",
    "qrbknrnb/pppppppp/8/8/8/8/PPPPPPPP/QRBKNRNB w KQkq - 0 1",
    "qrbkrnnb/pppppppp/8/8/8/8/PPPPPPPP/QRBKRNNB w KQkq - 0 1",
    "nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w KQkq - 0 1",
    "nqbrnkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBRNKRB w KQkq - 0 1",
    "nqbrknrb/pppppppp/8/8/8/8/PPPPPPPP/NQBRKNRB w KQkq - 0 1",
    "nqbrkrnb/pppppppp/8/8/8/8/PPPPPPPP/NQBRKRNB w KQkq - 0 1",
    "rqbnnkrb/pppppppp/8/8/8/8/PPPPPPPP/RQBNNKRB w KQkq - 0 1",
    "rqbnknrb/pppppppp/8/8/8/8/PPPPPPPP/RQBNKNRB w KQkq - 0 1",
    "rqbnkrnb/pppppppp/8/8/8/8/PPPPPPPP/RQBNKRNB w KQkq - 0 1",
    "rqbknnrb/pppppppp/8/8/8/8/PPPPPPPP/RQBKNNRB w KQkq - 0 1",
    "rqbknrnb/pppppppp/8/8/8/8/PPPPPPPP/RQBKNRNB w KQkq - 0 1",
    "rqbkrnnb/pppppppp/8/8/8/8/PPPPPPPP/RQBKRNNB w KQkq - 0 1",
    "nnbqrkrb/pppppppp/8/8/8/8/PPPPPPPP/NNBQRKRB w KQkq - 0 1",
    "nrbqnkrb/pppppppp/8/8/8/8/PPPPPPPP/NRBQNKRB w KQkq - 0 1",
    "nrbqknrb/pppppppp/8/8/8/8/PPPPPPPP/NRBQKNRB w KQkq - 0 1",
    "nrbqkrnb/pppppppp/8/8/8/8/PPPPPPPP/NRBQKRNB w KQkq - 0 1",
    "rnbqnkrb/pppppppp/8/8/8/8/PPPPPPPP/RNBQNKRB w KQkq - 0 1",
    "rnbqknrb/pppppppp/8/8/8/8/PPPPPPPP/RNBQKNRB w KQkq - 0 1",
    "rnbqkrnb/pppppppp/8/8/8/8/PPPPPPPP/RNBQKRNB w KQkq - 0 1",
    "rkbqnnrb/pppppppp/8/8/8/8/PPPPPPPP/RKBQNNRB w KQkq - 0 1",
    "rkbqnrnb/pppppppp/8/8/8/8/PPPPPPPP/RKBQNRNB w KQkq - 0 1",
    "rkbqrnnb/pppppppp/8/8/8/8/PPPPPPPP/RKBQRNNB w KQkq - 0 1",
    "nnbrqkrb/pppppppp/8/8/8/8/PPPPPPPP/NNBRQKRB w KQkq - 0 1",
    "nrbnqkrb/pppppppp/8/8/8/8/PPPPPPPP/NRBNQKRB w KQkq - 0 1",
    "nrbkqnrb/pppppppp/8/8/8/8/PPPPPPPP/NRBKQNRB w KQkq - 0 1",
    "nrbkqrnb/pppppppp/8/8/8/8/PPPPPPPP/NRBKQRNB w KQkq - 0 1",
    "rnbnqkrb/pppppppp/8/8/8/8/PPPPPPPP/RNBNQKRB w KQkq - 0 1",
    "rnbkqnrb/pppppppp/8/8/8/8/PPPPPPPP/RNBKQNRB w KQkq - 0 1",
    "rnbkqrnb/pppppppp/8/8/8/8/PPPPPPPP/RNBKQRNB w KQkq - 0 1",
    "rkbnqnrb/pppppppp/8/8/8/8/PPPPPPPP/RKBNQNRB w KQkq - 0 1",
    "rkbnqrnb/pppppppp/8/8/8/8/PPPPPPPP/RKBNQRNB w KQkq - 0 1",
    "rkbrqnnb/pppppppp/8/8/8/8/PPPPPPPP/RKBRQNNB w KQkq - 0 1",
    "nnbrkqrb/pppppppp/8/8/8/8/PPPPPPPP/NNBRKQRB w KQkq - 0 1",
    "nrbnkqrb/pppppppp/8/8/8/8/PPPPPPPP/NRBNKQRB w KQkq - 0 1",
    "nrbknqrb/pppppppp/8/8/8/8/PPPPPPPP/NRBKNQRB w KQkq - 0 1",
    "nrbkrqnb/pppppppp/8/8/8/8/PPPPPPPP/NRBKRQNB w KQkq - 0 1",
    "rnbnkqrb/pppppppp/8/8/8/8/PPPPPPPP/RNBNKQRB w KQkq - 0 1",
    "rnbknqrb/pppppppp/8/8/8/8/PPPPPPPP/RNBKNQRB w KQkq - 0 1",
    "rnbkrqnb/pppppppp/8/8/8/8/PPPPPPPP/RNBKRQNB w KQkq - 0 1",
    "rkbnnqrb/pppppppp/8/8/8/8/PPPPPPPP/RKBNNQRB w KQkq - 0 1",
    "rkbnrqnb/pppppppp/8/8/8/8/PPPPPPPP/RKBNRQNB w KQkq - 0 1",
    "rkbrnqnb/pppppppp/8/8/8/8/PPPPPPPP/RKBRNQNB w KQkq - 0 1",
    "nnbrkrqb/pppppppp/8/8/8/8/PPPPPPPP/NNBRKRQB w KQkq - 0 1",
    "nrbnkrqb/pppppppp/8/8/8/8/PPPPPPPP/NRBNKRQB w KQkq - 0 1",
    "nrbknrqb/pppppppp/8/8/8/8/PPPPPPPP/NRBKNRQB w KQkq - 0 1",
    "nrbkrnqb/pppppppp/8/8/8/8/PPPPPPPP/NRBKRNQB w KQkq - 0 1",
    "rnbnkrqb/pppppppp/8/8/8/8/PPPPPPPP/RNBNKRQB w KQkq - 0 1",
    "rnbknrqb/pppppppp/8/8/8/8/PPPPPPPP/RNBKNRQB w KQkq - 0 1",
    "rnbkrnqb/pppppppp/8/8/8/8/PPPPPPPP/RNBKRNQB w KQkq - 0 1",
    "rkbnnrqb/pppppppp/8/8/8/8/PPPPPPPP/RKBNNRQB w KQkq - 0 1",
    "rkbnrnqb/pppppppp/8/8/8/8/PPPPPPPP/RKBNRNQB w KQkq - 0 1",
    "rkbrnnqb/pppppppp/8/8/8/8/PPPPPPPP/RKBRNNQB w KQkq - 0 1",
    "qbnnbrkr/pppppppp/8/8/8/8/PPPPPPPP/QBNNBRKR w KQkq - 0 1",
    "qbnrbnkr/pppppppp/8/8/8/8/PPPPPPPP/QBNRBNKR w KQkq - 0 1",
    "qbnrbknr/pppppppp/8/8/8/8/PPPPPPPP/QBNRBKNR w KQkq - 0 1",
    "qbnrbkrn/pppppppp/8/8/8/8/PPPPPPPP/QBNRBKRN w KQkq - 0 1",
    "qbrnbnkr/pppppppp/8/8/8/8/PPPPPPPP/QBRNBNKR w KQkq - 0 1",
    "qbrnbknr/pppppppp/8/8/8/8/PPPPPPPP/QBRNBKNR w KQkq - 0 1",
    "qbrnbkrn/pppppppp/8/8/8/8/PPPPPPPP/QBRNBKRN w KQkq - 0 1",
    "qbrkbnnr/pppppppp/8/8/8/8/PPPPPPPP/QBRKBNNR w KQkq - 0 1",
    "qbrkbnrn/pppppppp/8/8/8/8/PPPPPPPP/QBRKBNRN w KQkq - 0 1",
    "qbrkbrnn/pppppppp/8/8/8/8/PPPPPPPP/QBRKBRNN w KQkq - 0 1",
    "nbqnbrkr/pppppppp/8/8/8/8/PPPPPPPP/NBQNBRKR w KQkq - 0 1",
    "nbqrbnkr/pppppppp/8/8/8/8/PPPPPPPP/NBQRBNKR w KQkq - 0 1",
    "nbqrbknr/pppppppp/8/8/8/8/PPPPPPPP/NBQRBKNR w KQkq - 0 1",
    "nbqrbkrn/pppppppp/8/8/8/8/PPPPPPPP/NBQRBKRN w KQkq - 0 1",
    "rbqnbnkr/pppppppp/8/8/8/8/PPPPPPPP/RBQNBNKR w KQkq - 0 1",
    "rbqnbknr/pppppppp/8/8/8/8/PPPPPPPP/RBQNBKNR w KQkq - 0 1",
    "rbqnbkrn/pppppppp/8/8/8/8/PPPPPPPP/RBQNBKRN w KQkq - 0 1",
    "rbqkbnnr/pppppppp/8/8/8/8/PPPPPPPP/RBQKBNNR w KQkq - 0 1",
    "rbqkbnrn/pppppppp/8/8/8/8/PPPPPPPP/RBQKBNRN w KQkq - 0 1",
    "rbqkbrnn/pppppppp/8/8/8/8/PPPPPPPP/RBQKBRNN w KQkq - 0 1",
    "nbnqbrkr/pppppppp/8/8/8/8/PPPPPPPP/NBNQBRKR w KQkq - 0 1",
    "nbrqbnkr/pppppppp/8/8/8/8/PPPPPPPP/NBRQBNKR w KQkq - 0 1",
    "nbrqbknr/pppppppp/8/8/8/8/PPPPPPPP/NBRQBKNR w KQkq - 0 1",
    "nbrqbkrn/pppppppp/8/8/8/8/PPPPPPPP/NBRQBKRN w KQkq - 0 1",
    "rbnqbnkr/pppppppp/8/8/8/8/PPPPPPPP/RBNQBNKR w KQkq - 0 1",
    "rbnqbknr/pppppppp/8/8/8/8/PPPPPPPP/RBNQBKNR w KQkq - 0 1",
    "rbnqbkrn/pppppppp/8/8/8/8/PPPPPPPP/RBNQBKRN w KQkq - 0 1",
    "rbkqbnnr/pppppppp/8/8/8/8/PPPPPPPP/RBKQBNNR w KQkq - 0 1",
    "rbkqbnrn/pppppppp/8/8/8/8/PPPPPPPP/RBKQBNRN w KQkq - 0 1",
    "rbkqbrnn/pppppppp/8/8/8/8/PPPPPPPP/RBKQBRNN w KQkq - 0 1",
    "nbnrbqkr/pppppppp/8/8/8/8/PPPPPPPP/NBNRBQKR w KQkq - 0 1",
    "nbrnbqkr/pppppppp/8/8/8/8/PPPPPPPP/NBRNBQKR w KQkq - 0 1",
    "nbrkbqnr/pppppppp/8/8/8/8/PPPPPPPP/NBRKBQNR w KQkq - 0 1",
    "nbrkbqrn/pppppppp/8/8/8/8/PPPPPPPP/NBRKBQRN w KQkq - 0 1",
    "rbnnbqkr/pppppppp/8/8/8/8/PPPPPPPP/RBNNBQKR w KQkq - 0 1",
    "rbnkbqnr/pppppppp/8/8/8/8/PPPPPPPP/RBNKBQNR w KQkq - 0 1",
    "rbnkbqrn/pppppppp/8/8/8/8/PPPPPPPP/RBNKBQRN w KQkq - 0 1",
    "rbknbqnr/pppppppp/8/8/8/8/PPPPPPPP/RBKNBQNR w KQkq - 0 1",
    "rbknbqrn/pppppppp/8/8/8/8/PPPPPPPP/RBKNBQRN w KQkq - 0 1",
    "rbkrbqnn/pppppppp/8/8/8/8/PPPPPPPP/RBKRBQNN w KQkq - 0 1",
    "nbnrbkqr/pppppppp/8/8/8/8/PPPPPPPP/NBNRBKQR w KQkq - 0 1",
    "nbrnbkqr/pppppppp/8/8/8/8/PPPPPPPP/NBRNBKQR w KQkq - 0 1",
    "nbrkbnqr/pppppppp/8/8/8/8/PPPPPPPP/NBRKBNQR w KQkq - 0 1",
    "nbrkbrqn/pppppppp/8/8/8/8/PPPPPPPP/NBRKBRQN w KQkq - 0 1",
    "rbnnbkqr/pppppppp/8/8/8/8/PPPPPPPP/RBNNBKQR w KQkq - 0 1",
    "rbnkbnqr/pppppppp/8/8/8/8/PPPPPPPP/RBNKBNQR w KQkq - 0 1",
    "rbnkbrqn/pppppppp/8/8/8/8/PPPPPPPP/RBNKBRQN w KQkq - 0 1",
    "rbknbnqr/pppppppp/8/8/8/8/PPPPPPPP/RBKNBNQR w KQkq - 0 1",
    "rbknbrqn/pppppppp/8/8/8/8/PPPPPPPP/RBKNBRQN w KQkq - 0 1",
    "rbkrbnqn/pppppppp/8/8/8/8/PPPPPPPP/RBKRBNQN w KQkq - 0 1",
    "nbnrbkrq/pppppppp/8/8/8/8/PPPPPPPP/NBNRBKRQ w KQkq - 0 1",
    "nbrnbkrq/pppppppp/8/8/8/8/PPPPPPPP/NBRNBKRQ w KQkq - 0 1",
    "nbrkbnrq/pppppppp/8/8/8/8/PPPPPPPP/NBRKBNRQ w KQkq - 0 1",
    "nbrkbrnq/pppppppp/8/8/8/8/PPPPPPPP/NBRKBRNQ w KQkq - 0 1",
    "rbnnbkrq/pppppppp/8/8/8/8/PPPPPPPP/RBNNBKRQ w KQkq - 0 1",
    "rbnkbnrq/pppppppp/8/8/8/8/PPPPPPPP/RBNKBNRQ w KQkq - 0 1",
    "rbnkbrnq/pppppppp/8/8/8/8/PPPPPPPP/RBNKBRNQ w KQkq - 0 1",
    "rbknbnrq/pppppppp/8/8/8/8/PPPPPPPP/RBKNBNRQ w KQkq - 0 1",
    "rbknbrnq/pppppppp/8/8/8/8/PPPPPPPP/RBKNBRNQ w KQkq - 0 1",
    "rbkrbnnq/pppppppp/8/8/8/8/PPPPPPPP/RBKRBNNQ w KQkq - 0 1",
    "qnnbbrkr/pppppppp/8/8/8/8/PPPPPPPP/QNNBBRKR w KQkq - 0 1",
    "qnrbbnkr/pppppppp/8/8/8/8/PPPPPPPP/QNRBBNKR w KQkq - 0 1",
    "qnrbbknr/pppppppp/8/8/8/8/PPPPPPPP/QNRBBKNR w KQkq - 0 1",
    "qnrbbkrn/pppppppp/8/8/8/8/PPPPPPPP/QNRBBKRN w KQkq - 0 1",
    "qrnbbnkr/pppppppp/8/8/8/8/PPPPPPPP/QRNBBNKR w KQkq - 0 1",
    "qrnbbknr/pppppppp/8/8/8/8/PPPPPPPP/QRNBBKNR w KQkq - 0 1",
    "qrnbbkrn/pppppppp/8/8/8/8/PPPPPPPP/QRNBBKRN w KQkq - 0 1",
    "qrkbbnnr/pppppppp/8/8/8/8/PPPPPPPP/QRKBBNNR w KQkq - 0 1",
    "qrkbbnrn/pppppppp/8/8/8/8/PPPPPPPP/QRKBBNRN w KQkq - 0 1",
    "qrkbbrnn/pppppppp/8/8/8/8/PPPPPPPP/QRKBBRNN w KQkq - 0 1",
    "nqnbbrkr/pppppppp/8/8/8/8/PPPPPPPP/NQNBBRKR w KQkq - 0 1",
    "nqrbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NQRBBNKR w KQkq - 0 1",
    "nqrbbknr/pppppppp/8/8/8/8/PPPPPPPP/NQRBBKNR w KQkq - 0 1",
    "nqrbbkrn/pppppppp/8/8/8/8/PPPPPPPP/NQRBBKRN w KQkq - 0 1",
    "rqnbbnkr/pppppppp/8/8/8/8/PPPPPPPP/RQNBBNKR w KQkq - 0 1",
    "rqnbbknr/pppppppp/8/8/8/8/PPPPPPPP/RQNBBKNR w KQkq - 0 1",
    "rqnbbkrn/pppppppp/8/8/8/8/PPPPPPPP/RQNBBKRN w KQkq - 0 1",
    "rqkbbnnr/pppppppp/8/8/8/8/PPPPPPPP/RQKBBNNR w KQkq - 0 1",
    "rqkbbnrn/pppppppp/8/8/8/8/PPPPPPPP/RQKBBNRN w KQkq - 0 1",
    "rqkbbrnn/pppppppp/8/8/8/8/PPPPPPPP/RQKBBRNN w KQkq - 0 1",
    "nnqbbrkr/pppppppp/8/8/8/8/PPPPPPPP/NNQBBRKR w KQkq - 0 1",
    "nrqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBNKR w KQkq - 0 1",
    "nrqbbknr/pppppppp/8/8/8/8/PPPPPPPP/NRQBBKNR w KQkq - 0 1",
    "nrqbbkrn/pppppppp/8/8/8/8/PPPPPPPP/NRQBBKRN w KQkq - 0 1",
    "rnqbbnkr/pppppppp/8/8/8/8/PPPPPPPP/RNQBBNKR w KQkq - 0 1",
    "rnqbbknr/pppppppp/8/8/8/8/PPPPPPPP/RNQBBKNR w KQkq - 0 1",
    "rnqbbkrn/pppppppp/8/8/8/8/PPPPPPPP/RNQBBKRN w KQkq - 0 1",
    "rkqbbnnr/pppppppp/8/8/8/8/PPPPPPPP/RKQBBNNR w KQkq - 0 1",
    "rkqbbnrn/pppppppp/8/8/8/8/PPPPPPPP/RKQBBNRN w KQkq - 0 1",
    "rkqbbrnn/pppppppp/8/8/8/8/PPPPPPPP/RKQBBRNN w KQkq - 0 1",
    "nnrbbqkr/pppppppp/8/8/8/8/PPPPPPPP/NNRBBQKR w KQkq - 0 1",
    "nrnbbqkr/pppppppp/8/8/8/8/PPPPPPPP/NRNBBQKR w KQkq - 0 1",
    "nrkbbqnr/pppppppp/8/8/8/8/PPPPPPPP/NRKBBQNR w KQkq - 0 1",
    "nrkbbqrn/pppppppp/8/8/8/8/PPPPPPPP/NRKBBQRN w KQkq - 0 1",
    "rnnbbqkr/pppppppp/8/8/8/8/PPPPPPPP/RNNBBQKR w KQkq - 0 1",
    "rnkbbqnr/pppppppp/8/8/8/8/PPPPPPPP/RNKBBQNR w KQkq - 0 1",
    "rnkbbqrn/pppppppp/8/8/8/8/PPPPPPPP/RNKBBQRN w KQkq - 0 1",
    "rknbbqnr/pppppppp/8/8/8/8/PPPPPPPP/RKNBBQNR w KQkq - 0 1",
    "rknbbqrn/pppppppp/8/8/8/8/PPPPPPPP/RKNBBQRN w KQkq - 0 1",
    "rkrbbqnn/pppppppp/8/8/8/8/PPPPPPPP/RKRBBQNN w KQkq - 0 1",
    "nnrbbkqr/pppppppp/8/8/8/8/PPPPPPPP/NNRBBKQR w KQkq - 0 1",
    "nrnbbkqr/pppppppp/8/8/8/8/PPPPPPPP/NRNBBKQR w KQkq - 0 1",
    "nrkbbnqr/pppppppp/8/8/8/8/PPPPPPPP/NRKBBNQR w KQkq - 0 1",
    "nrkbbrqn/pppppppp/8/8/8/8/PPPPPPPP/NRKBBRQN w KQkq - 0 1",
    "rnnbbkqr/pppppppp/8/8/8/8/PPPPPPPP/RNNBBKQR w KQkq - 0 1",
    "rnkbbnqr/pppppppp/8/8/8/8/PPPPPPPP/RNKBBNQR w KQkq - 0 1",
    "rnkbbrqn/pppppppp/8/8/8/8/PPPPPPPP/RNKBBRQN w KQkq - 0 1",
    "rknbbnqr/pppppppp/8/8/8/8/PPPPPPPP/RKNBBNQR w KQkq - 0 1",
    "rknbbrqn/pppppppp/8/8/8/8/PPPPPPPP/RKNBBRQN w KQkq - 0 1",
    "rkrbbnqn/pppppppp/8/8/8/8/PPPPPPPP/RKRBBNQN w KQkq - 0 1",
    "nnrbbkrq/pppppppp/8/8/8/8/PPPPPPPP/NNRBBKRQ w KQkq - 0 1",
    "nrnbbkrq/pppppppp/8/8/8/8/PPPPPPPP/NRNBBKRQ w KQkq - 0 1",
    "nrkbbnrq/pppppppp/8/8/8/8/PPPPPPPP/NRKBBNRQ w KQkq - 0 1",
    "nrkbbrnq/pppppppp/8/8/8/8/PPPPPPPP/NRKBBRNQ w KQkq - 0 1",
    "rnnbbkrq/pppppppp/8/8/8/8/PPPPPPPP/RNNBBKRQ w KQkq - 0 1",
    "rnkbbnrq/pppppppp/8/8/8/8/PPPPPPPP/RNKBBNRQ w KQkq - 0 1",
    "rnkbbrnq/pppppppp/8/8/8/8/PPPPPPPP/RNKBBRNQ w KQkq - 0 1",
    "rknbbnrq/pppppppp/8/8/8/8/PPPPPPPP/RKNBBNRQ w KQkq - 0 1",
    "rknbbrnq/pppppppp/8/8/8/8/PPPPPPPP/RKNBBRNQ w KQkq - 0 1",
    "rkrbbnnq/pppppppp/8/8/8/8/PPPPPPPP/RKRBBNNQ w KQkq - 0 1",
    "qnnrbbkr/pppppppp/8/8/8/8/PPPPPPPP/QNNRBBKR w KQkq - 0 1",
    "qnrnbbkr/pppppppp/8/8/8/8/PPPPPPPP/QNRNBBKR w KQkq - 0 1",
    "qnrkbbnr/pppppppp/8/8/8/8/PPPPPPPP/QNRKBBNR w KQkq - 0 1",
    "qnrkbbrn/pppppppp/8/8/8/8/PPPPPPPP/QNRKBBRN w KQkq - 0 1",
    "qrnnbbkr/pppppppp/8/8/8/8/PPPPPPPP/QRNNBBKR w KQkq - 0 1",
    "qrnkbbnr/pppppppp/8/8/8/8/PPPPPPPP/QRNKBBNR w KQkq - 0 1",
    "qrnkbbrn/pppppppp/8/8/8/8/PPPPPPPP/QRNKBBRN w KQkq - 0 1",
    "qrknbbnr/pppppppp/8/8/8/8/PPPPPPPP/QRKNBBNR w KQkq - 0 1",
    "qrknbbrn/pppppppp/8/8/8/8/PPPPPPPP/QRKNBBRN w KQkq - 0 1",
    "qrkrbbnn/pppppppp/8/8/8/8/PPPPPPPP/QRKRBBNN w KQkq - 0 1",
    "nqnrbbkr/pppppppp/8/8/8/8/PPPPPPPP/NQNRBBKR w KQkq - 0 1",
    "nqrnbbkr/pppppppp/8/8/8/8/PPPPPPPP/NQRNBBKR w KQkq - 0 1",
    "nqrkbbnr/pppppppp/8/8/8/8/PPPPPPPP/NQRKBBNR w KQkq - 0 1",
    "nqrkbbrn/pppppppp/8/8/8/8/PPPPPPPP/NQRKBBRN w KQkq - 0 1",
    "rqnnbbkr/pppppppp/8/8/8/8/PPPPPPPP/RQNNBBKR w KQkq - 0 1",
    "rqnkbbnr/pppppppp/8/8/8/8/PPPPPPPP/RQNKBBNR w KQkq - 0 1",
    "rqnkbbrn/pppppppp/8/8/8/8/PPPPPPPP/RQNKBBRN w KQkq - 0 1",
    "rqknbbnr/pppppppp/8/8/8/8/PPPPPPPP/RQKNBBNR w KQkq - 0 1",
    "rqknbbrn/pppppppp/8/8/8/8/PPPPPPPP/RQKNBBRN w KQkq - 0 1",
    "rqkrbbnn/pppppppp/8/8/8/8/PPPPPPPP/RQKRBBNN w KQkq - 0 1",
    "nnqrbbkr/pppppppp/8/8/8/8/PPPPPPPP/NNQRBBKR w KQkq - 0 1",
    "nrqnbbkr/pppppppp/8/8/8/8/PPPPPPPP/NRQNBBKR w KQkq - 0 1",
    "nrqkbbnr/pppppppp/8/8/8/8/PPPPPPPP/NRQKBBNR w KQkq - 0 1",
    "nrqkbbrn/pppppppp/8/8/8/8/PPPPPPPP/NRQKBBRN w KQkq - 0 1",
    "rnqnbbkr/pppppppp/8/8/8/8/PPPPPPPP/RNQNBBKR w KQkq - 0 1",
    "rnqkbbnr/pppppppp/8/8/8/8/PPPPPPPP/RNQKBBNR w KQkq - 0 1",
    "rnqkbbrn/pppppppp/8/8/8/8/PPPPPPPP/RNQKBBRN w KQkq - 0 1",
    "rkqnbbnr/pppppppp/8/8/8/8/PPPPPPPP/RKQNBBNR w KQkq - 0 1",
    "rkqnbbrn/pppppppp/8/8/8/8/PPPPPPPP/RKQNBBRN w KQkq - 0 1",
    "rkqrbbnn/pppppppp/8/8/8/8/PPPPPPPP/RKQRBBNN w KQkq - 0 1",
    "nnrqbbkr/pppppppp/8/8/8/8/PPPPPPPP/NNRQBBKR w KQkq - 0 1",
    "nrnqbbkr/pppppppp/8/8/8/8/PPPPPPPP/NRNQBBKR w KQkq - 0 1",
    "nrkqbbnr/pppppppp/8/8/8/8/PPPPPPPP/NRKQBBNR w KQkq - 0 1",
    "nrkqbbrn/pppppppp/8/8/8/8/PPPPPPPP/NRKQBBRN w KQkq - 0 1",
    "rnnqbbkr/pppppppp/8/8/8/8/PPPPPPPP/RNNQBBKR w KQkq - 0 1",
    "rnkqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RNKQBBNR w KQkq - 0 1",
    "rnkqbbrn/pppppppp/8/8/8/8/PPPPPPPP/RNKQBBRN w KQkq - 0 1",
    "rknqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RKNQBBNR w KQkq - 0 1",
    "rknqbbrn/pppppppp/8/8/8/8/PPPPPPPP/RKNQBBRN w KQkq - 0 1",
    "rkrqbbnn/pppppppp/8/8/8/8/PPPPPPPP/RKRQBBNN w KQkq - 0 1",
    "nnrkbbqr/pppppppp/8/8/8/8/PPPPPPPP/NNRKBBQR w KQkq - 0 1",
    "nrnkbbqr/pppppppp/8/8/8/8/PPPPPPPP/NRNKBBQR w KQkq - 0 1",
    "nrknbbqr/pppppppp/8/8/8/8/PPPPPPPP/NRKNBBQR w KQkq - 0 1",
    "nrkrbbqn/pppppppp/8/8/8/8/PPPPPPPP/NRKRBBQN w KQkq - 0 1",
    "rnnkbbqr/pppppppp/8/8/8/8/PPPPPPPP/RNNKBBQR w KQkq - 0 1",
    "rnknbbqr/pppppppp/8/8/8/8/PPPPPPPP/RNKNBBQR w KQkq - 0 1",
    "rnkrbbqn/pppppppp/8/8/8/8/PPPPPPPP/RNKRBBQN w KQkq - 0 1",
    "rknnbbqr/pppppppp/8/8/8/8/PPPPPPPP/RKNNBBQR w KQkq - 0 1",
    "rknrbbqn/pppppppp/8/8/8/8/PPPPPPPP/RKNRBBQN w KQkq - 0 1",
    "rkrnbbqn/pppppppp/8/8/8/8/PPPPPPPP/RKRNBBQN w KQkq - 0 1",
    "nnrkbbrq/pppppppp/8/8/8/8/PPPPPPPP/NNRKBBRQ w KQkq - 0 1",
    "nrnkbbrq/pppppppp/8/8/8/8/PPPPPPPP/NRNKBBRQ w KQkq - 0 1",
    "nrknbbrq/pppppppp/8/8/8/8/PPPPPPPP/NRKNBBRQ w KQkq - 0 1",
    "nrkrbbnq/pppppppp/8/8/8/8/PPPPPPPP/NRKRBBNQ w KQkq - 0 1",
    "rnnkbbrq/pppppppp/8/8/8/8/PPPPPPPP/RNNKBBRQ w KQkq - 0 1",
    "rnknbbrq/pppppppp/8/8/8/8/PPPPPPPP/RNKNBBRQ w KQkq - 0 1",
    "rnkrbbnq/pppppppp/8/8/8/8/PPPPPPPP/RNKRBBNQ w KQkq - 0 1",
    "rknnbbrq/pppppppp/8/8/8/8/PPPPPPPP/RKNNBBRQ w KQkq - 0 1",
    "rknrbbnq/pppppppp/8/8/8/8/PPPPPPPP/RKNRBBNQ w KQkq - 0 1",
    "rkrnbbnq/pppppppp/8/8/8/8/PPPPPPPP/RKRNBBNQ w KQkq - 0 1",
    "qnnrbkrb/pppppppp/8/8/8/8/PPPPPPPP/QNNRBKRB w KQkq - 0 1",
    "qnrnbkrb/pppppppp/8/8/8/8/PPPPPPPP/QNRNBKRB w KQkq - 0 1",
    "qnrkbnrb/pppppppp/8/8/8/8/PPPPPPPP/QNRKBNRB w KQkq - 0 1",
    "qnrkbrnb/pppppppp/8/8/8/8/PPPPPPPP/QNRKBRNB w KQkq - 0 1",
    "qrnnbkrb/pppppppp/8/8/8/8/PPPPPPPP/QRNNBKRB w KQkq - 0 1",
    "qrnkbnrb/pppppppp/8/8/8/8/PPPPPPPP/QRNKBNRB w KQkq - 0 1",
    "qrnkbrnb/pppppppp/8/8/8/8/PPPPPPPP/QRNKBRNB w KQkq - 0 1",
    "qrknbnrb/pppppppp/8/8/8/8/PPPPPPPP/QRKNBNRB w KQkq - 0 1",
    "qrknbrnb/pppppppp/8/8/8/8/PPPPPPPP/QRKNBRNB w KQkq - 0 1",
    "qrkrbnnb/pppppppp/8/8/8/8/PPPPPPPP/QRKRBNNB w KQkq - 0 1",
    "nqnrbkrb/pppppppp/8/8/8/8/PPPPPPPP/NQNRBKRB w KQkq - 0 1",
    "nqrnbkrb/pppppppp/8/8/8/8/PPPPPPPP/NQRNBKRB w KQkq - 0 1",
    "nqrkbnrb/pppppppp/8/8/8/8/PPPPPPPP/NQRKBNRB w KQkq - 0 1",
    "nqrkbrnb/pppppppp/8/8/8/8/PPPPPPPP/NQRKBRNB w KQkq - 0 1",
    "rqnnbkrb/pppppppp/8/8/8/8/PPPPPPPP/RQNNBKRB w KQkq - 0 1",
    "rqnkbnrb/pppppppp/8/8/8/8/PPPPPPPP/RQNKBNRB w KQkq - 0 1",
    "rqnkbrnb/pppppppp/8/8/8/8/PPPPPPPP/RQNKBRNB w KQkq - 0 1",
    "rqknbnrb/pppppppp/8/8/8/8/PPPPPPPP/RQKNBNRB w KQkq - 0 1",
    "rqknbrnb/pppppppp/8/8/8/8/PPPPPPPP/RQKNBRNB w KQkq - 0 1",
    "rqkrbnnb/pppppppp/8/8/8/8/PPPPPPPP/RQKRBNNB w KQkq - 0 1",
    "nnqrbkrb/pppppppp/8/8/8/8/PPPPPPPP/NNQRBKRB w KQkq - 0 1",
    "nrqnbkrb/pppppppp/8/8/8/8/PPPPPPPP/NRQNBKRB w KQkq - 0 1",
    "nrqkbnrb/pppppppp/8/8/8/8/PPPPPPPP/NRQKBNRB w KQkq - 0 1",
    "nrqkbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRQKBRNB w KQkq - 0 1",
    "rnqnbkrb/pppppppp/8/8/8/8/PPPPPPPP/RNQNBKRB w KQkq - 0 1",
    "rnqkbnrb/pppppppp/8/8/8/8/PPPPPPPP/RNQKBNRB w KQkq - 0 1",
    "rnqkbrnb/pppppppp/8/8/8/8/PPPPPPPP/RNQKBRNB w KQkq - 0 1",
    "rkqnbnrb/pppppppp/8/8/8/8/PPPPPPPP/RKQNBNRB w KQkq - 0 1",
    "rkqnbrnb/pppppppp/8/8/8/8/PPPPPPPP/RKQNBRNB w KQkq - 0 1",
    "rkqrbnnb/pppppppp/8/8/8/8/PPPPPPPP/RKQRBNNB w KQkq - 0 1",
    "nnrqbkrb/pppppppp/8/8/8/8/PPPPPPPP/NNRQBKRB w KQkq - 0 1",
    "nrnqbkrb/pppppppp/8/8/8/8/PPPPPPPP/NRNQBKRB w KQkq - 0 1",
    "nrkqbnrb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBNRB w KQkq - 0 1",
    "nrkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBRNB w KQkq - 0 1",
    "rnnqbkrb/pppppppp/8/8/8/8/PPPPPPPP/RNNQBKRB w KQkq - 0 1",
    "rnkqbnrb/pppppppp/8/8/8/8/PPPPPPPP/RNKQBNRB w KQkq - 0 1",
    "rnkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/RNKQBRNB w KQkq - 0 1",
    "rknqbnrb/pppppppp/8/8/8/8/PPPPPPPP/RKNQBNRB w KQkq - 0 1",
    "rknqbrnb/pppppppp/8/8/8/8/PPPPPPPP/RKNQBRNB w KQkq - 0 1",
    "rkrqbnnb/pppppppp/8/8/8/8/PPPPPPPP/RKRQBNNB w KQkq - 0 1",
    "nnrkbqrb/pppppppp/8/8/8/8/PPPPPPPP/NNRKBQRB w KQkq - 0 1",
    "nrnkbqrb/pppppppp/8/8/8/8/PPPPPPPP/NRNKBQRB w KQkq - 0 1",
    "nrknbqrb/pppppppp/8/8/8/8/PPPPPPPP/NRKNBQRB w KQkq - 0 1",
    "nrkrbqnb/pppppppp/8/8/8/8/PPPPPPPP/NRKRBQNB w KQkq - 0 1",
    "rnnkbqrb/pppppppp/8/8/8/8/PPPPPPPP/RNNKBQRB w KQkq - 0 1",
    "rnknbqrb/pppppppp/8/8/8/8/PPPPPPPP/RNKNBQRB w KQkq - 0 1",
    "rnkrbqnb/pppppppp/8/8/8/8/PPPPPPPP/RNKRBQNB w KQkq - 0 1",
    "rknnbqrb/pppppppp/8/8/8/8/PPPPPPPP/RKNNBQRB w KQkq - 0 1",
    "rknrbqnb/pppppppp/8/8/8/8/PPPPPPPP/RKNRBQNB w KQkq - 0 1",
    "rkrnbqnb/pppppppp/8/8/8/8/PPPPPPPP/RKRNBQNB w KQkq - 0 1",
    "nnrkbrqb/pppppppp/8/8/8/8/PPPPPPPP/NNRKBRQB w KQkq - 0 1",
    "nrnkbrqb/pppppppp/8/8/8/8/PPPPPPPP/NRNKBRQB w KQkq - 0 1",
    "nrknbrqb/pppppppp/8/8/8/8/PPPPPPPP/NRKNBRQB w KQkq - 0 1",
    "nrkrbnqb/pppppppp/8/8/8/8/PPPPPPPP/NRKRBNQB w KQkq - 0 1",
    "rnnkbrqb/pppppppp/8/8/8/8/PPPPPPPP/RNNKBRQB w KQkq - 0 1",
    "rnknbrqb/pppppppp/8/8/8/8/PPPPPPPP/RNKNBRQB w KQkq - 0 1",
    "rnkrbnqb/pppppppp/8/8/8/8/PPPPPPPP/RNKRBNQB w KQkq - 0 1",
    "rknnbrqb/pppppppp/8/8/8/8/PPPPPPPP/RKNNBRQB w KQkq - 0 1",
    "rknrbnqb/pppppppp/8/8/8/8/PPPPPPPP/RKNRBNQB w KQkq - 0 1",
    "rkrnbnqb/pppppppp/8/8/8/8/PPPPPPPP/RKRNBNQB w KQkq - 0 1",
    "qbnnrkbr/pppppppp/8/8/8/8/PPPPPPPP/QBNNRKBR w KQkq - 0 1",
    "qbnrnkbr/pppppppp/8/8/8/8/PPPPPPPP/QBNRNKBR w KQkq - 0 1",
    "qbnrknbr/pppppppp/8/8/8/8/PPPPPPPP/QBNRKNBR w KQkq - 0 1",
    "qbnrkrbn/pppppppp/8/8/8/8/PPPPPPPP/QBNRKRBN w KQkq - 0 1",
    "qbrnnkbr/pppppppp/8/8/8/8/PPPPPPPP/QBRNNKBR w KQkq - 0 1",
    "qbrnknbr/pppppppp/8/8/8/8/PPPPPPPP/QBRNKNBR w KQkq - 0 1",
    "qbrnkrbn/pppppppp/8/8/8/8/PPPPPPPP/QBRNKRBN w KQkq - 0 1",
    "qbrknnbr/pppppppp/8/8/8/8/PPPPPPPP/QBRKNNBR w KQkq - 0 1",
    "qbrknrbn/pppppppp/8/8/8/8/PPPPPPPP/QBRKNRBN w KQkq - 0 1",
    "qbrkrnbn/pppppppp/8/8/8/8/PPPPPPPP/QBRKRNBN w KQkq - 0 1",
    "nbqnrkbr/pppppppp/8/8/8/8/PPPPPPPP/NBQNRKBR w KQkq - 0 1",
    "nbqrnkbr/pppppppp/8/8/8/8/PPPPPPPP/NBQRNKBR w KQkq - 0 1",
    "nbqrknbr/pppppppp/8/8/8/8/PPPPPPPP/NBQRKNBR w KQkq - 0 1",
    "nbqrkrbn/pppppppp/8/8/8/8/PPPPPPPP/NBQRKRBN w KQkq - 0 1",
    "rbqnnkbr/pppppppp/8/8/8/8/PPPPPPPP/RBQNNKBR w KQkq - 0 1",
    "rbqnknbr/pppppppp/8/8/8/8/PPPPPPPP/RBQNKNBR w KQkq - 0 1",
    "rbqnkrbn/pppppppp/8/8/8/8/PPPPPPPP/RBQNKRBN w KQkq - 0 1",
    "rbqknnbr/pppppppp/8/8/8/8/PPPPPPPP/RBQKNNBR w KQkq - 0 1",
    "rbqknrbn/pppppppp/8/8/8/8/PPPPPPPP/RBQKNRBN w KQkq - 0 1",
    "rbqkrnbn/pppppppp/8/8/8/8/PPPPPPPP/RBQKRNBN w KQkq - 0 1",
    "nbnqrkbr/pppppppp/8/8/8/8/PPPPPPPP/NBNQRKBR w KQkq - 0 1",
    "nbrqnkbr/pppppppp/8/8/8/8/PPPPPPPP/NBRQNKBR w KQkq - 0 1",
    "nbrqknbr/pppppppp/8/8/8/8/PPPPPPPP/NBRQKNBR w KQkq - 0 1",
    "nbrqkrbn/pppppppp/8/8/8/8/PPPPPPPP/NBRQKRBN w KQkq - 0 1",
    "rbnqnkbr/pppppppp/8/8/8/8/PPPPPPPP/RBNQNKBR w KQkq - 0 1",
    "rbnqknbr/pppppppp/8/8/8/8/PPPPPPPP/RBNQKNBR w KQkq - 0 1",
    "rbnqkrbn/pppppppp/8/8/8/8/PPPPPPPP/RBNQKRBN w KQkq - 0 1",
    "rbkqnnbr/pppppppp/8/8/8/8/PPPPPPPP/RBKQNNBR w KQkq - 0 1",
    "rbkqnrbn/pppppppp/8/8/8/8/PPPPPPPP/RBKQNRBN w KQkq - 0 1",
    "rbkqrnbn/pppppppp/8/8/8/8/PPPPPPPP/RBKQRNBN w KQkq - 0 1",
    "nbnrqkbr/pppppppp/8/8/8/8/PPPPPPPP/NBNRQKBR w KQkq - 0 1",
    "nbrnqkbr/pppppppp/8/8/8/8/PPPPPPPP/NBRNQKBR w KQkq - 0 1",
    "nbrkqnbr/pppppppp/8/8/8/8/PPPPPPPP/NBRKQNBR w KQkq - 0 1",
    "nbrkqrbn/pppppppp/8/8/8/8/PPPPPPPP/NBRKQRBN w KQkq - 0 1",
    "rbnnqkbr/pppppppp/8/8/8/8/PPPPPPPP/RBNNQKBR w KQkq - 0 1",
    "rbnkqnbr/pppppppp/8/8/8/8/PPPPPPPP/RBNKQNBR w KQkq - 0 1",
    "rbnkqrbn/pppppppp/8/8/8/8/PPPPPPPP/RBNKQRBN w KQkq - 0 1",
    "rbknqnbr/pppppppp/8/8/8/8/PPPPPPPP/RBKNQNBR w KQkq - 0 1",
    "rbknqrbn/pppppppp/8/8/8/8/PPPPPPPP/RBKNQRBN w KQkq - 0 1",
    "rbkrqnbn/pppppppp/8/8/8/8/PPPPPPPP/RBKRQNBN w KQkq - 0 1",
    "nbnrkqbr/pppppppp/8/8/8/8/PPPPPPPP/NBNRKQBR w KQkq - 0 1",
    "nbrnkqbr/pppppppp/8/8/8/8/PPPPPPPP/NBRNKQBR w KQkq - 0 1",
    "nbrknqbr/pppppppp/8/8/8/8/PPPPPPPP/NBRKNQBR w KQkq - 0 1",
    "nbrkrqbn/pppppppp/8/8/8/8/PPPPPPPP/NBRKRQBN w KQkq - 0 1",
    "rbnnkqbr/pppppppp/8/8/8/8/PPPPPPPP/RBNNKQBR w KQkq - 0 1",
    "rbnknqbr/pppppppp/8/8/8/8/PPPPPPPP/RBNKNQBR w KQkq - 0 1",
    "rbnkrqbn/pppppppp/8/8/8/8/PPPPPPPP/RBNKRQBN w KQkq - 0 1",
    "rbknnqbr/pppppppp/8/8/8/8/PPPPPPPP/RBKNNQBR w KQkq - 0 1",
    "rbknrqbn/pppppppp/8/8/8/8/PPPPPPPP/RBKNRQBN w KQkq - 0 1",
    "rbkrnqbn/pppppppp/8/8/8/8/PPPPPPPP/RBKRNQBN w KQkq - 0 1",
    "nbnrkrbq/pppppppp/8/8/8/8/PPPPPPPP/NBNRKRBQ w KQkq - 0 1",
    "nbrnkrbq/pppppppp/8/8/8/8/PPPPPPPP/NBRNKRBQ w KQkq - 0 1",
    "nbrknrbq/pppppppp/8/8/8/8/PPPPPPPP/NBRKNRBQ w KQkq - 0 1",
    "nbrkrnbq/pppppppp/8/8/8/8/PPPPPPPP/NBRKRNBQ w KQkq - 0 1",
    "rbnnkrbq/pppppppp/8/8/8/8/PPPPPPPP/RBNNKRBQ w KQkq - 0 1",
    "rbnknrbq/pppppppp/8/8/8/8/PPPPPPPP/RBNKNRBQ w KQkq - 0 1",
    "rbnkrnbq/pppppppp/8/8/8/8/PPPPPPPP/RBNKRNBQ w KQkq - 0 1",
    "rbknnrbq/pppppppp/8/8/8/8/PPPPPPPP/RBKNNRBQ w KQkq - 0 1",
    "rbknrnbq/pppppppp/8/8/8/8/PPPPPPPP/RBKNRNBQ w KQkq - 0 1",
    "rbkrnnbq/pppppppp/8/8/8/8/PPPPPPPP/RBKRNNBQ w KQkq - 0 1",
    "qnnbrkbr/pppppppp/8/8/8/8/PPPPPPPP/QNNBRKBR w KQkq - 0 1",
    "qnrbnkbr/pppppppp/8/8/8/8/PPPPPPPP/QNRBNKBR w KQkq - 0 1",
    "qnrbknbr/pppppppp/8/8/8/8/PPPPPPPP/QNRBKNBR w KQkq - 0 1",
    "qnrbkrbn/pppppppp/8/8/8/8/PPPPPPPP/QNRBKRBN w KQkq - 0 1",
    "qrnbnkbr/pppppppp/8/8/8/8/PPPPPPPP/QRNBNKBR w KQkq - 0 1",
    "qrnbknbr/pppppppp/8/8/8/8/PPPPPPPP/QRNBKNBR w KQkq - 0 1",
    "qrnbkrbn/pppppppp/8/8/8/8/PPPPPPPP/QRNBKRBN w KQkq - 0 1",
    "qrkbnnbr/pppppppp/8/8/8/8/PPPPPPPP/QRKBNNBR w KQkq - 0 1",
    "qrkbnrbn/pppppppp/8/8/8/8/PPPPPPPP/QRKBNRBN w KQkq - 0 1",
    "qrkbrnbn/pppppppp/8/8/8/8/PPPPPPPP/QRKBRNBN w KQkq - 0 1",
    "nqnbrkbr/pppppppp/8/8/8/8/PPPPPPPP/NQNBRKBR w KQkq - 0 1",
    "nqrbnkbr/pppppppp/8/8/8/8/PPPPPPPP/NQRBNKBR w KQkq - 0 1",
    "nqrbknbr/pppppppp/8/8/8/8/PPPPPPPP/NQRBKNBR w KQkq - 0 1",
    "nqrbkrbn/pppppppp/8/8/8/8/PPPPPPPP/NQRBKRBN w KQkq - 0 1",
    "rqnbnkbr/pppppppp/8/8/8/8/PPPPPPPP/RQNBNKBR w KQkq - 0 1",
    "rqnbknbr/pppppppp/8/8/8/8/PPPPPPPP/RQNBKNBR w KQkq - 0 1",
    "rqnbkrbn/pppppppp/8/8/8/8/PPPPPPPP/RQNBKRBN w KQkq - 0 1",
    "rqkbnnbr/pppppppp/8/8/8/8/PPPPPPPP/RQKBNNBR w KQkq - 0 1",
    "rqkbnrbn/pppppppp/8/8/8/8/PPPPPPPP/RQKBNRBN w KQkq - 0 1",
    "rqkbrnbn/pppppppp/8/8/8/8/PPPPPPPP/RQKBRNBN w KQkq - 0 1",
    "nnqbrkbr/pppppppp/8/8/8/8/PPPPPPPP/NNQBRKBR w KQkq - 0 1",
    "nrqbnkbr/pppppppp/8/8/8/8/PPPPPPPP/NRQBNKBR w KQkq - 0 1",
    "nrqbknbr/pppppppp/8/8/8/8/PPPPPPPP/NRQBKNBR w KQkq - 0 1",
    "nrqbkrbn/pppppppp/8/8/8/8/PPPPPPPP/NRQBKRBN w KQkq - 0 1",
    "rnqbnkbr/pppppppp/8/8/8/8/PPPPPPPP/RNQBNKBR w KQkq - 0 1",
    "rnqbknbr/pppppppp/8/8/8/8/PPPPPPPP/RNQBKNBR w KQkq - 0 1",
    "rnqbkrbn/pppppppp/8/8/8/8/PPPPPPPP/RNQBKRBN w KQkq - 0 1",
    "rkqbnnbr/pppppppp/8/8/8/8/PPPPPPPP/RKQBNNBR w KQkq - 0 1",
    "rkqbnrbn/pppppppp/8/8/8/8/PPPPPPPP/RKQBNRBN w KQkq - 0 1",
    "rkqbrnbn/pppppppp/8/8/8/8/PPPPPPPP/RKQBRNBN w KQkq - 0 1",
    "nnrbqkbr/pppppppp/8/8/8/8/PPPPPPPP/NNRBQKBR w KQkq - 0 1",
    "nrnbqkbr/pppppppp/8/8/8/8/PPPPPPPP/NRNBQKBR w KQkq - 0 1",
    "nrkbqnbr/pppppppp/8/8/8/8/PPPPPPPP/NRKBQNBR w KQkq - 0 1",
    "nrkbqrbn/pppppppp/8/8/8/8/PPPPPPPP/NRKBQRBN w KQkq - 0 1",
    "rnnbqkbr/pppppppp/8/8/8/8/PPPPPPPP/RNNBQKBR w KQkq - 0 1",
    "rnkbqnbr/pppppppp/8/8/8/8/PPPPPPPP/RNKBQNBR w KQkq - 0 1",
    "rnkbqrbn/pppppppp/8/8/8/8/PPPPPPPP/RNKBQRBN w KQkq - 0 1",
    "rknbqnbr/pppppppp/8/8/8/8/PPPPPPPP/RKNBQNBR w KQkq - 0 1",
    "rknbqrbn/pppppppp/8/8/8/8/PPPPPPPP/RKNBQRBN w KQkq - 0 1",
    "rkrbqnbn/pppppppp/8/8/8/8/PPPPPPPP/RKRBQNBN w KQkq - 0 1",
    "nnrbkqbr/pppppppp/8/8/8/8/PPPPPPPP/NNRBKQBR w KQkq - 0 1",
    "nrnbkqbr/pppppppp/8/8/8/8/PPPPPPPP/NRNBKQBR w KQkq - 0 1",
    "nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/NRKBNQBR w KQkq - 0 1",
    "nrkbrqbn/pppppppp/8/8/8/8/PPPPPPPP/NRKBRQBN w KQkq - 0 1",
    "rnnbkqbr/pppppppp/8/8/8/8/PPPPPPPP/RNNBKQBR w KQkq - 0 1",
    "rnkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/RNKBNQBR w KQkq - 0 1",
    "rnkbrqbn/pppppppp/8/8/8/8/PPPPPPPP/RNKBRQBN w KQkq - 0 1",
    "rknbnqbr/pppppppp/8/8/8/8/PPPPPPPP/RKNBNQBR w KQkq - 0 1",
    "rknbrqbn/pppppppp/8/8/8/8/PPPPPPPP/RKNBRQBN w KQkq - 0 1",
    "rkrbnqbn/pppppppp/8/8/8/8/PPPPPPPP/RKRBNQBN w KQkq - 0 1",
    "nnrbkrbq/pppppppp/8/8/8/8/PPPPPPPP/NNRBKRBQ w KQkq - 0 1",
    "nrnbkrbq/pppppppp/8/8/8/8/PPPPPPPP/NRNBKRBQ w KQkq - 0 1",
    "nrkbnrbq/pppppppp/8/8/8/8/PPPPPPPP/NRKBNRBQ w KQkq - 0 1",
    "nrkbrnbq/pppppppp/8/8/8/8/PPPPPPPP/NRKBRNBQ w KQkq - 0 1",
    "rnnbkrbq/pppppppp/8/8/8/8/PPPPPPPP/RNNBKRBQ w KQkq - 0 1",
    "rnkbnrbq/pppppppp/8/8/8/8/PPPPPPPP/RNKBNRBQ w KQkq - 0 1",
    "rnkbrnbq/pppppppp/8/8/8/8/PPPPPPPP/RNKBRNBQ w KQkq - 0 1",
    "rknbnrbq/pppppppp/8/8/8/8/PPPPPPPP/RKNBNRBQ w KQkq - 0 1",
    "rknbrnbq/pppppppp/8/8/8/8/PPPPPPPP/RKNBRNBQ w KQkq - 0 1",
    "rkrbnnbq/pppppppp/8/8/8/8/PPPPPPPP/RKRBNNBQ w KQkq - 0 1",
    "qnnrkbbr/pppppppp/8/8/8/8/PPPPPPPP/QNNRKBBR w KQkq - 0 1",
    "qnrnkbbr/pppppppp/8/8/8/8/PPPPPPPP/QNRNKBBR w KQkq - 0 1",
    "qnrknbbr/pppppppp/8/8/8/8/PPPPPPPP/QNRKNBBR w KQkq - 0 1",
    "qnrkrbbn/pppppppp/8/8/8/8/PPPPPPPP/QNRKRBBN w KQkq - 0 1",
    "qrnnkbbr/pppppppp/8/8/8/8/PPPPPPPP/QRNNKBBR w KQkq - 0 1",
    "qrnknbbr/pppppppp/8/8/8/8/PPPPPPPP/QRNKNBBR w KQkq - 0 1",
    "qrnkrbbn/pppppppp/8/8/8/8/PPPPPPPP/QRNKRBBN w KQkq - 0 1",
    "qrknnbbr/pppppppp/8/8/8/8/PPPPPPPP/QRKNNBBR w KQkq - 0 1",
    "qrknrbbn/pppppppp/8/8/8/8/PPPPPPPP/QRKNRBBN w KQkq - 0 1",
    "qrkrnbbn/pppppppp/8/8/8/8/PPPPPPPP/QRKRNBBN w KQkq - 0 1",
    "nqnrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NQNRKBBR w KQkq - 0 1",
    "nqrnkbbr/pppppppp/8/8/8/8/PPPPPPPP/NQRNKBBR w KQkq - 0 1",
    "nqrknbbr/pppppppp/8/8/8/8/PPPPPPPP/NQRKNBBR w KQkq - 0 1",
    "nqrkrbbn/pppppppp/8/8/8/8/PPPPPPPP/NQRKRBBN w KQkq - 0 1",
    "rqnnkbbr/pppppppp/8/8/8/8/PPPPPPPP/RQNNKBBR w KQkq - 0 1",
    "rqnknbbr/pppppppp/8/8/8/8/PPPPPPPP/RQNKNBBR w KQkq - 0 1",
    "rqnkrbbn/pppppppp/8/8/8/8/PPPPPPPP/RQNKRBBN w KQkq - 0 1",
    "rqknnbbr/pppppppp/8/8/8/8/PPPPPPPP/RQKNNBBR w KQkq - 0 1",
    "rqknrbbn/pppppppp/8/8/8/8/PPPPPPPP/RQKNRBBN w KQkq - 0 1",
    "rqkrnbbn/pppppppp/8/8/8/8/PPPPPPPP/RQKRNBBN w KQkq - 0 1",
    "nnqrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNQRKBBR w KQkq - 0 1",
    "nrqnkbbr/pppppppp/8/8/8/8/PPPPPPPP/NRQNKBBR w KQkq - 0 1",
    "nrqknbbr/pppppppp/8/8/8/8/PPPPPPPP/NRQKNBBR w KQkq - 0 1",
    "nrqkrbbn/pppppppp/8/8/8/8/PPPPPPPP/NRQKRBBN w KQkq - 0 1",
    "rnqnkbbr/pppppppp/8/8/8/8/PPPPPPPP/RNQNKBBR w KQkq - 0 1",
    "rnqknbbr/pppppppp/8/8/8/8/PPPPPPPP/RNQKNBBR w KQkq - 0 1",
    "rnqkrbbn/pppppppp/8/8/8/8/PPPPPPPP/RNQKRBBN w KQkq - 0 1",
    "rkqnnbbr/pppppppp/8/8/8/8/PPPPPPPP/RKQNNBBR w KQkq - 0 1",
    "rkqnrbbn/pppppppp/8/8/8/8/PPPPPPPP/RKQNRBBN w KQkq - 0 1",
    "rkqrnbbn/pppppppp/8/8/8/8/PPPPPPPP/RKQRNBBN w KQkq - 0 1",
    "nnrqkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNRQKBBR w KQkq - 0 1",
    "nrnqkbbr/pppppppp/8/8/8/8/PPPPPPPP/NRNQKBBR w KQkq - 0 1",
    "nrkqnbbr/pppppppp/8/8/8/8/PPPPPPPP/NRKQNBBR w KQkq - 0 1",
    "nrkqrbbn/pppppppp/8/8/8/8/PPPPPPPP/NRKQRBBN w KQkq - 0 1",
    "rnnqkbbr/pppppppp/8/8/8/8/PPPPPPPP/RNNQKBBR w KQkq - 0 1",
    "rnkqnbbr/pppppppp/8/8/8/8/PPPPPPPP/RNKQNBBR w KQkq - 0 1",
    "rnkqrbbn/pppppppp/8/8/8/8/PPPPPPPP/RNKQRBBN w KQkq - 0 1",
    "rknqnbbr/pppppppp/8/8/8/8/PPPPPPPP/RKNQNBBR w KQkq - 0 1",
    "rknqrbbn/pppppppp/8/8/8/8/PPPPPPPP/RKNQRBBN w KQkq - 0 1",
    "rkrqnbbn/pppppppp/8/8/8/8/PPPPPPPP/RKRQNBBN w KQkq - 0 1",
    "nnrkqbbr/pppppppp/8/8/8/8/PPPPPPPP/NNRKQBBR w KQkq - 0 1",
    "nrnkqbbr/pppppppp/8/8/8/8/PPPPPPPP/NRNKQBBR w KQkq - 0 1",
    "nrknqbbr/pppppppp/8/8/8/8/PPPPPPPP/NRKNQBBR w KQkq - 0 1",
    "nrkrqbbn/pppppppp/8/8/8/8/PPPPPPPP/NRKRQBBN w KQkq - 0 1",
    "rnnkqbbr/pppppppp/8/8/8/8/PPPPPPPP/RNNKQBBR w KQkq - 0 1",
    "rnknqbbr/pppppppp/8/8/8/8/PPPPPPPP/RNKNQBBR w KQkq - 0 1",
    "rnkrqbbn/pppppppp/8/8/8/8/PPPPPPPP/RNKRQBBN w KQkq - 0 1",
    "rknnqbbr/pppppppp/8/8/8/8/PPPPPPPP/RKNNQBBR w KQkq - 0 1",
    "rknrqbbn/pppppppp/8/8/8/8/PPPPPPPP/RKNRQBBN w KQkq - 0 1",
    "rkrnqbbn/pppppppp/8/8/8/8/PPPPPPPP/RKRNQBBN w KQkq - 0 1",
    "nnrkrbbq/pppppppp/8/8/8/8/PPPPPPPP/NNRKRBBQ w KQkq - 0 1",
    "nrnkrbbq/pppppppp/8/8/8/8/PPPPPPPP/NRNKRBBQ w KQkq - 0 1",
    "nrknrbbq/pppppppp/8/8/8/8/PPPPPPPP/NRKNRBBQ w KQkq - 0 1",
    "nrkrnbbq/pppppppp/8/8/8/8/PPPPPPPP/NRKRNBBQ w KQkq - 0 1",
    "rnnkrbbq/pppppppp/8/8/8/8/PPPPPPPP/RNNKRBBQ w KQkq - 0 1",
    "rnknrbbq/pppppppp/8/8/8/8/PPPPPPPP/RNKNRBBQ w KQkq - 0 1",
    "rnkrnbbq/pppppppp/8/8/8/8/PPPPPPPP/RNKRNBBQ w KQkq - 0 1",
    "rknnrbbq/pppppppp/8/8/8/8/PPPPPPPP/RKNNRBBQ w KQkq - 0 1",
    "rknrnbbq/pppppppp/8/8/8/8/PPPPPPPP/RKNRNBBQ w KQkq - 0 1",
    "rkrnnbbq/pppppppp/8/8/8/8/PPPPPPPP/RKRNNBBQ w KQkq - 0 1",
    "qnnrkrbb/pppppppp/8/8/8/8/PPPPPPPP/QNNRKRBB w KQkq - 0 1",
    "qnrnkrbb/pppppppp/8/8/8/8/PPPPPPPP/QNRNKRBB w KQkq - 0 1",
    "qnrknrbb/pppppppp/8/8/8/8/PPPPPPPP/QNRKNRBB w KQkq - 0 1",
    "qnrkrnbb/pppppppp/8/8/8/8/PPPPPPPP/QNRKRNBB w KQkq - 0 1",
    "qrnnkrbb/pppppppp/8/8/8/8/PPPPPPPP/QRNNKRBB w KQkq - 0 1",
    "qrnknrbb/pppppppp/8/8/8/8/PPPPPPPP/QRNKNRBB w KQkq - 0 1",
    "qrnkrnbb/pppppppp/8/8/8/8/PPPPPPPP/QRNKRNBB w KQkq - 0 1",
    "qrknnrbb/pppppppp/8/8/8/8/PPPPPPPP/QRKNNRBB w KQkq - 0 1",
    "qrknrnbb/pppppppp/8/8/8/8/PPPPPPPP/QRKNRNBB w KQkq - 0 1",
    "qrkrnnbb/pppppppp/8/8/8/8/PPPPPPPP/QRKRNNBB w KQkq - 0 1",
    "nqnrkrbb/pppppppp/8/8/8/8/PPPPPPPP/NQNRKRBB w KQkq - 0 1",
    "nqrnkrbb/pppppppp/8/8/8/8/PPPPPPPP/NQRNKRBB w KQkq - 0 1",
    "nqrknrbb/pppppppp/8/8/8/8/PPPPPPPP/NQRKNRBB w KQkq - 0 1",
    "nqrkrnbb/pppppppp/8/8/8/8/PPPPPPPP/NQRKRNBB w KQkq - 0 1",
    "rqnnkrbb/pppppppp/8/8/8/8/PPPPPPPP/RQNNKRBB w KQkq - 0 1",
    "rqnknrbb/pppppppp/8/8/8/8/PPPPPPPP/RQNKNRBB w KQkq - 0 1",
    "rqnkrnbb/pppppppp/8/8/8/8/PPPPPPPP/RQNKRNBB w KQkq - 0 1",
    "rqknnrbb/pppppppp/8/8/8/8/PPPPPPPP/RQKNNRBB w KQkq - 0 1",
    "rqknrnbb/pppppppp/8/8/8/8/PPPPPPPP/RQKNRNBB w KQkq - 0 1",
    "rqkrnnbb/pppppppp/8/8/8/8/PPPPPPPP/RQKRNNBB w KQkq - 0 1",
    "nnqrkrbb/pppppppp/8/8/8/8/PPPPPPPP/NNQRKRBB w KQkq - 0 1",
    "nrqnkrbb/pppppppp/8/8/8/8/PPPPPPPP/NRQNKRBB w KQkq - 0 1",
    "nrqknrbb/pppppppp/8/8/8/8/PPPPPPPP/NRQKNRBB w KQkq - 0 1",
    "nrqkrnbb/pppppppp/8/8/8/8/PPPPPPPP/NRQKRNBB w KQkq - 0 1",
    "rnqnkrbb/pppppppp/8/8/8/8/PPPPPPPP/RNQNKRBB w KQkq - 0 1",
    "rnqknrbb/pppppppp/8/8/8/8/PPPPPPPP/RNQKNRBB w KQkq - 0 1",
    "rnqkrnbb/pppppppp/8/8/8/8/PPPPPPPP/RNQKRNBB w KQkq - 0 1",
    "rkqnnrbb/pppppppp/8/8/8/8/PPPPPPPP/RKQNNRBB w KQkq - 0 1",
    "rkqnrnbb/pppppppp/8/8/8/8/PPPPPPPP/RKQNRNBB w KQkq - 0 1",
    "rkqrnnbb/pppppppp/8/8/8/8/PPPPPPPP/RKQRNNBB w KQkq - 0 1",
    "nnrqkrbb/pppppppp/8/8/8/8/PPPPPPPP/NNRQKRBB w KQkq - 0 1",
    "nrnqkrbb/pppppppp/8/8/8/8/PPPPPPPP/NRNQKRBB w KQkq - 0 1",
    "nrkqnrbb/pppppppp/8/8/8/8/PPPPPPPP/NRKQNRBB w KQkq - 0 1",
    "nrkqrnbb/pppppppp/8/8/8/8/PPPPPPPP/NRKQRNBB w KQkq - 0 1",
    "rnnqkrbb/pppppppp/8/8/8/8/PPPPPPPP/RNNQKRBB w KQkq - 0 1",
    "rnkqnrbb/pppppppp/8/8/8/8/PPPPPPPP/RNKQNRBB w KQkq - 0 1",
    "rnkqrnbb/pppppppp/8/8/8/8/PPPPPPPP/RNKQRNBB w KQkq - 0 1",
    "rknqnrbb/pppppppp/8/8/8/8/PPPPPPPP/RKNQNRBB w KQkq - 0 1",
    "rknqrnbb/pppppppp/8/8/8/8/PPPPPPPP/RKNQRNBB w KQkq - 0 1",
    "rkrqnnbb/pppppppp/8/8/8/8/PPPPPPPP/RKRQNNBB w KQkq - 0 1",
    "nnrkqrbb/pppppppp/8/8/8/8/PPPPPPPP/NNRKQRBB w KQkq - 0 1",
    "nrnkqrbb/pppppppp/8/8/8/8/PPPPPPPP/NRNKQRBB w KQkq - 0 1",
    "nrknqrbb/pppppppp/8/8/8/8/PPPPPPPP/NRKNQRBB w KQkq - 0 1",
    "nrkrqnbb/pppppppp/8/8/8/8/PPPPPPPP/NRKRQNBB w KQkq - 0 1",
    "rnnkqrbb/pppppppp/8/8/8/8/PPPPPPPP/RNNKQRBB w KQkq - 0 1",
    "rnknqrbb/pppppppp/8/8/8/8/PPPPPPPP/RNKNQRBB w KQkq - 0 1",
    "rnkrqnbb/pppppppp/8/8/8/8/PPPPPPPP/RNKRQNBB w KQkq - 0 1",
    "rknnqrbb/pppppppp/8/8/8/8/PPPPPPPP/RKNNQRBB w KQkq - 0 1",
    "rknrqnbb/pppppppp/8/8/8/8/PPPPPPPP/RKNRQNBB w KQkq - 0 1",
    "rkrnqnbb/pppppppp/8/8/8/8/PPPPPPPP/RKRNQNBB w KQkq - 0 1",
    "nnrkrqbb/pppppppp/8/8/8/8/PPPPPPPP/NNRKRQBB w KQkq - 0 1",
    "nrnkrqbb/pppppppp/8/8/8/8/PPPPPPPP/NRNKRQBB w KQkq - 0 1",
    "nrknrqbb/pppppppp/8/8/8/8/PPPPPPPP/NRKNRQBB w KQkq - 0 1",
    "nrkrnqbb/pppppppp/8/8/8/8/PPPPPPPP/NRKRNQBB w KQkq - 0 1",
    "rnnkrqbb/pppppppp/8/8/8/8/PPPPPPPP/RNNKRQBB w KQkq - 0 1",
    "rnknrqbb/pppppppp/8/8/8/8/PPPPPPPP/RNKNRQBB w KQkq - 0 1",
    "rnkrnqbb/pppppppp/8/8/8/8/PPPPPPPP/RNKRNQBB w KQkq - 0 1",
    "rknnrqbb/pppppppp/8/8/8/8/PPPPPPPP/RKNNRQBB w KQkq - 0 1",
    "rknrnqbb/pppppppp/8/8/8/8/PPPPPPPP/RKNRNQBB w KQkq - 0 1",
    "rkrnnqbb/pppppppp/8/8/8/8/PPPPPPPP/RKRNNQBB w KQkq - 0 1"
]

function getFenForChess960Pos(posNum) {
    const standardFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    if (posNum === 518) return standardFen; // Common convention for standard position
    if (posNum >= 0 && posNum < chess960Fens.length && chess960Fens[posNum] && chess960Fens[posNum].trim() !== '') {
        return chess960Fens[posNum];
    }
    console.warn(`FEN for Chess960 position ${posNum} not found or invalid. Using standard. Max known index ${chess960Fens.length - 1}.`);
    return standardFen;
}

function requestPlayerPiecesBestMoves(fen) {
    if (!engine || !helpersVisible) {
        if (currentPlayerPieceBestEvals.evals.size > 0) {
            currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; requestRedraw();
        }
        return;
    }
    const tempGame = new Chess(fen); tempGame.chess960 = true;
    if (tempGame.game_over()) {
        if (currentPlayerPieceBestEvals.evals.size > 0) {
            currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; requestRedraw();
        }
        return;
    }
    const turn = tempGame.turn(); let allUci = []; const board = tempGame.board();
    for (let r=0; r<8; r++) for (let f=0; f<8; f++) {
        const sqInfo = board[r][f];
        if (sqInfo && sqInfo.color === turn) {
            const pSq = String.fromCharCode('a'.charCodeAt(0)+f)+(8-r);
            const moves = tempGame.moves({square:pSq, verbose:true});
            allUci.push(...moves.map(m=>m.from+m.to+(m.promotion?m.promotion:'')));
        }
    }
    if (allUci.length > 0) {
        if (analysisManager.currentAnalysisType === 'player_pieces_best_moves' && analysisManager.lastPlayerPiecesBestMovesFen === fen && currentPlayerPieceBestEvals.fen === fen) {
            return; // Already processing or data is fresh
        }
        requestAnalysis(fen, 'player_pieces_best_moves', { searchMovesUci: allUci });
        statusMessage("Evaluating piece outlines...");
    } else {
        if (currentPlayerPieceBestEvals.evals.size > 0) { currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; requestRedraw(); }
    }
}

// --- Saving PGN ---
function saveGameHistory() {
    if (!currentNode || getNodePath(currentNode).length <= 1) return;
    const path = getNodePath(currentNode);
    const header = { Event: "Chess960 Analysis (Web)", Site: "Local Browser", Date: new Date().toISOString().split('T')[0].replace(/-/g,'.'), Round: "-", White: whiteNameInput.value || "White", Black: blackNameInput.value || "Black", Result: game.game_over() ? getGameResultString() : "*", FEN: path[0].fen, SetUp: "1", Variant: "Chess960" };
    const tempGame = new Chess(path[0].fen); tempGame.chess960 = true;
    for (let i=1; i<path.length; i++) { if(path[i].move) tempGame.move(path[i].move); else break; }
    for (const k in header) tempGame.header(k, header[k]);
    const pgn = tempGame.pgn({ maxWidth: 80, newline_char: '\n' });
    try {
        const blob = new Blob([pgn],{type:'text/plain;charset=utf-8'});
        const fname = `game_${startTime.toISOString().replace(/[:\-T]/g,'').substring(0,15)}_pos${currentBoardNumber}.pgn`;
        const link = document.createElement("a"); link.href=URL.createObjectURL(blob); link.download=fname;
        document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(link.href);
        statusMessage("Game history saved as PGN.");
    } catch(e) { statusMessage("Error saving game."); }
}
function getGameResultString() {
    if (game.in_checkmate()) return game.turn() === 'b' ? '1-0' : '0-1';
    if (game.in_stalemate() || game.in_threefold_repetition() || game.insufficient_material() || game.in_draw()) return '1/2-1/2'; return "*";
}

// --- Event Listeners Setup ---
function setupEventListeners() {
    showBestMoveBtn.addEventListener('click', handleShowBestMoveClick);
    toggleHelpersBtn.addEventListener('click', handleToggleHelpersClick);
    saveGameBtn.addEventListener('click', saveGameHistory);
    explainPlanBtn.addEventListener('click', handleExplainPlanClick);
    document.addEventListener('keydown', handleKeyDown);
    setupTreeDrag(); window.addEventListener('resize', handleResize);
}
function handleShowBestMoveClick() {
    if (!engine || !currentNode || game.game_over()) { statusMessage("Engine unavailable or game over."); return; }
    const fen = currentNode.fen;
    if (currentBestMovesResult && currentBestMovesResult.fen === fen && currentBestMovesResult.moves?.length > 0) {
        const moves = currentBestMovesResult.moves;
        currentBestMoveIndex = (currentBestMoveIndex + 1) % moves.length;
        highlightedEngineMove = uciToMoveObject(moves[currentBestMoveIndex].move);
        statusMessage(`Best ${currentBestMoveIndex+1}/${moves.length}: ${moves[currentBestMoveIndex].san} (${moves[currentBestMoveIndex].score_str})`);
        requestRedraw();
    } else {
        statusMessage("Analyzing for best moves..."); engineStatusMessage("");
        highlightedEngineMove = null; currentBestMoveIndex = -1; currentBestMovesResult = null; updateShowBestButtonText("Analyzing...");
        requestAnalysis(fen, 'best_moves');
    }
    updateShowBestButtonText();
}
function handleToggleHelpersClick() {
    helpersVisible = !helpersVisible;
    toggleHelpersBtn.textContent = `Helpers: ${helpersVisible ? 'ON' : 'OFF'}`;
    toggleHelpersBtn.classList.toggle('active', helpersVisible);
    statusMessage(`Helpers ${helpersVisible ? 'ON' : 'OFF'}`);
    treeLayout.needsRedraw = true; // For tree badges

    if (!helpersVisible) {
        selectedPieceEvaluations = null;
        currentPlayerPieceBestEvals = { fen: null, evals: new Map() }; // Clear outlines
        if (analysisManager.currentAnalysisType === 'player_pieces_best_moves') {
            if(engine) engine.postMessage('stop');
            analysisManager.isProcessing = false; analysisManager.currentAnalysisType = null;
            if (analysisManager.requestQueue.length === 0 && currentNode && engine) {
                requestAnalysis(currentNode.fen, 'continuous'); processAnalysisQueue();
            }
        }
    } else { // Helpers ON
        if (selectedSquare) { // Re-eval selected piece moves if any
            const p = game.get(selectedSquare); const turn = game.turn();
            if (p && p.color === turn && legalMovesForSelected.length > 0) {
                const uci = legalMovesForSelected.map(m=>m.from+m.to+(m.promotion?m.promotion:''));
                requestAnalysis(game.fen(), 'selected_piece_moves', {searchMovesUci:uci, pieceSq:selectedSquare});
            }
        }
        requestPlayerPiecesBestMoves(game.fen()); // Request outlines for all pieces
    }
    requestRedraw(); // Redraw to apply/remove highlights, overlays, outlines
}
function handleClickOutside(event) {
    if (promotionPopup && !promotionPopup.contains(event.target) && promotionState.pending &&
        !event.target.closest('.promo-choice') && event.target !== promotionPopup) {
        cancelPromotion();
    }
}
function cancelPromotion() { if (promotionState.pending) { hidePromotionPopup(); statusMessage("Promotion cancelled."); clearSelection(); }}
function handleKeyDown(event) {
    if (document.activeElement === blackNameInput || document.activeElement === whiteNameInput) return;
    if (promotionState.pending) { if (event.key === 'Escape') cancelPromotion(); return; }
    let changed=false; let targetN=null;
    switch(event.key) {
        case 'ArrowLeft': if(currentNode?.parent){targetN=currentNode.parent;changed=true;} break;
        case 'ArrowRight': if(currentNode?.children.length>0){targetN=currentNode.children[0];changed=true;} break;
        case 'ArrowUp': if(currentNode?.parent){const S=currentNode.parent.children;const i=S.indexOf(currentNode);if(i>0){targetN=S[i-1];changed=true;}} break;
        case 'ArrowDown': if(currentNode?.parent){const S=currentNode.parent.children;const i=S.indexOf(currentNode);if(i<S.length-1){targetN=S[i+1];changed=true;}} break;
        case 'a': handleShowBestMoveClick(); event.preventDefault(); break;
        case 'h': handleToggleHelpersClick(); event.preventDefault(); break;
        case 's': saveGameHistory(); event.preventDefault(); break;
        case 'm': plotVisible=!plotVisible; document.getElementById('plot-panel').style.display=plotVisible?'block':'none'; if(plotVisible)updatePlot(); event.preventDefault(); break;
        case 'Escape': clearSelection(); statusMessage("Selection cleared"); break;
    }
    if (changed && targetN) {
        const elInfo = treeNodeElements.get(targetN.id);
        if(elInfo?.group) handleTreeNodeClick({currentTarget: elInfo.group});
        else { /* Fallback if SVG element not found, less ideal direct update */
            currentNode = targetN;
            game = new Chess(currentNode.fen); game.chess960 = true;
            liveRawScore = currentNode.raw_score; targetWhitePercentage = scoreToWhitePercentage(liveRawScore, EVAL_CLAMP_LIMIT);
            requestRedraw();
        }
        scrollToNode(targetN); event.preventDefault();
    }
}
function handleResize() { initBoard(); treeLayout.needsRedraw = true; if(currentEvalPlot){currentEvalPlot.resize();updatePlot();} requestRedraw(); }

async function handleExplainPlanClick() {
    if (!currentBestMovesResult || currentBestMovesResult.fen !== currentNode.fen || currentBestMoveIndex === -1) {
        statusMessage("Please select a best move first using 'Show Best' to get a plan explanation.");
        return;
    }

    const selectedMoveData = currentBestMovesResult.moves[currentBestMoveIndex];
    if (!selectedMoveData || !selectedMoveData.pv_sequence || selectedMoveData.pv_sequence.length === 0) {
        statusMessage("No move sequence (plan) available for the selected move.");
        return;
    }

    const fen = currentBestMovesResult.fen;
    const move_sequence = selectedMoveData.pv_sequence; // This was added in the previous step

    statusMessage("Asking AI for explanation of the plan...");
    engineStatusMessage(""); // Clear engine status

    try {
        const response = await fetch('http://localhost:5000/explain_moves', { // Assuming Flask runs on port 5000
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ fen: fen, move_sequence: move_sequence }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown error from server" }));
            let errorMsg = `Error from explanation server: ${response.status}`;
            if (errorData && errorData.detail) {
                errorMsg += ` - ${errorData.detail}`;
            } else if (response.statusText) {
                 errorMsg += ` - ${response.statusText}`;
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();
        if (data.explanation) {
            // For now, display in statusMessage. A modal or dedicated panel could be a future improvement.
            statusMessage(`AI Explanation: ${data.explanation}`);
        } else {
            statusMessage("Received an empty explanation from AI.");
        }

    } catch (error) {
        console.error('Error fetching explanation:', error);
        statusMessage(`Failed to get explanation: ${error.message}`);
    }
}