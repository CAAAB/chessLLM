document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const STOCKFISH_JS_PATH = 'libs/stockfish.js'; // Adjust if needed
    const ANALYSIS_DEPTH = 15; // Deeper = Slower but stronger
    const EVAL_CLAMP_CP = 1000; // Centipawns value corresponding to 100% / 0%

    // --- Globals ---
    let game = new Chess();
    let board = null;
    let stockfish = null;
    let evalChart = null;
    let pgnMoves = []; // Array of moves from the loaded PGN {san: 'e4', fen_before: '...', fen_after: '...'}
    let moveHistoryEvaluations = []; // Array of {ply: N, score: {type:'cp', value: V} | {type:'mate', value: M}}
    let currentMoveIndex = -1; // -1 means starting position
    let isAnalyzingHistory = false;
    let analysisQueue = [];
    let currentAnalysisPromise = null;

    // --- DOM Elements ---
    const pgnInput = document.getElementById('pgnInput');
    const loadPgnBtn = document.getElementById('loadPgnBtn');
    const analysisArea = document.getElementById('analysisArea');
    const boardEl = document.getElementById('board');
    const startBtn = document.getElementById('startBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const endBtn = document.getElementById('endBtn');
    const moveCounterEl = document.getElementById('moveCounter');
    const statusEl = document.getElementById('status');
    const evaluationEl = document.getElementById('evaluation');
    const bestMoveEl = document.getElementById('bestMove');
    const depthEl = document.getElementById('depth');
    const chartCanvas = document.getElementById('evalChart').getContext('2d');
    const errorArea = document.getElementById('errorArea');

    // --- Initialization ---
    function initStockfish() {
        updateStatus("Loading Stockfish engine...");
        try {
            // Make sure stockfish.js is accessible. Manifest V3 might require specific worker creation methods.
            stockfish = new Worker(STOCKFISH_JS_PATH);
            stockfish.postMessage('uci'); // Start UCI protocol
            stockfish.postMessage('isready');

            stockfish.onmessage = (event) => {
                const message = event.data;
                // console.log("SF:", message); // Debugging

                if (message === 'readyok') {
                    updateStatus("Stockfish Ready");
                    stockfish.postMessage('setoption name UCI_Chess960 value false'); // Standard chess
                    stockfish.postMessage(`setoption name Skill Level value 20`); // Max strength
                     stockfish.postMessage(`setoption name Threads value 1`); // Use 1 thread for stability in worker
                     stockfish.postMessage(`setoption name Hash value 16`); // Small hash table
                    console.log("Stockfish initialized successfully.");
                    if (currentAnalysisPromise && currentAnalysisPromise.resolveReady) {
                         currentAnalysisPromise.resolveReady(); // Signal readiness if waiting
                    }
                } else if (message.startsWith('info')) {
                    parseStockfishInfo(message);
                } else if (message.startsWith('bestmove')) {
                    parseStockfishBestmove(message);
                }
            };
            stockfish.onerror = (error) => {
                 console.error("Stockfish Worker Error:", error);
                 updateStatus("Stockfish Worker Error!");
                 setError(`Stockfish worker error: ${error.message}`);
                 if (currentAnalysisPromise && currentAnalysisPromise.reject) {
                      currentAnalysisPromise.reject(error); // Reject pending analysis
                 }
            };

        } catch (err) {
            console.error("Failed to initialize Stockfish:", err);
            updateStatus("Failed to load Stockfish!");
            setError("Could not load Stockfish. Check console for details.");
            stockfish = null; // Ensure stockfish is null if failed
        }
    }

    function initBoard() {
        const config = {
            draggable: false, // Read-only board
            position: 'start',
            pieceTheme: 'libs/img/chesspieces/wikipedia/{piece}.png' // Default theme, requires images folder from chessboard.js download
        };
        board = Chessboard('board', config); // Use the ID of the div
    }

    function initChart() {
        evalChart = new Chart(chartCanvas, {
            type: 'line', // Use line chart type
            data: {
                labels: [], // Ply numbers
                datasets: [{
                    label: 'Evaluation (%)',
                    data: [], // Evaluation percentages
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)', // Fill color
                    tension: 0.1,
                    fill: true // Enable area fill
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,   // Percentage scale
                        max: 100,
                        ticks: {
                           callback: function(value) {
                               return value + '%'
                           }
                        }
                    },
                    x: {
                       title: {
                            display: true,
                            text: 'Ply'
                        }
                    }
                },
                 plugins: {
                    legend: {
                        display: false // Hide legend if desired
                    },
                    tooltip: {
                         callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    // Find original score for tooltip
                                    const ply = context.parsed.x; // ply index
                                    const evalData = moveHistoryEvaluations[ply];
                                    let scoreText = 'N/A';
                                    if(evalData && evalData.score){
                                        scoreText = formatEvaluation(evalData.score, false);
                                    }
                                    label += `${context.parsed.y.toFixed(1)}% (${scoreText})`;
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    // --- UI Update Functions ---
    function updateStatus(text) {
        statusEl.textContent = text;
    }
     function setError(text) {
        errorArea.textContent = text;
    }

    function updateEvaluationDisplay(score, depth = null) {
        evaluationEl.textContent = formatEvaluation(score);
         if(depth) depthEl.textContent = depth;
    }

    function updateBestMoveDisplay(move) {
        bestMoveEl.textContent = move || 'N/A';
    }

    function updateBoardPosition() {
        const fen = game.fen();
        if (board) {
            board.position(fen, false); // Update board without animation
        }
        requestAnalysisForCurrentPosition(); // Analyze the new position
    }

     function updateMoveCounter() {
        const currentPly = currentMoveIndex + 1;
        const totalPly = pgnMoves.length;
        moveCounterEl.textContent = `Move: ${Math.ceil(currentPly / 2)} (${currentPly}/${totalPly})`;
        // Update button states
        prevBtn.disabled = currentPly <= 0;
        startBtn.disabled = currentPly <= 0;
        nextBtn.disabled = currentPly >= totalPly;
        endBtn.disabled = currentPly >= totalPly;
    }

    function updateChart() {
        if (!evalChart) return;

        const labels = moveHistoryEvaluations.map(e => e.ply);
        const data = moveHistoryEvaluations.map(e => evaluationToPercentage(e.score));

        evalChart.data.labels = labels;
        evalChart.data.datasets[0].data = data;
        evalChart.update();
    }

    // --- PGN Handling ---
    function loadPgn() {
        setError(""); // Clear previous errors
        const pgnString = pgnInput.value.trim();
        if (!pgnString) {
             setError("PGN input is empty.");
             return;
        }

        const loaded = game.load_pgn(pgnString, { sloppy: true }); // Allow slightly malformed PGNs
        if (!loaded) {
            setError("Invalid PGN data.");
            resetAnalysisState();
            return;
        }

        console.log("PGN Loaded Successfully");
        resetAnalysisState(); // Clear previous game state/analysis

        // Extract moves with FEN before/after each move
        const history = game.history({ verbose: true });
        const tempGame = new Chess(); // Use a temporary game to get FENs
        pgnMoves = history.map(move => {
            const fen_before = tempGame.fen();
            tempGame.move(move.san); // Make the move
            const fen_after = tempGame.fen();
            return { ...move, fen_before, fen_after };
        });

        game.reset(); // Reset main game instance to the start
        currentMoveIndex = -1;
        analysisArea.style.display = 'flex'; // Show analysis section

        initBoard(); // Re-initialize board if needed, or just set position
        board.position(game.fen());
        updateMoveCounter();
        requestAnalysisForCurrentPosition(); // Analyze starting position
        startBackgroundAnalysis(); // Start analyzing the whole game for the plot
    }

     function resetAnalysisState() {
        game.reset();
        if(board) board.position('start');
        pgnMoves = [];
        moveHistoryEvaluations = [];
        currentMoveIndex = -1;
        isAnalyzingHistory = false;
        analysisQueue = [];
        updateStatus("Idle");
        updateEvaluationDisplay({ type: 'cp', value: 0 }); // Reset eval
        updateBestMoveDisplay('N/A');
        depthEl.textContent = 'N/A';
        if (evalChart) {
            evalChart.data.labels = [];
            evalChart.data.datasets[0].data = [];
            evalChart.update();
        }
         updateMoveCounter();
    }

    // --- Navigation ---
    function navigateToMove(index) {
        if (index < -1 || index >= pgnMoves.length) return; // Invalid index

        // Reset game and apply moves up to the target index
        game.reset();
        for (let i = 0; i <= index; i++) {
            game.move(pgnMoves[i].san);
        }
        currentMoveIndex = index;
        updateBoardPosition();
        updateMoveCounter();
    }

    // --- Stockfish Interaction ---
    function requestAnalysis(fen, isHistoryAnalysis = false, depth = ANALYSIS_DEPTH) {
         if (!stockfish || !fen) return;

        // If currently analyzing history, queue requests unless this IS a history request
        if (isAnalyzingHistory && !isHistoryAnalysis) {
            // Prioritize user navigation analysis - interrupt history? Or just run after?
            // Simple approach: Just run it now, potentially delaying history.
             console.log("Prioritizing user navigation analysis for FEN:", fen);
        } else if (isHistoryAnalysis) {
            // Add to queue if history analysis is running
             analysisQueue.push({ fen, depth });
            // console.log("Queueing history analysis for FEN:", fen, "Queue size:", analysisQueue.length);
            return; // Don't start if already processing queue
        }


         // Setup a promise to track this specific analysis request
        let resolveFn, rejectFn;
        const promise = new Promise((resolve, reject) => {
             resolveFn = resolve;
             rejectFn = reject;
        });
        // Attach resolve/reject to the promise object itself for access in callbacks
        promise.resolve = resolveFn;
        promise.reject = rejectFn;
        // If Stockfish isn't ready yet, add a ready promise part
        promise.resolveReady = null;
        const readyPromise = new Promise(resolve => { promise.resolveReady = resolve; });


        // Clear previous potentially unfinished analysis info
        updateEvaluationDisplay({ type: 'cp', value: 0}); // Tentative reset
        updateBestMoveDisplay('Thinking...');
        depthEl.textContent = '0';


        // Store the promise to manage callbacks correctly
        currentAnalysisPromise = promise;


        const sendGoCommand = () => {
             updateStatus(`Analyzing (Depth ${depth})...`);
             stockfish.postMessage(`position fen ${fen}`);
             stockfish.postMessage(`go depth ${depth}`);
        };

        // Check if stockfish is ready before sending commands
        stockfish.postMessage('isready'); // Ask again to ensure readiness state is current
        readyPromise.then(sendGoCommand).catch(err => {
            console.error("Error during analysis readiness check:", err);
             setError("Error communicating with Stockfish.");
             if(currentAnalysisPromise && currentAnalysisPromise.reject) currentAnalysisPromise.reject(err);
        });


        return promise; // Return promise for history analysis coordination
    }

     function requestAnalysisForCurrentPosition() {
         const fen = game.fen();
         // Don't analyze if already analyzing history, user nav takes priority (handled in requestAnalysis)
         requestAnalysis(fen, false, ANALYSIS_DEPTH); // false = not part of the batch history analysis
    }

    function parseStockfishInfo(message) {
         if (!currentAnalysisPromise) return; // Ignore info if no analysis is active

        const parts = message.split(' ');
        let score = null;
        let currentDepth = null;

        for (let i = 0; i < parts.length; i++) {
            if (parts[i] === 'score' && parts[i + 1] === 'cp' && parts.length > i + 2) {
                score = { type: 'cp', value: parseInt(parts[i + 2], 10) };
                 // Adjust score based on whose turn it is (Stockfish reports from white's perspective)
                 if (game.turn() === 'b') {
                     score.value *= -1;
                 }
            } else if (parts[i] === 'score' && parts[i + 1] === 'mate' && parts.length > i + 2) {
                score = { type: 'mate', value: parseInt(parts[i + 2], 10) };
                 if (game.turn() === 'b') {
                     score.value *= -1;
                 }
            } else if (parts[i] === 'depth' && parts.length > i + 1) {
                 currentDepth = parseInt(parts[i+1], 10);
            } else if (parts[i] === 'pv' && parts.length > i + 1) {
                // Could potentially update best move display progressively here
                // bestMoveEl.textContent = parts[i+1];
            }
        }

        // Update evaluation display progressively if score is found
         if (score !== null) {
            updateEvaluationDisplay(score, currentDepth); // Update eval and depth together
            currentAnalysisPromise.currentScore = score; // Store latest score on the promise
         } else if (currentDepth !== null) {
             depthEl.textContent = currentDepth; // Update depth even if score hasn't changed
         }
    }

    function parseStockfishBestmove(message) {
        if (!currentAnalysisPromise) {
            console.warn("Received bestmove but no analysis promise active.");
            return; // Ignore if no analysis was requested or it was already resolved/rejected
        }

        const parts = message.split(' ');
        const bestMove = parts[1];
        // parts[3] might be the ponder move

        updateStatus("Analysis Complete");
        updateBestMoveDisplay(bestMove); // Show final best move

        // Resolve the promise with the final evaluation stored during 'info' messages
        const finalScore = currentAnalysisPromise.currentScore || { type: 'cp', value: 0 }; // Use last known score
         // Make sure evaluation is updated one last time with the final score
         updateEvaluationDisplay(finalScore, depthEl.textContent); // Keep last depth shown

        currentAnalysisPromise.resolve({ bestMove: bestMove, score: finalScore });
        currentAnalysisPromise = null; // Clear the promise

        // If this was a history analysis, trigger the next one in the queue
        if (isAnalyzingHistory) {
             processAnalysisQueue();
        }
    }

     // --- Background Analysis for Plot ---
    async function startBackgroundAnalysis() {
        if (isAnalyzingHistory || pgnMoves.length === 0 || !stockfish) return; // Don't start if already running or no moves/engine

        console.log("Starting background analysis for evaluation history...");
        isAnalyzingHistory = true;
        moveHistoryEvaluations = []; // Clear previous results
        analysisQueue = []; // Clear queue

        updateStatus("Analyzing game history...");

        // Queue analysis for position AFTER each move
        const tempGame = new Chess(); // Use a separate instance for FEN generation
        for (let i = 0; i < pgnMoves.length; i++) {
             const move = pgnMoves[i];
             tempGame.move(move.san); // Make move to get FEN *after* the move
             const fenAfterMove = tempGame.fen();
             analysisQueue.push({ fen: fenAfterMove, depth: ANALYSIS_DEPTH, ply: i + 1 }); // Add ply number
        }

        // Reset temp game if needed later
        tempGame.reset();

        // Start processing the queue
        processAnalysisQueue();
    }

    async function processAnalysisQueue() {
         if (analysisQueue.length === 0) {
            console.log("Background analysis complete.");
            isAnalyzingHistory = false;
            updateStatus("History analysis complete.");
            updateChart(); // Update chart with all data
            // Re-analyze current position if user navigated during history analysis
            requestAnalysisForCurrentPosition();
            return;
        }

        if(currentAnalysisPromise) {
            // console.log("Waiting for current analysis to finish before starting next queue item.");
            return; // Wait for the current analysis (likely user-triggered) to finish
        }


        const nextJob = analysisQueue.shift(); // Get the next job
        updateStatus(`Analyzing history (Ply ${nextJob.ply}/${pgnMoves.length})...`);
        // console.log(`Analyzing Ply ${nextJob.ply}: FEN ${nextJob.fen}`);


         try {
            // requestAnalysis returns a promise tracking this specific analysis
             const result = await requestAnalysis(nextJob.fen, true, nextJob.depth); // true = history analysis
              if(result && result.score) {
                moveHistoryEvaluations.push({ ply: nextJob.ply, score: result.score });
                moveHistoryEvaluations.sort((a, b) => a.ply - b.ply); // Keep sorted just in case
                updateChart(); // Update chart progressively
            } else {
                 console.warn(`Analysis for ply ${nextJob.ply} did not return a score.`);
                 // Optionally push a placeholder or skip
                 moveHistoryEvaluations.push({ ply: nextJob.ply, score: {type: 'cp', value: 0 }}); // Placeholder
            }
            // processAnalysisQueue() will be called automatically by parseStockfishBestmove
            // when the analysis completes IF isAnalyzingHistory is true.

        } catch (error) {
            console.error(`Error analyzing ply ${nextJob.ply}:`, error);
            setError(`Error during background analysis at ply ${nextJob.ply}.`);
            // Decide whether to stop or continue
             // For now, let's try to continue
             // Push a placeholder for the failed analysis
              moveHistoryEvaluations.push({ ply: nextJob.ply, score: {type: 'cp', value: 0 }});
              moveHistoryEvaluations.sort((a, b) => a.ply - b.ply);
              updateChart();
              currentAnalysisPromise = null; // Ensure promise is cleared on error too
              setTimeout(processAnalysisQueue, 50); // Add small delay before trying next
        }
    }


    // --- Evaluation Formatting & Conversion ---
    function formatEvaluation(score, usePercent = false) {
        if (!score) return 'N/A';
        if (score.type === 'cp') {
            const evalNum = (score.value / 100.0).toFixed(1);
            return `${evalNum > 0 ? '+' : ''}${evalNum}`;
        } else if (score.type === 'mate') {
            return `Mate in ${score.value}`;
        }
        return 'N/A';
    }

    function evaluationToPercentage(score) {
        if (!score) return 50; // Default to 50% if no score

        if (score.type === 'mate') {
            return score.value > 0 ? 100 : 0;
        } else if (score.type === 'cp') {
            const cp = score.value;
            // Clamp the score
            const clampedCp = Math.max(-EVAL_CLAMP_CP, Math.min(EVAL_CLAMP_CP, cp));
            // Scale linearly from [-CLAMP, +CLAMP] to [0, 100]
            const percentage = ((clampedCp + EVAL_CLAMP_CP) / (2 * EVAL_CLAMP_CP)) * 100;
            return percentage;
        }
        return 50; // Fallback
    }


    // --- Event Listeners ---
    loadPgnBtn.addEventListener('click', loadPgn);
    startBtn.addEventListener('click', () => navigateToMove(-1));
    prevBtn.addEventListener('click', () => navigateToMove(currentMoveIndex - 1));
    nextBtn.addEventListener('click', () => navigateToMove(currentMoveIndex + 1));
    endBtn.addEventListener('click', () => navigateToMove(pgnMoves.length - 1));

    // --- Initial Setup ---
    initChart();
    initStockfish(); // Start loading the engine immediately
     analysisArea.style.display = 'none'; // Hide until PGN loaded

});