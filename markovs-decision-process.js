// define variables

// define the grid dimensions of |S| = (i,j) with rows, columns;
// define the semantic type of any state with a string label: "empty", "start", "goal", "obstacle", "path";
let rows, columns, board = [];

// define deliveryReward as a fixed-value reward for a goal state, i.e. absorbing state immutable value reward;
// define powerCost as the negative reward: transitionary immediate, for any transition from non-terminal state;
// define repairCost as a negative reward: immutable; repair == rival == obstacle;
// define discount factor (gamma) by denoting how much agent values future rewards relative to immediate;
let deliveryReward, powerCost, repairCost, discount;

// define contrast as parameter for visualization normalization exponent;
// define directions as an action space, A, cardinality (4, maybe 8 with grater resolution);
let contrast, directions, values = [];

// define (i,j) pairs of start and goal node;
// define in an array all pairs (i,j) representing obstacles;
let startNode, goalNode, rivals = [];

let isMouseDown, mode, gradientActive = false, lastCompressedValues = [];

// event listeners to grid dimensions changes;
document.getElementById("rows").addEventListener("input", update);
document.getElementById("columns").addEventListener("input", update);


function setup() {
    // call update to read all input fields into the global variables;
    update();
    board[1][1] = "start";
    startNode = [1,1]; // we put a start node
    board[rows - 2][columns - 2] = "goal";
    goalNode = [rows - 2][columns - 2]; // we put an end node
    createBoard(); // we generate the board

    // disable the gradient button until a path has been computed;
    let gradient = document.getElementById("toggleGradient");
    gradient.disabled = true;
}

window.onload = setup;

/* Reads in scalar HTML param inputs and regenerates grid */
function update() {
    rows = parseInt(document.getElementById("rows").value);
    columns = parseInt(document.getElementById("columns").value);
    deliveryReward = parseFloat(document.getElementById("deliveryReward").value);
    powerCost = parseFloat(document.getElementById("powerCost").value);
    repairCost = parseFloat(document.getElementById("repairCost").value);
    discount = parseFloat(document.getElementById("discount").value);
    contrast = parseInt(document.getElementById("contrast").value);
    directions = parseInt(document.querySelector('input[name="directionMode"]:checked').value);
    createBoard();
}

function setupEvents(cell) {
    cell.addEventListener("mousedown", (event) => {
        isMouseDown = true;
        draw(event); // paint start/end/obstacle
    });
    cell.addEventListener("mouseup", () => {
        isMouseDown = false;
    })
    cell.addEventListener("mouseover", (event) => {
        if (isMouseDown)draw(event);
    })
}

/* Linear search brute force O(|S|) to locate start/goal node in board array, where |S| is node number */
function indexElement(arr, target) { // used for searching for start/goal nodes
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr[i].length; j++) {
            if (arr[i][j] === target)
                return {i, j}; // return the index we are searching for as an object
        }
    }
    return null; // if we can't find the node that matches
}

function createBoard() {

    const gridContainer = document.querySelector(".grid-container");
    gridContainer.replaceChildren();

    // store previous state labels
    const oldBoard = board;

    // fill with state label "empty", a state-reset rows x columns matrix
    board = new Array(rows).fill(0).map(() => new Array(columns).fill("empty"));

    // create a new div element for each (i,j)
    for (let i = 0; i < rows; i++) {
        const row = document.createElement("div");
        row.classList.add("row"); // creating a div with class "row"
        for (let j = 0; j < columns; j++) {
            const cell = document.createElement("div");
            cell.classList.add("cell");
            cell.dataset.row = i;
            cell.dataset.col = j;

            setupEvents(cell);

            // copy over a label from oldBoard if it existed
            if (oldBoard[i] && oldBoard[i][j]) {
                cell.classList.add(oldBoard[i][j]);
                board[i][j] = oldBoard[i][j]; // we preserve the existing cell color if it is available
            }
            row.appendChild(cell);
        }
        gridContainer.appendChild(row);
    }
}

function selectMode(newMode) { // used mainly to change UI, to see which mode is active by darkening the button
    mode = newMode; // set the active mode from the html
    const drawModeButtons = document.querySelectorAll(".button.draw-mode"); // grabs all the buttons with these classes

    drawModeButtons.forEach((drawModeButton) => {
        if (drawModeButton.value === mode) drawModeButton.classList.add("active"); // adds the active class used for coloring if selected
        else
            drawModeButton.classList.remove("active"); // removes the active class
    });
}

function draw(event) {

    const cell = event.target;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);

    if (mode === "obstacle") {
        if (cell.classList.contains("obstacle")) {
            cell.classList.remove("obstacle");
            board[row][col] = "empty";
        }
        else {
            cell.classList.add("obstacle");
            board[row][col] = "obstacle";
        }
    } else if (mode === "startNode") {
        let index = indexElement(board, "start");
        clearDuplicates(index);
        cell.classList.add("start");
        board[row][col] = "start";
    } else if (mode === "goalNode") {
        let index = indexElement(board, "goal");
        clearDuplicates(index);
        cell.classList.add("goal");
        board[row][col] = "goal";
    }

}

function clearDuplicates(index) {
    if (index) {
        const selector = `.cell[data-row="${index["i"]}"][data-col="${index["j"]}"]`;
        const oldCell = document.querySelector(selector);
        oldCell.className = "cell";
        board[index["i"]][index["j"]] = "empty";
    }
}


function togglePath() {

    update();
    toggleButtons(); // buttons are being turned off so that it doesn't regenerate at every swap of values if path is generated

    const generatedPath = document.getElementById("togglePath");

    generatedPath.classList.toggle("active");

    const path = document.querySelectorAll(".cell.path"); // select all cells with the .path class
    path.forEach((cell) => {
        cell.classList.remove("path"); // remove the .path class from each cell
    });

    if (generatedPath.classList.contains("active")) {
        planPath();
    } else {
        if (gradientActive) toggleGradient(); // disabling the gradient if it's on and we toggled the path off
    }
}

//!!!
/**
 * Applies Anderson Acceleration to speed up fixed-point iterations (like Value Iteration).
 * Pure JS standalone method.
 *
 * @param {Array} vHistory - Array of previous 2D value matrices V_k
 * @param {Array} fHistory - Array of previous 2D Bellman updates F(V_k)
 * @param {number} rows - Number of rows in the grid
 * @param {number} cols - Number of columns in the grid
 * @returns {Array} - The accelerated 2D value matrix for the next iteration
 */
function applyAndersonAcceleration(vHistory, fHistory, rows, cols) {

    // m = |{V}| i.e. the cardinality of the history of past iterations
    const m = vHistory.length;
    if (m === 0) return []; // empty set
    if (m === 1) return fHistory[0]; // F(V_1): Not enough history, fall back to standard Bellman update; F(V_k) is
    // notation for Bellman update at iteration k;
    // Explanation: Anderson acceleration requires at least 2 pts to draw a line; for 1 data pt, we default to standard iteration.


    // Helper: Flatten 2D matrix to 1D vector
    // Vectorization operator vec: R^{rows * cols} -> R^N
    const flatten = (matrix) => matrix.flat();

    // Helper: Unflatten 1D vector back to 2D matrix
    // vec^{-1}: R^N -> R^{rows * cols}
    const unflatten = (vector) => {
        let matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix.push(vector.slice(i * cols, (i + 1) * cols));
        }
        return matrix;
    };

    // Helper: Dot product
    // <a,b> = a^T * b = sum_{k=1}^N {a_k * b_k}
    const dot = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);

    // 1. Flatten matrices and calculate residuals: R_k = F(V_k) - V_k
    // Any residual is counted as: R_k = F(V_k) - V_k;
    // R is initialized as an empty set to hold residual vectors from the calculation given here;
    // F is initialized as an empty set to hold the (flattened) Bellman vector updates - explained below
    let R = [];
    let F = [];
    for (let i = 0; i < m; i++) {
        // we vectorize (flatten) the 2D Bellman update matrix into a 1D vector
        // f_i = vec(F(V_i));
        let f_flat = flatten(fHistory[i]);
        // we vectorize (flatten) the 2D value matrix into 1D vectors
        // v_i = vec(V_i);
        let v_flat = flatten(vHistory[i]);
        // F <- F U {f_i}
        // we store the (flattened) Bellman vector in set F
        F.push(f_flat);
        // residual vector: r_i = f_i - v_i;
        // we define the residual vector r_i as the element wise error between Bellman vector update and prev values
        // R <- R U {r_i}
        let r_flat = f_flat.map((val, idx) => val - v_flat[idx]);
        R.push(r_flat);
    }

    // 2. Build the normal equations matrix A for the least squares problem;
    // Minimizing || sum(alpha_i * R_i) || subject to sum(alpha_i) = 1;
    // We use a Lagrange multiplier, creating an (m+1) x (m+1) system;
    // Explanation: we initialize a square matrix A with zeros and a vector b as well:
    // A is an element of R^{(m+1) * (m+1)}; R here is the real numbers set
    // the size here is m+1 because we have m historical points
    // plus 1 row/ column for the Lagrange multiplier constraint;
    // Vector b is an element of R^(m+1), R here being the real number set again
    let A = Array.from({ length: m + 1 }, () => Array(m + 1).fill(0));
    let B = Array(m + 1).fill(0);

    // Gram matrix creation explanation:
    // we populate top left m * m size (out of total (m+1)(m+1) size matrix)
    // by taking the dot products of every residual (flattened) vector with every other one, mathematically:
    // for every i,j in {1,...,m} we calculate A_{i,j} = r_i^T * r_j;
    // we also append a unit 1 value to every final column element of every row i, and also 1 to every final row element of every column i;
    // In calculus, this arises directly from taking the partial derivative of the Lagrange multiplier lambda;
    // mathematically for i in {1,...,m} we calculate A_{i,m+1} = 1 and A_{m+1,i} = 1.
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            A[i][j] = dot(R[i], R[j]);
        }
        A[i][m] = 1; // Lagrange multiplier row
        A[m][i] = 1; // Lagrange multiplier col
    }
    // B_{m+1} = 1; we set the final element of the vector b to 1
    B[m] = 1; // The constraint: sum(alpha) = 1

    // 3. Solve the linear system A * x = b for vector x
    // Explanation: the solution vector x in R^{m+1} will take the form
    // x = [alpha_1, alpha_2, alpha_3, ..., alpha_m, lambda]^T
    // giving us the optimal weights

    let X = solveGaussianElimination(A, B);

    // If det(A) is approx 0, return {vec}^{-1}(f_m)
    // The system may be singular or nearly singular when the residual vectors r_1, ..., r_m are linearly dependent or
    // nearly dependent. In that case, the coefficients are not uniquely or stably determined, so the solver
    // returns failure and the caller falls back to the latest Bellman update.
    if (!X) return unflatten(F[m - 1]);

    // alpha = [x_1, x_2, ..., x_m]^T
    // Explanation: We discard the Lagrange multiplier lambda (which is x_{m+1}) because it served its purpose during the
    // matrix solve. We keep only the m calculated weights.
    const alphas = X.slice(0, m);

    // 4. Finally, the core Anderson acceleration step:
    // we compute the accelerated vector: V_accel = sum(alpha_i * F_i)

    // We first initialize an empty 1D vector to hold our final accelerated values
    // Let v_{accel} in R^N, v_{accel} = 0; R here is set of real numbers
    let v_accel_flat = Array(rows * cols).fill(0);

    // Mathematically, v_{accel} = sum_{i=1}^m alpha_i * f_i
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < v_accel_flat.length; j++) {
            // We scale each historical Bellman update vector (f_i) by its corresponding optimal weight alpha_i
            // and sum them all together to create the new accelerated 1D vector.
            v_accel_flat[j] += alphas[i] * F[i][j];
        }
    }

    return unflatten(v_accel_flat);
}

// !!!
/**
 * Standard Gaussian elimination with partial pivoting.
 * Used internally by the Anderson Accelerator to solve the m x m linear system.
 */
function solveGaussianElimination(A, B) {
    let n = B.length;
    let M = A.map((row, i) => [...row, B[i]]); // Augmented matrix

    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
                maxRow = k;
            }
        }
        // Swap rows
        [M[i], M[maxRow]] = [M[maxRow], M[i]];

        // Check for singular matrix (near-zero pivot)
        if (Math.abs(M[i][i]) < 1e-10) return null;

        // Eliminate
        for (let k = i + 1; k < n; k++) {
            let c = -M[k][i] / M[i][i];
            for (let j = i; j <= n; j++) {
                if (i === j) M[k][j] = 0;
                else M[k][j] += c * M[i][j];
            }
        }
    }

    // Back substitution
    let X = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        X[i] = M[i][n] / M[i][i];
        for (let k = i - 1; k >= 0; k--) {
            M[k][n] -= M[k][i] * X[i];
        }
    }
    return X;
}

// !!!
/**
 * Computes the Log-Sum-Exp of an array of Q-values to prevent floating-point overflow/underflow.
 *
 * @param {number[]} qValues - Array of Q-values for available actions at state s.
 * @param {number} tau - Temperature hyperparameter (tau > 0). Smaller tau approaches hard max.
 * @returns {number} Soft max value for state s.
 */
function logSumExpQValues(qValues, tau = 0.1) {
    if (!qValues || qValues.length === 0) return 0;

    // Find maximum Q-value for numerical stability (Log-Sum-Exp trick)
    const qMax = Math.max(...qValues);

    let sumExp = 0;
    for (let i = 0; i < qValues.length; i++) {
        sumExp += Math.exp((qValues[i] - qMax) / tau);
    }

    return qMax + tau * Math.log(sumExp);
}

// !!!
/**
 * Calculates the Soft Bellman state update across all available actions.
 * Drop-in alternative to hard-max calculateNextMoves.
 *
 * @param {number[]} qValues - Array of Q-values for each action.
 * @param {boolean} useSoftBellman - Toggle between hard Bellman and Soft Bellman.
 * @param {number} tau - Temperature parameter for Soft Bellman.
 * @returns {number} New state value V(s).
 */
function computeSoftBellmanUpdate(qValues, useSoftBellman = true, tau = 0.15) {
    if (!useSoftBellman) {
        // Standard Hard Bellman Optimality Operator: V(s) = max_a Q(s, a)
        return Math.max(...qValues);
    }

    // Soft Bellman Optimality Operator: V_soft(s) = tau * log(sum(exp(Q(s,a) / tau)))
    return logSumExpQValues(qValues, tau);
}

/* The core solver is given in the planPath and computePolicies functions */
function planPath() {

    // define policies matrix pi_k(s) initialized to 0 ("no action assigned")
    let policies = Array.from({ length: rows }, () => Array(columns).fill(0)); // initialize each policy cell to no moves yet
    // define for each cell s, value function V_k(s), initialized to 0
    values   = Array.from({ length: rows }, () => Array(columns).fill(0)); // initialize each value cell to starting utility

    //Get the selected mode from the HTML dropdown
    const mode = document.getElementById("solverMode").value;

    if (mode === "enhanced") {
        // --- ENHANCED MODE ---
        // Uses Anderson Acceleration + returns new matrix + uses smoothed path
        values = computePolicies(values, policies);
        showSmoothedPath(policies);
    } else {
        // --- CLASSIC MODE ---
        computePoliciesPreAccel(values, policies);
        showPath(policies);
    }

}

/**
 * Performs a full sweep of the grid to compute the Bellman update F(V). This creates a new matrix based on the current
 * values without modifying the original matrix, which is a requirement for Anderson Acceleration.
 *
 * @param {Array} currentValues - The current value function matrix V_k.
 * @returns {Array} - The resulting value function matrix after one Bellman update F(V_k).
 */
function performBellmanSweep(currentValues) {
    // Initialize a new matrix to store the results of the update (F(V))
    let nextValues = Array.from({ length: rows }, () => Array(columns).fill(0));

    // 1. Fixed values for absorbing states:
    // The goal state must maintain its delivery reward in the new matrix.
    nextValues[goalNode[0]][goalNode[1]] = deliveryReward;

    // Obstacles must maintain their negative repair cost in the new matrix.
    for (let r of rivals) {
        nextValues[r[0]][r[1]] = -repairCost;
    }

    // 2. Compute the Bellman update for all non-terminal states:
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            let currPosn = [i, j];

            // Only calculate updates for states that are NOT the goal and NOT obstacles.
            if ((currPosn[0] !== goalNode[0] || currPosn[1] !== goalNode[1]) &&
                !rivals.some((pos) => currPosn[0] === pos[0] && currPosn[1] === pos[1])) {

                // We call calculateNextMoves with updateInPlace = false.
                // This uses the math logic to find the new value but DOES NOT modify the current values matrix.
                // We pass an empty array for policies as we only need the value for the snapshot.
                nextValues[i][j] = calculateNextMoves(currPosn, currentValues, [], false);
            }
        }
    }

    return nextValues;
}

function computePolicies(values, policies) {

    mapHazardPositions();

    values[goalNode[0]][goalNode[1]] = deliveryReward;

    for (let r of rivals) {
        let i = r[0], j = r[1];
        values[i][j] = -repairCost;
    }

    // Anderson Acceleration setup
    let vHistory = []; // Stores V_k snapshots
    let fHistory = []; // Stores F(V_k) snapshots
    const m = 5;       // History window size

    while (true) {
        // Store snapshot for convergence check and history
        let prev = copyValues(values);

        // Use performBellmanSweep instead of in-place loop:
        // this generates F(V) without modifying the current values matrix
        let nextValues = performBellmanSweep(values);

        // Manage Anderson histories
        vHistory.push(prev);
        fHistory.push(nextValues);

        if (vHistory.length > m) {
            vHistory.shift();
            fHistory.shift();
        }

        // Update the value function using Anderson Acceleration
        values = applyAndersonAcceleration(vHistory, fHistory, rows, columns);

        // Comparing accelerated values
        if (converges(prev, values)) {
            break;
        }
    }

    // Since Anderson Acceleration only converges the values, we run one final pass to populate the policies matrix and
    // ensure values are consistent.
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            let currPosn = [i, j];
            if ((currPosn[0] !== goalNode[0] || currPosn[1] !== goalNode[1]) && !rivals.some((pos) => currPosn[0] === pos[0] && currPosn[1] === pos[1])) {
                // Use updateInPlace = true to save the deterministic policy
                calculateNextMoves(currPosn, values, policies, true);
            }
        }
    }

    // Return the entire converged matrix, not just the scalar value of the start node.
    // The original Value Iteration implementation relied on in-place mutation of the global values matrix (in
    // calculateNextMoves), ensuring that the memory reference used by the visualization remained constant while the
    // data within it updated. In contrast, Anderson Acceleration requires the creation of new matrices to store
    // snapshots for its fixed-point iteration logic. This results in the solver returning a new array reference
    // rather than modifying the existing one. Consequently, the global values variable used by the gradient function
    // remained pointed at the initial zero-initialized matrix, resulting in a monochrome display.
    // To resolve this, computePolicies was modified to return the final converged matrix instead of a scalar value.
    // In planPath, the global values reference is now explicitly updated with the return value of the solver.
    // This ensures that the normalization and quantization processes in the gradient visualization access the updated
    // utility values, restoring the grayscale transition across the state space.
    return values;
}

/* Deprecated */
function computePoliciesPreAccel(values, policies) {

    // creates a deep copy of the init "values" matrix V(s), where any s is a (i,j) node
    let prev = copyValues(values); // getting a copy to track changes from convergence
    mapHazardPositions();

    // V(s_goal) <-- R_goal == deliveryReward;
    // goal is treated as an absorbing state with fixed value
    values[goalNode[0]][goalNode[1]] = deliveryReward; // set the utility of the goal cell to the delivery reward

    for (let r of rivals) {
        let i = r[0], j = r[1];

        // -R_repair = -repairCost for all s in S_obstacles;
        // obstacles are absorbing terminal states with fixed negative value
        values[i][j] = -repairCost; // set the utility of each obstacle to the negative repair cost
    }

    // this is the Bellman optimality operator iterative application
    while (true) { // loop until the value function converges (values stop changing significantly)
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < columns; j++) {
                let currPosn = [i, j]; // possible error here

                // this condition ensures that the Bellman update is only applied to non-terminal states: skipping end goal and all obstacles/rivals;
                // S_absorbing === S_goal + S_obstacle
                if ((currPosn[0] !== goalNode[0] || currPosn[1] !== goalNode[1]) && !rivals.some((pos) => currPosn[0] === pos[0] && currPosn[1] === pos[1])) {
                    calculateNextMoves(currPosn, values, policies); // calculate the best move, and update utility for cell
                }
            }
        }
        if (converges(prev, values)) {
            break; // stop looping and return if board converges
        } else {
            prev = copyValues(values); // continue looping and keep track of previous values nxn matrix
        }
    }
    return values[startNode[0]][startNode[1]]; // returns utility values at the start position
}


function converges(prev, curr, converge_factor = 0.01) {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            if (Math.abs(prev[i][j] - curr[i][j]) > converge_factor) {
                return false; // if the difference is larger than the convergence threshold, matrices have not converged
            }
        }
    }
    return true; // if all the differences are within the threshold, they converged
}

// !!!
/* Implementation of the Bellman optimality update for a single state s=(i,j) */
// Added 'updateInPlace' parameter (defaulting to true) to allow the function to be used for snapshotting in
// Anderson Acceleration without modifying the source matrix.
/* The original calculateNextMoves updates the values matrix in-place (values[currPosn[0]][currPosn[1]] = new_val).
 Anderson Acceleration mathematically requires a "Jacobi" update—meaning we calculate all the new values
 for the whole board based on the old values before updating anything. If we update in-place, the acceleration fails.
 To fix this without duplicating your code, we only need to make one change to the existing calculateNextMoves to
 make it flexible, and then update computePolicies. The non-redundant minimal way to do this is given here. */
function calculateNextMoves(currPosn, values, policies, updateInPlace = true) {

    // assume all neighboring positions are in range
    // immediately set, then reset if needed
    let s_range = true;
    let w_range = true;
    let n_range = true;
    let e_range = true;

    // checking if moving would go out of bounds;
    // the boolean flags determine if the successor states in the cardinal directions lie within the grid;
    // defining the deterministic successor function succ(s,a) for only 4 cardinal directions first (if cardinality of action space is 4);
    // if the move is illegal, succ(s,a) = s

    // down/south
    if (currPosn[0] + 1 > rows - 1) s_range = false;
    // left/west
    if (currPosn[1] - 1 < 0) w_range = false;
    // up/north
    if (currPosn[0] - 1 < 0) n_range = false;
    // right/east
    if (currPosn[1] + 1 > columns - 1) e_range = false;

    let s_posn;
    let w_posn;
    let n_posn;
    let e_posn;
    let sw_posn;
    let se_posn;
    let nw_posn;
    let ne_posn;

    // determining actual neighboring positions, if out of bounds, stay in place

    // move south
    if (s_range) s_posn = [currPosn[0] + 1, currPosn[1]];
    else s_posn = currPosn;

    // move west
    if (w_range) w_posn = [currPosn[0], currPosn[1] - 1];
    else w_posn = currPosn;

    // move north
    if (n_range) n_posn = [currPosn[0] - 1, currPosn[1]];
    else n_posn = currPosn;

    // move east
    if (e_range) e_posn = [currPosn[0], currPosn[1] + 1];
    else e_posn = currPosn;

    // analogous for diagonals, if enabled

    if (s_range && w_range) sw_posn = [currPosn[0] + 1, currPosn[1] - 1];
    else sw_posn = currPosn;

    if (s_range && e_range) se_posn = [currPosn[0] + 1, currPosn[1] + 1];
    else se_posn = currPosn;

    if (n_range && w_range) nw_posn = [currPosn[0] - 1, currPosn[1] - 1];
    else nw_posn = currPosn;

    if (n_range && e_range) ne_posn = [currPosn[0] - 1, currPosn[1] + 1];
    else ne_posn = currPosn;

    // The code computes expected utility Q(s,a): sum of all possible successor states weighted by transition probabilities
    // calculate the utility at all neighboring positions;
    // direction_probability * (-1 * powerCost + (discount * values[next[y]][next[x]]));
    // next === 2 adjacent to current direction

    let s =
        0.7 * (-1 * powerCost + discount * values[s_posn[0]][s_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[w_posn[0]][w_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[e_posn[0]][e_posn[1]]);

    let w =
        0.7 * (-1 * powerCost + discount * values[w_posn[0]][w_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[n_posn[0]][n_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[s_posn[0]][s_posn[1]]);

    let n =
        0.7 * (-1 * powerCost + discount * values[n_posn[0]][n_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[w_posn[0]][w_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[e_posn[0]][e_posn[1]]);

    let e =
        0.7 * (-1 * powerCost + discount * values[e_posn[0]][e_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[n_posn[0]][n_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[s_posn[0]][s_posn[1]]);

    // diagonal directions
    // The same perpendicular-slip logic applies: the 2 directions at 90 to the intended diagonal receive lower probability each

    let sw =
        0.7 * (-1 * powerCost + discount * values[sw_posn[0]][sw_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[se_posn[0]][se_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[nw_posn[0]][nw_posn[1]]);

    let se =
        0.7 * (-1 * powerCost + discount * values[se_posn[0]][se_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[sw_posn[0]][sw_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[ne_posn[0]][ne_posn[1]]);

    let nw =
        0.7 * (-1 * powerCost + discount * values[nw_posn[0]][nw_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[sw_posn[0]][sw_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[ne_posn[0]][ne_posn[1]]);

    let ne =
        0.7 * (-1 * powerCost + discount * values[ne_posn[0]][ne_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[nw_posn[0]][nw_posn[1]]) +
        0.15 * (-1 * powerCost + discount * values[se_posn[0]][se_posn[1]]);

    // Select the max and store the argmax;
    // V_new(s) = max{Q(s, E), Q(s, N), Q(s, W), Q(s, S)};
    // pi(s) = max{Q(s, E), Q(s, N), Q(s, W), Q(s, S)};
    // policy is encoded as 1=East; 2=North; 3=West; 4=South;
    if (directions === 4) {
        let moves = [e, n, w, s]; // add all possible 4 moves to a list

        // Here's the trick: we want to apply the Soft Bellman operator to update the Value landscape
        // (which makes the gradient smoother and helps convergence), but we still want to extract the strict highest
        // Q-value to define the deterministic Policy so your showPath visualizer doesn't break.
        // Soft update for the value landscape (calculating V_new(s))
        let new_val = computeSoftBellmanUpdate(moves, true, 0.15);

        // Hard argmax for the deterministic policy trajectory visualization
        let best_q = Math.max(...moves);
        let max_move = moves.indexOf(best_q);

        // update value & policies matrices
        // values[currPosn[0]][currPosn[1]] = new_val; // assign soft utility to current cell
        // policies[currPosn[0]][currPosn[1]] = max_move + 1; // store best deterministic move

        // Wrapped update in 'if (updateInPlace)' block.
        // This prevents modifying the 'values' matrix when we only need to compute a result for a snapshot.
        if (updateInPlace) {
            values[currPosn[0]][currPosn[1]] = new_val; // assign soft utility to current cell
            policies[currPosn[0]][currPosn[1]] = max_move + 1; // store best deterministic move
        }

        // Return new_val so the calling function can collect values into a new matrix.
        return new_val;

    }
    // Here the encoding is: 1=E, 2=N, 3=W, 4=S, 5=SW, 6=SE,7=NW, 8=NE
    else if (directions === 8) {
        let moves = [e, n, w, s, sw, se, nw, ne];

        // Soft update for the value landscape (calculating V_new(s))
        let new_val = computeSoftBellmanUpdate(moves, true, 0.15);

        // Hard argmax for the deterministic policy trajectory visualization
        let best_q = Math.max(...moves);
        let max_move = moves.indexOf(best_q);

        // update value & policies matrices
        // values[currPosn[0]][currPosn[1]] = new_val; // assign soft utility to current cell
        // policies[currPosn[0]][currPosn[1]] = max_move + 1; // store best deterministic move

        // Wrapped update in 'if (updateInPlace)' block.
        // This prevents modifying the 'values' matrix when we only need to compute a result for a snapshot.
        if (updateInPlace) {
            values[currPosn[0]][currPosn[1]] = new_val; // assign soft utility to current cell
            policies[currPosn[0]][currPosn[1]] = max_move + 1; // store best deterministic move
        }

        // Return new_val so the calling function can collect values into a new matrix.
        return new_val;
    }
}

/* Scan the entire state space S and classify each cell: reconstruct the sets {s_start}, {s_goal}, S_obs from the
current board configuration. The user may have moved start/ goal or added obstacles since the last computation. */
function mapHazardPositions() {
    rivals = [];
    // startNode = null; goalNode = null;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            if (board[i][j] === "start") {
                startNode = [i, j]; // store the start position
            }
            if (board[i][j] === "goal") {
                goalNode = [i, j]; // store the goal position
            }
            if (board[i][j] === "obstacle") {
                rivals.push([i, j]); // add an obstacle to the array
            }
        }
    }
}

/* UI state locking: when the path is generated, parameters are locked to prevent the user from changing mdp definition
while the solution is being displayed, ensuring consistency between policy being displayed and the parameters that generated it */
function toggleButtons() {
    const ids = ["powerCost", "repairCost", "deliveryReward", "discount", "contrast", "wall", "start_btn", "goal_btn", "four-directions", "eight-directions", "toggleGradient"];
    ids.forEach(id => { // disabling/enabling buttons depending on if 
        const element = document.getElementById(id);
        if (element) element.disabled = !element.disabled;
        if (element.disabled && element.classList.contains("active"))
            element.classList.remove("active");
    }    );
    selectMode(""); // un-selecting the last selected mode after generating a path
}

// !!!
/* Path display using Gradient Descent */
function showSmoothedPath(policies) {
    // clear the previous path visualization
    const gridContainer = document.querySelector(".grid-container");

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            if (board[i][j] === "path") {
                board[i][j] = "empty";  // clear all existing paths
                // remove CSS class
                const oldCell = gridContainer.querySelector(`.cell[data-row="${i}"][data-col="${j}"]`);
                if (oldCell) oldCell.classList.remove("path");
            }
        }
    }

    /*
       CRITICAL DIFFERENCE FROM showPath(): showPath() colors cells in real-time as it walks the policy.
       To apply Gradient Descent, we cannot draw immediately. We must follow a "Collect -> Smooth -> Draw" pipeline.
       First, we collect all coordinates of the optimal discrete path into an array.
    */
    let pathCoords = [];
    let currRow = startNode[0], currCol = startNode[1]; // start tracing path
    let steps = 0;

    // Store the starting position
    pathCoords.push([currRow, currCol]);

    while (policies[currRow] && policies[currRow][currCol] !== 0 && steps < rows * columns) {
        steps++; // safety measure to stop & not go overboard
        switch (policies[currRow][currCol]) {  // move to the next cell based on the policy
            case 1: // move right
                currCol++;
                break;
            case 2: // move up
                currRow--;
                break;
            case 3: // move left
                currCol--;
                break;
            case 4: // move down
                currRow++;
                break;
            case 5: // move south-west
                currRow++; currCol--;
                break;
            case 6: // move south-east
                currRow++; currCol++;
                break;
            case 7: // move north-west
                currRow--; currCol--;
                break;
            case 8: // move north-east
                currRow--; currCol++;
                break;
            default: // invalid policy
                return null;
        }
        // Instead of coloring the cell now, we save the coordinate to the list
        pathCoords.push([currRow, currCol]);
    }

    /*
       SMOOTHING STEP: Now that we have the complete discrete path, we pass it to the PathOptimizer. The Gradient Descent
       logic will shift these points to remove "stair-stepping" while ensuring they don't collide with obstacles (rivals).
    */
    let smoothedPath = PathOptimizer.smoothPath(pathCoords, rivals);

    /*
       FINAL RENDERING: We now iterate through the smoothed floating-point coordinates. We use Math.round() to find
        the nearest discrete cell to color on the grid.
    */
    smoothedPath.forEach(([r, c]) => {
        let row = Math.round(r);
        let col = Math.round(c);

        // Ensure the smoothed point is still within the grid boundaries
        if (row >= 0 && row < rows && col >= 0 && col < columns) {
            const selector = `.cell[data-row="${row}"][data-col="${col}"]`;
            const cell = gridContainer.querySelector(selector);

            if (cell && !cell.classList.contains("obstacle") && !cell.classList.contains("start") && !cell.classList.contains("goal")) {
                board[row][col] = "path";
                cell.classList.add("path"); // add the "path" class to the current cell
            }
        }
    });
}


/* Path display without Gradient Descent, deprecated.
* This is a policy rollout, i.e. a trajectory simulation. Starting from s = s_start , the sequence is generated by:
* s_{t+1} = succ(s_t, π(s_t));
* Loop terminates when: policy at the current state is 0 (which occurs at the goal and obstacles, what with them being
* fixed), or once a safety counter "steps" exceeds rows * columns, preventing infinite loops if the policy contains a
* cycle. Note that this traces the intended path under the optimal policy, not a stochastic sample. It visualizes the
* deterministic plan that the agent intends to follow, even though the actual execution in the stochastic environment
* would deviate with some probability.
* */
function showPath(policies) {
    // clear the previous path visualization
    const gridContainer = document.querySelector(".grid-container");

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            if (board[i][j] === "path") {
                board[i][j] = "empty";  // clear all existing paths
                // remove CSS class
                const oldCell = gridContainer.querySelector(`.cell[data-row="${i}"][data-col="${j}"]`);
                if (oldCell) oldCell.classList.remove("path");
            }
        }
    }

    let currRow = startNode[0], currCol = startNode[1]; // start tracing path
    let steps = 0;

    while (policies[currRow] && policies[currRow][currCol] !== 0 && steps < rows*columns) {
        steps++; // safety measure to stop & not go overboard
        switch (policies[currRow][currCol]) {  // move to the next cell based on the policy
            case 1: // move right
                currCol++;
                break;
            case 2: // move up
                currRow--;
                break;
            case 3: // move left
                currCol--;
                break;
            case 4: // move down
                currRow++;
                break;
            case 5: // move south-west
                currRow++; currCol--;
                break;
            case 6: // move south-east
                currRow++; currCol++;
                break;
            case 7: // move north-west
                currRow--; currCol--;
                break;
            case 8: // move north-east
                currRow--; currCol++;
                break;
            default: // invalid policy
                return null;
        }
        const selector = `.cell[data-row="${currRow}"][data-col="${currCol}"]`;
        const cell = gridContainer.querySelector(selector);

        if (cell && !cell.classList.contains("obstacle") && !cell.classList.contains("start") && !cell.classList.contains("goal")) {
            board[currRow][currCol] = "path";
            cell.classList.add("path"); // add the "path" class to the current cell
        }
    }
}

function toggleGrid() {
    const toggleButton = document.getElementById("toggleGrid");
    toggleButton.classList.toggle("active"); // button appearance on toggle

    const gridContainer = document.querySelector(".grid-container");
    gridContainer.classList.toggle("no-border"); // grid visibility
}

function compressMatrixTo255(matrix) { // linear compression algorithm for turning the negative values instead to values 0-255
    const flat = matrix.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);

    if (min === max) return matrix.map(row => row.map(() => 128)); // 128 cause it looked better than 255, 255 was too light

    return matrix.map(row =>
        row.map(value => {
            // normalize 0–1;
            // apply a min max affine transformation that collapses the dynamic range of the value function
            // into the unit interval
            let norm = (value - min) / (max - min);
            // gamma correction power law contrast: the greater the coefficient the more emphasized the difference,
            // and vice versa for values under 1.
            // apply contrast custom, so it can work nice with big boards and small ones :D
            norm = Math.pow(norm, contrast);
            // quantize: scale to 8 bit grayscale
            // scale to 0–255
            return Math.round(norm * 255);
        })
    );
}

/* The gradient toggle overlays the value function onto the grid. High-utility states (close to the goal) appear bright,
*  while low-utility states (near obstacles or far from the goal) appear dark. This provides an intuitive heatmap of the
*  "desirability" of each state under the optimal policy. */
function toggleGradient() {

    const btn_togglePath = document.getElementById("togglePath");
    if (!btn_togglePath.classList.contains("active")) return;

    const toggledGradient = document.getElementById("toggleGradient");
    const gridContainer = document.querySelector(".grid-container");

    toggledGradient.classList.toggle("active"); // toggle the active class for the button

    if (!values || values.length === 0) return;

    // compress values only when turning gradient on
    if (!gradientActive) lastCompressedValues = compressMatrixTo255(values);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {

            const cell = gridContainer.querySelector(`.cell[data-row="${i}"][data-col="${j}"]`);
            if (!cell) continue;

            if (!cell.dataset.originalBg) { // store original background
                cell.dataset.originalBg = window.getComputedStyle(cell).backgroundColor;
            }

            const preservedClasses = new Set(["start", "goal", "obstacle", "path"]);

            if (!gradientActive) {
                // apply gradient only to empty cells
                if (![...preservedClasses].some(cls => cell.classList.contains(cls))) {
                    const value = lastCompressedValues[i][j];
                    cell.style.backgroundColor = `rgb(${value}, ${value}, ${value})`;
                }
            } else {
                // restore original background
                if (![...preservedClasses].some(cls => cell.classList.contains(cls))) {
                    cell.style.backgroundColor = ""; // let other functions take control
                }
            }
        }
    }
    gradientActive = !gradientActive;
}

function clearBoard() {

    const generatedPath = document.getElementById("togglePath");
    if (generatedPath.classList.contains("active")) togglePath(); // if path is toggled when clearing, we turn it off
    board = new Array(rows).fill(0).map(() => new Array(columns).fill("empty")); // regenerating the map
    rivals = []; // resetting rivals array
    setup();   // update UI
}

/* Deep independent copy of the value function matrix V_k(s); used to save the state before a sweep (prev) for a convergence test */
function copyValues(values){
    let new_values = [];
    for (let i = 0; i < rows; i++) {
        new_values[i] = [];
        for (let j = 0; j < columns; j++) {
            new_values[i][j] = values[i][j];
        }
    }
    return new_values; // return the newly created copy of the matrix
}

// =============================================================================
// TRAJECTORY OPTIMIZATION MODULE (Gradient Descent Smoothing)
// =============================================================================

class PathOptimizer {
    /**
     * Smooths a discrete grid path using a Gradient Descent approach.
     *
     * This method is fully dynamic: it does not rely on hardcoded coordinates.
     * It treats the first and last coordinates of the input path as the
     * dynamic Start and Goal anchors, ensuring the smoothed trajectory
     * always begins and ends exactly where the user placed the nodes.
     *
     * @param {Array} path - The discrete path produced by the solver [[r, c], [r, c]...].
     * @param {Array} obstacles - The dynamic 'rivals' array [ [r, c]... ].
     * @returns {Array} - A smoothed trajectory with floating-point coordinates.
     */
    static smoothPath(path, obstacles) {
        // Safety check: If the path is too short or empty, smoothing is impossible/unnecessary.
        if (!path || path.length < 3) return path;

        // Hyperparameters for the Gradient Descent process
        const iterations = 100;       // Number of refinement passes
        const learningRate = 0.1;     // Step size for coordinate updates
        const smoothnessWeight = 0.5; // Influence of the internal straightening force
        const repulsionWeight = 2.0;  // Influence of the obstacle avoidance force
        const repulsionRadius = 1.5;  // The distance threshold for obstacle repulsion

        // Create a deep copy of the path.
        // We map the discrete integers to floats to allow the points to move
        // smoothly between cell centers (sub-pixel precision).
        let smoothed = path.map(p => [parseFloat(p[0]), parseFloat(p[1])]);

        for (let iter = 0; iter < iterations; iter++) {
            // DYNAMIC ANCHORING:
            // We loop from index 1 to length - 2.
            // This explicitly 'pins' smoothed[0] (Dynamic Start) and
            // smoothed[smoothed.length - 1] (Dynamic Goal), ensuring
            // the trajectory never drifts away from the user-defined endpoints.
            for (let i = 1; i < smoothed.length - 1; i++) {
                let curr = smoothed[i];
                let prev = smoothed[i - 1];
                let next = smoothed[i + 1];

                // --- 1. INTERNAL TENSION FORCE (Smoothing) ---
                // This calculates the gradient toward the midpoint of the neighbors.
                // It removes the "stair-step" blockiness of the grid.
                let smoothForceR = (prev[0] + next[0]) / 2 - curr[0];
                let smoothForceC = (prev[1] + next[1]) / 2 - curr[1];

                // --- 2. EXTERNAL REPULSION FORCE (Obstacle Avoidance) ---
                let repelForceR = 0;
                let repelForceC = 0;

                // Dynamically iterate through all obstacles currently on the board
                for (let obs of obstacles) {
                    let distR = curr[0] - obs[0];
                    let distC = curr[1] - obs[1];
                    let distanceSq = distR * distR + distC * distC;
                    let distance = Math.sqrt(distanceSq);

                    // Only apply force if the point is within the influence radius of the obstacle
                    if (distance < repulsionRadius && distance > 0) {
                        // Force magnitude is inversely proportional to distance (1/d^3)
                        // This creates a "hard" push-back as the path gets closer to a wall.
                        let strength = repulsionWeight / (distanceSq * distance);
                        repelForceR += distR * strength;
                        repelForceC += distC * strength;
                    }
                }

                // --- 3. GRADIENT DESCENT UPDATE ---
                // We update the coordinates by moving them in the direction of the combined forces.
                curr[0] += learningRate * (smoothnessWeight * smoothForceR + repelForceR);
                curr[1] += learningRate * (smoothnessWeight * smoothForceC + repelForceC);
            }
        }

        return smoothed;
    }
}