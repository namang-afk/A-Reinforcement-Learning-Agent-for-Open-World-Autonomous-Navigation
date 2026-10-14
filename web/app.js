/**
 * Autonomous Navigation RL Agent - Frontend Dashboard Controller
 * Handles canvas rendering, telemetry updates, playback loop, and API sync.
 */

// State Management
const state = {
  grid: [],
  gridSize: 20,
  agentPos: [0, 0],
  targetPos: [0, 0],
  dynamicObstacles: [],
  trajectory: [],
  localView: [],
  steps: 0,
  stepReward: 0.0,
  totalReward: 0.0,
  distance: 0,
  isTerminated: false,
  isTruncated: false,
  isSuccess: false,
  policy: 'ppo',
  isPlaying: false,
  playSpeed: 5, // steps per second
  intervalId: null,
  seed: 42,
  episodeHistory: []
};

// DOM References
const navCanvas = document.getElementById('nav-canvas');
const navCtx = navCanvas.getContext('2d');
const sensorCanvas = document.getElementById('sensor-canvas');
const sensorCtx = sensorCanvas.getContext('2d');

const policySelect = document.getElementById('policy-select');
const btnPlay = document.getElementById('btn-play');
const btnStep = document.getElementById('btn-step');
const btnRunFull = document.getElementById('btn-run-full');
const btnResetAgent = document.getElementById('btn-reset-agent');
const btnGenerateMap = document.getElementById('btn-generate-map');
const btnRandomSeed = document.getElementById('btn-random-seed');

const speedSlider = document.getElementById('speed-slider');
const speedDisplay = document.getElementById('speed-display');
const gridSizeSlider = document.getElementById('grid-size-slider');
const gridSizeVal = document.getElementById('grid-size-val');
const wallDensitySlider = document.getElementById('wall-density-slider');
const wallDensityVal = document.getElementById('wall-density-val');
const dynObsSlider = document.getElementById('dyn-obs-slider');
const dynObsVal = document.getElementById('dyn-obs-val');
const mapSeedInput = document.getElementById('map-seed-input');
const manualHint = document.getElementById('manual-hint');

// Telemetry DOM elements
const statSteps = document.getElementById('stat-steps');
const statDistance = document.getElementById('stat-distance');
const statStepReward = document.getElementById('stat-step-reward');
const statTotalReward = document.getElementById('stat-total-reward');
const actionChosenBadge = document.getElementById('action-chosen-badge');
const episodeStatusPill = document.getElementById('episode-status-pill');
const viewportHeading = document.getElementById('viewport-heading');
const historyTableBody = document.getElementById('history-table-body');
const backendStatusText = document.getElementById('backend-status-text');

// Toast Notification
function showToast(message, icon = 'ℹ️') {
  const toast = document.getElementById('toast');
  document.getElementById('toast-icon').textContent = icon;
  document.getElementById('toast-message').textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2800);
}

// ---------------------------------------------------------------------------
// API Communication
// ---------------------------------------------------------------------------

async function checkBackendStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ppo_available) {
      backendStatusText.textContent = `Model Ready (${data.device.toUpperCase()})`;
    } else {
      backendStatusText.textContent = 'PPO Model Not Found (Baselines Only)';
    }
  } catch (err) {
    backendStatusText.textContent = 'Disconnected';
  }
}

async function resetEnvironment(newMap = false) {
  pause();

  const payload = {
    grid_size: parseInt(gridSizeSlider.value),
    wall_density: parseInt(wallDensitySlider.value) / 100.0,
    num_dynamic_obstacles: parseInt(dynObsSlider.value),
    seed: parseInt(mapSeedInput.value),
    regenerate: newMap
  };

  try {
    const res = await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    state.grid = data.grid;
    state.gridSize = data.grid_size;
    state.agentPos = data.agent_pos;
    state.targetPos = data.target_pos;
    state.dynamicObstacles = data.dynamic_obstacles || [];
    state.trajectory = [data.agent_pos];
    state.localView = data.local_view;
    state.steps = 0;
    state.stepReward = 0.0;
    state.totalReward = 0.0;
    state.distance = data.distance;
    state.isTerminated = false;
    state.isTruncated = false;
    state.isSuccess = false;

    updateUI();
    renderMainGrid();
    renderSensorView();

    if (newMap) {
      showToast(`Generated Map Seed #${payload.seed}`, '🗺️');
    }
  } catch (err) {
    console.error('Reset error:', err);
    showToast('Failed to reset environment', '⚠️');
  }
}

async function stepEnvironment(manualAction = null) {
  if (state.isTerminated || state.isTruncated) {
    pause();
    return;
  }

  const payload = {
    policy: state.policy,
    action: manualAction
  };

  try {
    const res = await fetch('/api/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    state.agentPos = data.agent_pos;
    state.dynamicObstacles = data.dynamic_obstacles || [];
    state.trajectory = data.trajectory;
    state.localView = data.local_view;
    state.steps = data.steps;
    state.stepReward = data.reward;
    state.totalReward = data.total_reward;
    state.distance = data.distance;
    state.isTerminated = data.terminated;
    state.isTruncated = data.truncated;
    state.isSuccess = data.is_success;

    actionChosenBadge.textContent = data.action_name || '--';

    updateUI();
    renderMainGrid();
    renderSensorView();

    if (state.isSuccess) {
      pause();
      showToast('Goal Reached! Target Achieved!', '🎯');
      recordEpisodeHistory('SUCCESS');
    } else if (state.isTerminated) {
      pause();
      showToast('Collision! Episode Terminated.', '💥');
      recordEpisodeHistory('COLLISION');
    } else if (state.isTruncated) {
      pause();
      showToast('Time Limit Exceeded (Truncated)', '⏱️');
      recordEpisodeHistory('TIMEOUT');
    }
  } catch (err) {
    console.error('Step error:', err);
    pause();
  }
}

async function runFullEpisode() {
  pause();
  showToast('Simulating episode...', '⚡');

  const payload = {
    policy: state.policy,
    grid_size: parseInt(gridSizeSlider.value),
    wall_density: parseInt(wallDensitySlider.value) / 100.0,
    num_dynamic_obstacles: parseInt(dynObsSlider.value),
    seed: parseInt(mapSeedInput.value)
  };

  try {
    const res = await fetch('/api/run_episode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    // Animate the simulated episode smoothly
    let frameIdx = 0;
    const frames = data.frames;
    const interval = Math.max(1000 / state.playSpeed, 40);

    const animTimer = setInterval(() => {
      if (frameIdx >= frames.length) {
        clearInterval(animTimer);
        const last = frames[frames.length - 1];
        if (last.is_success) {
          showToast(`Episode Solved in ${last.steps} steps!`, '🏆');
          recordEpisodeHistory('SUCCESS');
        } else if (last.terminated) {
          showToast('Collision occurred', '💥');
          recordEpisodeHistory('COLLISION');
        } else {
          showToast('Episode ended', '🏁');
          recordEpisodeHistory('TIMEOUT');
        }
        return;
      }

      const f = frames[frameIdx];
      state.agentPos = f.agent_pos;
      state.dynamicObstacles = f.dynamic_obstacles;
      state.trajectory = f.trajectory;
      state.localView = f.local_view;
      state.steps = f.steps;
      state.stepReward = f.reward;
      state.totalReward = f.total_reward;
      state.distance = f.distance;
      state.isTerminated = f.terminated;
      state.isTruncated = f.truncated;
      state.isSuccess = f.is_success;
      actionChosenBadge.textContent = f.action_name;

      updateUI();
      renderMainGrid();
      renderSensorView();
      frameIdx++;
    }, interval);
  } catch (err) {
    console.error('Run episode error:', err);
    showToast('Failed to run full episode', '⚠️');
  }
}

// ---------------------------------------------------------------------------
// Canvas Rendering
// ---------------------------------------------------------------------------

function resizeCanvasToFit() {
  const container = document.querySelector('.canvas-container');
  const size = Math.min(container.clientWidth - 16, 520);
  navCanvas.width = size;
  navCanvas.height = size;
}

function renderMainGrid() {
  const width = navCanvas.width;
  const height = navCanvas.height;
  const n = state.gridSize;
  const cellSize = width / n;

  navCtx.clearRect(0, 0, width, height);

  // Background grid cells
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val = state.grid[r] ? state.grid[r][c] : 0;
      const x = c * cellSize;
      const y = r * cellSize;

      if (val === 1) {
        // Wall
        navCtx.fillStyle = '#334155';
        navCtx.fillRect(x, y, cellSize, cellSize);
        navCtx.strokeStyle = '#1e293b';
        navCtx.lineWidth = 1;
        navCtx.strokeRect(x, y, cellSize, cellSize);
      } else {
        // Free cell
        navCtx.fillStyle = (r + c) % 2 === 0 ? '#0f172a' : '#141e33';
        navCtx.fillRect(x, y, cellSize, cellSize);
      }
    }
  }

  // Draw Trajectory Trail
  if (state.trajectory.length > 1) {
    navCtx.beginPath();
    navCtx.strokeStyle = 'rgba(6, 182, 212, 0.65)';
    navCtx.lineWidth = Math.max(cellSize * 0.22, 2);
    navCtx.lineCap = 'round';
    navCtx.lineJoin = 'round';

    state.trajectory.forEach((pos, idx) => {
      const px = pos[1] * cellSize + cellSize / 2;
      const py = pos[0] * cellSize + cellSize / 2;
      if (idx === 0) navCtx.moveTo(px, py);
      else navCtx.lineTo(px, py);
    });
    navCtx.stroke();

    // Start marker
    const start = state.trajectory[0];
    navCtx.fillStyle = '#8b5cf6';
    navCtx.fillRect(
      start[1] * cellSize + cellSize * 0.2,
      start[0] * cellSize + cellSize * 0.2,
      cellSize * 0.6,
      cellSize * 0.6
    );
  }

  // Draw Dynamic Obstacles
  state.dynamicObstacles.forEach(pos => {
    const ox = pos[1] * cellSize + cellSize / 2;
    const oy = pos[0] * cellSize + cellSize / 2;
    const radius = cellSize * 0.38;

    navCtx.beginPath();
    navCtx.arc(ox, oy, radius, 0, Math.PI * 2);
    navCtx.fillStyle = '#f59e0b';
    navCtx.fill();
    navCtx.strokeStyle = '#fbbf24';
    navCtx.lineWidth = 2;
    navCtx.stroke();
  });

  // Draw Target (with glowing pulse effect)
  if (state.targetPos) {
    const tx = state.targetPos[1] * cellSize + cellSize / 2;
    const ty = state.targetPos[0] * cellSize + cellSize / 2;
    const tRadius = cellSize * 0.42;

    navCtx.save();
    navCtx.shadowColor = '#10b981';
    navCtx.shadowBlur = 12;
    navCtx.beginPath();
    navCtx.arc(tx, ty, tRadius, 0, Math.PI * 2);
    navCtx.fillStyle = '#10b981';
    navCtx.fill();
    navCtx.restore();

    // Bullseye inner ring
    navCtx.beginPath();
    navCtx.arc(tx, ty, tRadius * 0.5, 0, Math.PI * 2);
    navCtx.fillStyle = '#ecfdf5';
    navCtx.fill();
  }

  // Draw Agent
  if (state.agentPos) {
    const ax = state.agentPos[1] * cellSize + cellSize / 2;
    const ay = state.agentPos[0] * cellSize + cellSize / 2;
    const aRadius = cellSize * 0.4;

    navCtx.save();
    navCtx.shadowColor = '#ef4444';
    navCtx.shadowBlur = 14;
    navCtx.beginPath();
    navCtx.arc(ax, ay, aRadius, 0, Math.PI * 2);
    navCtx.fillStyle = '#ef4444';
    navCtx.fill();
    navCtx.restore();

    // Inner agent dot
    navCtx.beginPath();
    navCtx.arc(ax, ay, aRadius * 0.35, 0, Math.PI * 2);
    navCtx.fillStyle = '#ffffff';
    navCtx.fill();
  }
}

function renderSensorView() {
  const size = sensorCanvas.width;
  sensorCtx.clearRect(0, 0, size, size);

  if (!state.localView || state.localView.length === 0) return;

  const viewDim = 5;
  const cellSize = size / viewDim;

  for (let r = 0; r < viewDim; r++) {
    for (let c = 0; c < viewDim; c++) {
      const idx = r * viewDim + c;
      const val = state.localView[idx];
      const x = c * cellSize;
      const y = r * cellSize;

      if (val === 1) {
        sensorCtx.fillStyle = '#334155'; // Wall
      } else if (val === 2) {
        sensorCtx.fillStyle = '#f59e0b'; // Dynamic Obstacle
      } else if (val === 3) {
        sensorCtx.fillStyle = '#10b981'; // Target
      } else {
        sensorCtx.fillStyle = '#0f172a'; // Empty
      }

      sensorCtx.fillRect(x, y, cellSize, cellSize);
      sensorCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      sensorCtx.strokeRect(x, y, cellSize, cellSize);

      // Center cell is the agent itself
      if (r === 2 && c === 2) {
        sensorCtx.beginPath();
        sensorCtx.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.35, 0, Math.PI * 2);
        sensorCtx.fillStyle = '#ef4444';
        sensorCtx.fill();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Telemetry & UI Updates
// ---------------------------------------------------------------------------

function updateUI() {
  statSteps.textContent = state.steps;
  statDistance.textContent = state.distance !== undefined ? state.distance.toFixed(1) : '--';
  statStepReward.textContent = state.stepReward.toFixed(2);
  statTotalReward.textContent = state.totalReward.toFixed(2);

  // Status pill
  if (state.isSuccess) {
    episodeStatusPill.textContent = 'GOAL REACHED';
    episodeStatusPill.style.background = 'rgba(16, 185, 129, 0.2)';
    episodeStatusPill.style.color = '#34d399';
  } else if (state.isTerminated) {
    episodeStatusPill.textContent = 'COLLISION';
    episodeStatusPill.style.background = 'rgba(239, 68, 68, 0.2)';
    episodeStatusPill.style.color = '#f87171';
  } else if (state.isTruncated) {
    episodeStatusPill.textContent = 'TIMEOUT';
    episodeStatusPill.style.background = 'rgba(245, 158, 11, 0.2)';
    episodeStatusPill.style.color = '#fbbf24';
  } else {
    episodeStatusPill.textContent = 'ACTIVE';
    episodeStatusPill.style.background = 'rgba(6, 182, 212, 0.15)';
    episodeStatusPill.style.color = 'var(--accent-cyan)';
  }

  viewportHeading.textContent = `Environment: Seed #${state.seed} (${state.gridSize}×${state.gridSize})`;
}

function recordEpisodeHistory(result) {
  const row = {
    policy: state.policy.toUpperCase(),
    steps: state.steps,
    reward: state.totalReward.toFixed(1),
    result: result
  };
  state.episodeHistory.unshift(row);
  if (state.episodeHistory.length > 8) state.episodeHistory.pop();

  historyTableBody.innerHTML = state.episodeHistory.map(item => `
    <tr>
      <td>${item.policy}</td>
      <td>${item.steps}</td>
      <td>${item.reward}</td>
      <td class="${item.result === 'SUCCESS' ? 'tag-success' : 'tag-fail'}">${item.result}</td>
    </tr>
  `).join('');
}

// ---------------------------------------------------------------------------
// Playback Loop
// ---------------------------------------------------------------------------

function play() {
  if (state.isPlaying) return;
  state.isPlaying = true;
  btnPlay.textContent = '⏸ Pause';
  btnPlay.classList.add('btn-primary');

  const interval = 1000 / state.playSpeed;
  state.intervalId = setInterval(() => {
    stepEnvironment();
  }, interval);
}

function pause() {
  state.isPlaying = false;
  btnPlay.textContent = '▶ Play';
  if (state.intervalId) {
    clearInterval(state.intervalId);
    state.intervalId = null;
  }
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

btnPlay.addEventListener('click', () => {
  if (state.isPlaying) pause();
  else play();
});

btnStep.addEventListener('click', () => {
  pause();
  stepEnvironment();
});

btnRunFull.addEventListener('click', runFullEpisode);

btnResetAgent.addEventListener('click', () => {
  resetEnvironment(false);
});

btnGenerateMap.addEventListener('click', () => {
  state.seed = parseInt(mapSeedInput.value);
  resetEnvironment(true);
});

btnRandomSeed.addEventListener('click', () => {
  const newSeed = Math.floor(Math.random() * 10000);
  mapSeedInput.value = newSeed;
  state.seed = newSeed;
  resetEnvironment(true);
});

policySelect.addEventListener('change', (e) => {
  state.policy = e.target.value;
  manualHint.style.display = state.policy === 'manual' ? 'block' : 'none';
  showToast(`Active Policy: ${e.target.options[e.target.selectedIndex].text}`, '🤖');
});

speedSlider.addEventListener('input', (e) => {
  state.playSpeed = parseInt(e.target.value);
  speedDisplay.textContent = `${state.playSpeed} steps/s`;
  if (state.isPlaying) {
    pause();
    play();
  }
});

gridSizeSlider.addEventListener('input', (e) => {
  gridSizeVal.textContent = `${e.target.value} × ${e.target.value}`;
});

wallDensitySlider.addEventListener('input', (e) => {
  wallDensityVal.textContent = `${e.target.value}%`;
});

dynObsSlider.addEventListener('input', (e) => {
  dynObsVal.textContent = e.target.value;
});

// Manual Keyboard Movement (WASD / Arrows)
window.addEventListener('keydown', (e) => {
  if (state.policy !== 'manual') return;

  let action = null;
  if (e.key === 'ArrowUp' || e.key.toLowerCase() === 'w') action = 0;
  else if (e.key === 'ArrowDown' || e.key.toLowerCase() === 's') action = 1;
  else if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') action = 2;
  else if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') action = 3;

  if (action !== null) {
    e.preventDefault();
    stepEnvironment(action);
  }
});

window.addEventListener('resize', () => {
  resizeCanvasToFit();
  renderMainGrid();
});

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  resizeCanvasToFit();
  checkBackendStatus();
  resetEnvironment(true);
});
