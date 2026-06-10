import { GameClient } from '/aigf/framework.js';

const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const livesEl = document.getElementById('lives');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');

const TILE_SIZE = 40;
let width = 11;
let height = 9;

// Nord Palette
const NORD = {
    nord0: "#2e3440",
    nord1: "#3b4252",
    nord2: "#434c5e",
    nord3: "#4c566a",
    nord4: "#d8dee9",
    nord5: "#e5e9f0",
    nord6: "#eceff4",
    nord7: "#8fbcbb",
    nord8: "#88c0d0",
    nord9: "#81a1c1",
    nord10: "#5e81ac",
    nord11: "#bf616a", // red
    nord12: "#d08770", // orange
    nord13: "#ebcb8b", // yellow
    nord14: "#a3be8c", // green
    nord15: "#b48ead"  // purple
};

const CAR_COLORS = {
    "small_fast": NORD.nord13,
    "small_slow": NORD.nord12,
    "large_fast": NORD.nord11,
    "large_slow": NORD.nord15
};

// Initialize canvas size immediately
canvas.width = width * TILE_SIZE;
canvas.height = height * TILE_SIZE;

const client = new GameClient(Number(window.location.port) || 8765);

client.onSetup = (data) => {
    width = data.width;
    height = data.height;
    canvas.width = width * TILE_SIZE;
    canvas.height = height * TILE_SIZE;
};

client.onUpdate = (data) => {
    scoreEl.innerText = data.score;
    livesEl.innerText = data.lives;
    
    const state = data._framework.state;
    statusEl.innerText = state;
    statusEl.className = 'badge ' + (state === 'RUNNING' ? 'badge-running' : 'badge-lobby');
    
    draw(data);
};

startBtn.onclick = () => client.sendCommand('START');
resetBtn.onclick = () => client.sendCommand('RESET');

function draw(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw lanes
    for (let y = 0; y < height; y++) {
        if (y === 0 || y === 4 || y === 8) {
            ctx.fillStyle = NORD.nord3; // Safe Lanes
        } else {
            ctx.fillStyle = NORD.nord1; // Road
        }
        ctx.fillRect(0, canvas.height - (y + 1) * TILE_SIZE, canvas.width, TILE_SIZE);
        
        // Highway marking between lane 3 and 4, and 4 and 5
        if (y === 3 || y === 4) {
             ctx.strokeStyle = NORD.nord4;
             ctx.lineWidth = 1;
             ctx.setLineDash([5, 5]);
             ctx.beginPath();
             ctx.moveTo(0, canvas.height - (y + 1) * TILE_SIZE);
             ctx.lineTo(canvas.width, canvas.height - (y + 1) * TILE_SIZE);
             ctx.stroke();
             ctx.setLineDash([]);
        }
    }

    // Draw obstacles
    if (state.obstacles) {
        state.obstacles.forEach(obs => {
            ctx.fillStyle = CAR_COLORS[obs.variant] || NORD.nord4;
            
            // Draw car body
            ctx.fillRect(
                obs.x * TILE_SIZE + 2, 
                canvas.height - (obs.y + 1) * TILE_SIZE + 2, 
                obs.width * TILE_SIZE - 4, 
                TILE_SIZE - 4
            );

            // Handle wrap-around rendering
            if (obs.x + obs.width > width) {
                ctx.fillRect(
                    (obs.x - width) * TILE_SIZE + 2,
                    canvas.height - (obs.y + 1) * TILE_SIZE + 2,
                    obs.width * TILE_SIZE - 4,
                    TILE_SIZE - 4
                );
            } else if (obs.x < 0) {
                ctx.fillRect(
                    (obs.x + width) * TILE_SIZE + 2,
                    canvas.height - (obs.y + 1) * TILE_SIZE + 2,
                    obs.width * TILE_SIZE - 4,
                    TILE_SIZE - 4
                );
            }
        });
    }

    // Draw Frog
    ctx.fillStyle = NORD.nord14;
    ctx.beginPath();
    const fx = state.frog_x * TILE_SIZE + TILE_SIZE / 2;
    const fy = canvas.height - (state.frog_y + 1) * TILE_SIZE + TILE_SIZE / 2;
    ctx.arc(fx, fy, TILE_SIZE / 2 - 5, 0, Math.PI * 2);
    ctx.fill();

    if (state.game_over) {
        ctx.fillStyle = 'rgba(46, 52, 64, 0.85)'; // nord0 with alpha
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = NORD.nord6;
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText("GAME OVER", canvas.width / 2, canvas.height / 2 - 40);
        ctx.font = 'bold 24px Arial';
        ctx.fillText(`Final Score: ${state.score}`, canvas.width / 2, canvas.height / 2 + 10);
        ctx.fillText(`Max Points: ${state.high_score}`, canvas.width / 2, canvas.height / 2 + 50);
    }
}
