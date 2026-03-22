<?php
/**
 * act.php
 * Receives a manual action from the dashboard and writes it to the
 * per-game command file that snake.py / moon_lander.py poll.
 *
 * Called by index.html via fetch('act.php', {method:'POST', body: JSON})
 * Body: {"game": 3, "action": "up"}
 *
 * The Python game reads /tmp/plaiades_cmd_<game>.txt on its next step,
 * consumes it (deletes it), and uses it in place of the AI decision.
 *
 * Allowed games:  3 (Snake), 10 (Moon Lander)
 * Allowed actions:
 *   G3:  up, down, left, right
 *   G10: idle, main, left, right, main_left, main_right, stabilize
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'POST required']);
    exit;
}

// Parse JSON body
$raw  = file_get_contents('php://input');
$body = json_decode($raw, true);

if (!$body || !isset($body['game'], $body['action'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Missing game or action field']);
    exit;
}

$game   = (int) $body['game'];
$action = strtolower(trim($body['action']));

// Validate game
$allowed_games = [3, 10];
if (!in_array($game, $allowed_games, true)) {
    http_response_code(400);
    echo json_encode(['error' => "Game {$game} does not support manual play"]);
    exit;
}

// Validate action per game
$allowed_actions = [
    3  => ['up', 'down', 'left', 'right'],
    10 => ['idle', 'main', 'left', 'right', 'main_left', 'main_right', 'stabilize'],
];

if (!in_array($action, $allowed_actions[$game], true)) {
    http_response_code(400);
    echo json_encode(['error' => "Invalid action '{$action}' for game {$game}"]);
    exit;
}

// Write command to the temp file that Python polls
$cmdFile = sys_get_temp_dir() . "/plaiades_cmd_{$game}.txt";

// Use exclusive lock to avoid race conditions
$written = file_put_contents($cmdFile, $action, LOCK_EX);

if ($written === false) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to write command file']);
    exit;
}

echo json_encode([
    'ok'     => true,
    'game'   => $game,
    'action' => $action,
    'file'   => $cmdFile,
]);