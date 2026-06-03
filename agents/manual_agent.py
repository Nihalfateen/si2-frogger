import asyncio
import json
import select
import sys

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

try:
    import websockets
except ImportError:
    print("Error: The 'websockets' library is required to run the agent.")
    print("Please install it: pip install websockets")
    sys.exit(1)

async def receive_loop(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "setup":
                print(f"\n[Handshake Complete] Assigned Frogger Player ID: {data.get('player_id')}")
                print("Controls: W/S/A/D to move, Q to quit.")
                print("="*60)
            elif data.get("type") == "state":
                score = data.get("score", 0)
                lives = data.get("lives", 0)
                high_score = data.get("high_score", 0)
                frog_x = data.get("frog_x", 0.0)
                frog_y = data.get("frog_y", 0)
                game_over = data.get("game_over", False)

                # Clean screen redraw
                sys.stdout.write("\033[H\033[2J")
                sys.stdout.flush()

                print("="*18 + " FROGGER TERMINAL HUD " + "="*18)
                print(f"HIGH SCORE: {high_score:<10} | SCORE: {score:<10} | LIVES: {lives:<5}")
                print("-" * 58)
                print(f"Frog Position  : ({frog_x:.1f}, {frog_y})")
                if game_over:
                    print("="*18 + " 💥 GAME OVER! 💥 " + "="*18)
                else:
                    print("="*58)
                print("\n[ACTIVE INPUT] Focus this terminal. Press W/A/S/D to move frog. Press Q to quit.")
    except websockets.exceptions.ConnectionClosed:
        print("\nDisconnected from Frogger Server.")

async def send_loop(websocket):
    fd = sys.stdin.fileno() if HAS_TERMIOS else None
    old_settings = None
    if HAS_TERMIOS and fd is not None:
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    key_mapping = {
        "w": "NORTH",
        "s": "SOUTH",
        "a": "WEST",
        "d": "EAST"
    }

    try:
        while True:
            key = ""
            if HAS_TERMIOS and fd is not None:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
            else:
                line = sys.stdin.readline().strip().lower()
                if line in key_mapping:
                    key = line
                elif line == "q":
                    key = "q"

            if key:
                if key.lower() == "q":
                    break
                mapped_direction = key_mapping.get(key.lower())
                if mapped_direction:
                    await websocket.send(json.dumps({"action": "move", "direction": mapped_direction}))
            await asyncio.sleep(0.02)
    finally:
        if HAS_TERMIOS and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting Manual Agent...")

async def main():
    url = "ws://localhost:8765/ws"
    print(f"Connecting to Frogger Server on {url}...")
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps({"client": "agent", "name": "Terminal Frogger"}))
            await asyncio.gather(
                receive_loop(websocket),
                send_loop(websocket)
            )
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nFrogger driver exited.")
