import logging
from typing import Dict, Any, Optional

import aigf.main as aigf_main
import aigf.interface as interface
import server.logic as logic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FROGGER - %(levelname)s - %(message)s")

class FroggerGameServer(interface.GameInterface):
    """
    Frogger game server implementation using the AI Game Framework.
    """

    def __init__(self) -> None:
        super().__init__()
        self.game = logic.Frogger()
        self.is_real_time = True
        self.fps = 30
        self.player_id: Optional[int] = None

    async def on_player_connect(self, player_id: int) -> None:
        logging.info(f"Player {player_id} connected.")
        if self.player_id is None:
            self.player_id = player_id
            # Start game automatically when player connects
            self.state = interface.GameState.RUNNING
        else:
            logging.warning(f"Extra player {player_id} connected. Only one player supported.")

    async def on_player_disconnect(self, player_id: int) -> None:
        logging.info(f"Player {player_id} disconnected.")
        if self.player_id == player_id:
            self.player_id = None
            self.state = interface.GameState.LOBBY
            self.game.reset_game()

    async def on_reset_sim(self) -> None:
        """
        Natively called by the framework when a RESET command is received.
        """
        self.game.reset_game()
        self.state = interface.GameState.LOBBY

    async def process_action(self, player_id: int, action: Dict[str, Any]) -> None:
        logging.debug(f"Processing action from {player_id}: {action}")
        
        # System/Frontend commands (like start_sim, stop_sim, reset_sim)
        # are handled natively by the new framework scaffolding.

        # Player actions
        if self.state == interface.GameState.RUNNING and player_id == self.player_id:
            if action.get("action") == "move":
                direction = action.get("direction")
                if isinstance(direction, str):
                    self.game.move_frog(direction)

    async def tick(self, dt: float) -> None:
        if self.state == interface.GameState.RUNNING:
            self.game.update(dt)
            manager = getattr(aigf_main, "manager", None)
            if manager is not None and self.player_id is not None:
                await manager.send_agent_state(self.player_id, self.get_state())
            if self.game.game_over:
                logging.info("Game Over!")
                self.state = interface.GameState.LOBBY

    def get_state(self) -> Dict[str, Any]:
        state = self.game.get_state()
        state["player_id"] = self.player_id
        return state

    def get_setup_payload(self) -> Dict[str, Any]:
        return {
            "width": self.game.width,
            "height": self.game.height
        }

if __name__ == "__main__":
    import argparse
    from aigf.main import run_app
    
    parser = argparse.ArgumentParser(description="Frogger Game Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    args = parser.parse_args()
    
    server = FroggerGameServer()
    run_app(server, host=args.host, port=args.port)
