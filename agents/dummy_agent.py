import asyncio
import random
from typing import Optional
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    async def deliberate(self) -> Optional[str]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        
        # Simple random move using rotation-invariant directions
        return random.choice(["NORTH", "SOUTH", "EAST", "WEST"])

if __name__ == "__main__":
    agent = DummyAgent()
    asyncio.run(agent.run())
