from GenericTile import GenericTile

class AgentTile(GenericTile):
    spritePath: str

    def __init__(self, agentNumber: int):
        self.spritePath = "Sprites/Drone_" + str(agentNumber) + ".png"

