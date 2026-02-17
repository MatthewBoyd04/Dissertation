from GenericTile import GenericTile

class BlockedTile(GenericTile):
    spritePath: str

    def __init__(self):
        self.spritePath = "Sprites/BlockedTile"