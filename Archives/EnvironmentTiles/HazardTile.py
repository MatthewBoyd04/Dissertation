from GenericTile import GenericTile

class HazardTile(GenericTile):
    spritePath: str
    isFatal: bool = True

    def __init__(self):
        self.spritePath = "Sprites/HazardTile"