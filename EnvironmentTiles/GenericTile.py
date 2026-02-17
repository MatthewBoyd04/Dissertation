from abc import ABC, abstractmethod

class GenericTile(ABC):
    isFatal: bool = False
    spritePath: str

    @abstractmethod
    def getSprite(self):
        pass

    def getIsFatal(self):
        return self.isFatal
    
    
