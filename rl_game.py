import sys
from pkg.game.ProcessGame import ProcessGame
from pkg.config import config as PkgConfig

if __name__ == '__main__':
    if not PkgConfig.GAME_PUSH_ALGORITHM:
        sys.exit(0)
    game = ProcessGame()
    game.run() 
