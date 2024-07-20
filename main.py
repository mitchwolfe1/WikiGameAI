import sys

from src.game import WikiGame

if __name__ == "__main__":
    start_url = sys.argv[1]
    end_url = sys.argv[2]
    game = WikiGame(start_url, end_url, 10)
    game.start_game()
