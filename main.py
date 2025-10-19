from flappy_saila import FlappySailaGame
import sys

def main(argv=None):
	window = FlappySailaGame(width=800, height=600, fps=60, headless=True)
	window.run()

if __name__ == "__main__":
	main(sys.argv[1:])