from flappy_saila import FlappySailaGame
import sys

def main(argv=None):
	headless = "--headless" in (argv or [])
	window = FlappySailaGame(width=800, height=600, fps=60, headless=headless)
	window.run()

if __name__ == "__main__":
	main(sys.argv[1:])