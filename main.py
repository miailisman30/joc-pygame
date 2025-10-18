from game_engine import GameWindow, SpriteObject, GameObject, RectObject
import sys
import pygame
import random

width = 800
height = 600
game_engine = None

class PlayerObject(RectObject):
	def __init__(self, x, y, width, height, color, pipes_manager, on_lose=lambda: None):
		super().__init__(x, y, width, height)
		self.sprite = SpriteObject(0, 0, width, height, color)
		self.add_child(self.sprite)
		self.speed = 200
		self.velocity_y = 0
		self.pipes_manager = pipes_manager
		self.on_lose = on_lose

	def update(self, dt):
		super().update(dt) 
		keys = pygame.key.get_pressed()
		if keys[pygame.K_SPACE]:
			self.velocity_y = -300

		self.velocity_y += 800 * dt
		ground_level = height - self.sprite.height
		new_pos = self.y + self.velocity_y * dt
		if new_pos > ground_level:
			new_pos = ground_level
			self.velocity_y = 0
		elif new_pos < 0:
			new_pos = 0
			self.velocity_y = 0
		self.y = new_pos
		# keep our collision rect in sync with position

		# collision detection with pipes
		# PipesManager now stores pipes as a list of (top_pipe, bottom_pipe) tuples.
		# Build a flat list of rects for collision testing.
		pipe_rects = []
		for pair in self.pipes_manager.pipes:
			top, bottom = pair
			pipe_rects.append(top.rect)
			pipe_rects.append(bottom.rect)

		idx_collision = self.collide_list(pipe_rects)
		if idx_collision != -1:
			self.on_lose()
			
	def draw(self, surface):
		ax, ay = self.get_abs_pos()
		fps_text = f"Pos: {int(ax)}, {int(ay)}"
		surf = pygame.font.Font(None, 24).render(fps_text, True, pygame.Color("white"))
		surface.blit(surf, (150, 8))

		# Draw a horizontal line at the y position of the next pipe gap for debugging
		# Use the player's world x to query the pipes manager. If no pipes exist,
		# next_pipe_pos returns (None, None) and we skip drawing the line.
		gap_x, gap_y = self.pipes_manager.next_pipe_pos(self.x)
		if gap_y is not None:
			# draw across the whole screen
			color = pygame.Color("yellow")
			pygame.draw.line(surface, color, (0, int(gap_y)), (width, int(gap_y)), 2)

class PipeObject(SpriteObject):
	def __init__(self, x, y, width, height, color):
		super().__init__(x, y, width, height, color)
		self.passed_score = False

class PipesManager(GameObject):
	def __init__(self, x_pos_score):
		'''
		x_pos_score: x position where the player scores by passing pipes
		'''
		super().__init__()
		self.x_pos_score = x_pos_score
		self.score = 0

		self.pipes = []
		# optional shared pipe image (Surface) that will be applied to spawned pipes
		self.pipe_image = None
		self.pipe_speed = 150
		self.spawn_timer = 0.0
		self.spawn_interval = 2.0
		self.gap_height = 150
		self.pipe_width = 80

	def update(self, dt):
		self.spawn_timer += dt
		if self.spawn_timer >= self.spawn_interval:
			self.spawn_timer -= self.spawn_interval
			self.spawn_pipe()

		# Iterate over pipe pairs (top, bottom)
		for pair in self.pipes[:]:
			top, bottom = pair
			# move both pipes left
			top.x -= self.pipe_speed * dt
			bottom.x -= self.pipe_speed * dt

			# update their rects via their update calls
			top.update(dt)
			bottom.update(dt)

			# increase score once per pair when the pair passes scoring x position
			if not getattr(top, 'passed_score', False) and top.x + top.width < self.x_pos_score:
				top.passed_score = True
				print("Score!")
				self.score += 1

			# remove pair if completely off-screen
			if top.x + top.width < 0:
				# remove both from children and from pipes list
				try:
					self.children.remove(top)
				except ValueError:
					pass
				try:
					self.children.remove(bottom)
				except ValueError:
					pass
				try:
					self.pipes.remove(pair)
				except ValueError:
					pass

	def spawn_pipe(self):
		top_height = random.randint(50, height - self.gap_height - 50)
		bottom_height = height - top_height - self.gap_height
		x_spawn = width
		top_pipe = PipeObject(x_spawn, 0, self.pipe_width, top_height, (0,255,0))
		bottom_pipe = PipeObject(x_spawn, top_height + self.gap_height, self.pipe_width, bottom_height, (0,255,0))
		# if a pipe image is set on this manager, assign images to pipes
		if getattr(self, 'pipe_image', None):
			# flip top vertically so opening faces down
			top_img = pygame.transform.flip(self.pipe_image, False, True)
			top_pipe.image = top_img
			bottom_pipe.image = self.pipe_image

		# store as a tuple (top, bottom)
		self.pipes.append((top_pipe, bottom_pipe))
		self.add_child(top_pipe)
		self.add_child(bottom_pipe)

	def set_pipe_image(self, surface):
		"""Set the shared pipe image Surface used for newly spawned pipes
		and apply it immediately to any existing pipes. Top pipes will get a
		vertically flipped version so the openings face correctly.

		Args:
			surface (pygame.Surface|None): image to use for pipes, or None to
			use the colored rect fallback.
		"""
		self.pipe_image = surface
		for pair in self.pipes:
			top, bottom = pair
			if surface is None:
				top.image = None
				bottom.image = None
			else:
				top.image = pygame.transform.flip(surface, False, True)
				bottom.image = surface

	def load_and_set_pipe_image(self, path):
		"""Load an image from `path` and set it as the pipe image. Returns
		True on success, False otherwise.
		"""
		try:
			img = pygame.image.load(path).convert_alpha()
			self.set_pipe_image(img)
			return True
		except Exception as e:
			print(f"Failed to load pipe image '{path}':", e)
			self.set_pipe_image(None)
			return False

	def reset(self):
		# remove pipe pairs and children safely
		for pair in self.pipes[:]:
			top, bottom = pair
			try:
				self.children.remove(top)
			except ValueError:
				pass
			try:
				self.children.remove(bottom)
			except ValueError:
				pass
		self.pipes.clear()
		# children that are not pipes will remain
		self.spawn_timer = 0.0
		self.score = 0

	def next_pipe_pos(self, x_player_pos, offset=0):
		"""
		Return the (x, y) position of the gap center for the pipe pair directly in front of
		the player (based on x_player_pos). If the player is already inside the gap, an
		offset can be provided to select the pipe further ahead (offset > 0 moves the
		selection forward by that many pixels).

		Args:
			x_player_pos (float): the player's x position (world coordinates).
			offset (float): number of pixels to treat as if the player is further back; this
			lets the function return the current pipe's gap when the player is inside it.

		Returns:
			(tuple): (gap_x, gap_y) coordinates of the center of the gap, or (None, None)
			if no pipes exist.
		"""
		# Defensive: no pipes
		if not self.pipes:
			return (None, None)

		# We consider the pipe's reference x to be the left edge of the pipe sprite
		# Compute an effective player x that moves the selection backward by offset
		effective_x = x_player_pos - offset

		# Find the first pipe pair whose right edge is >= effective_x, or whose
		# center is >= effective_x. Using left edge + width/2 gives center x.
		for pair in self.pipes:
			top, bottom = pair
			pipe_center_x = top.x + top.width / 2
			# If the pipe center is in front of (or at) the effective player x, choose it
			if pipe_center_x >= effective_x:
				gap_center_x = pipe_center_x
				# gap center y is top.height + gap_height/2
				gap_center_y = top.height + self.gap_height / 2
				return (gap_center_x, gap_center_y)

		# If none found (all pipes are behind), return the last pipe's gap
		last_top, last_bottom = self.pipes[-1]
		return (last_top.x + last_top.width / 2, last_top.height + self.gap_height / 2)

	def draw(self, surface):
		fps_text = f"Pipes Score: {self.score}"
		surf = pygame.font.Font(None, 24).render(fps_text, True, pygame.Color("white"))
		surface.blit(surf, (8, 28))


fps = 60

class EndScreenManager(GameObject):
	def __init__(self):
		super().__init__()
		self.is_game_over = False

	def show_game_over(self):
		self.is_game_over = True
		self.set_enable_children(False)

	def draw(self, surface):
		if self.is_game_over:
			fps_text = "Game Over! Press R to Restart"
			surf = pygame.font.Font(None, 48).render(fps_text, True, pygame.Color("red"))

			rect = surf.get_rect(center=(width//2, height//2))
			surface.blit(surf, rect.topleft)

	# detect if you passed a pipe and increase score


	def update(self, dt):
		# listen for R key to restart if game is over
		if self.is_game_over:
			keys = pygame.key.get_pressed()
			if keys[pygame.K_r]:
				print("Restarting game...")
				self.is_game_over = False
				self.set_enable_children(True)
				# reset player position and pipes
				pl = self.get_child_of_type(PlayerObject)
				pl.x = 100
				pl.y = 100
				pl.velocity_y = 0
				pm = self.get_child_of_type(PipesManager)
				# properly remove existing pipe sprites and reset spawn timer so
				# pipes don't immediately respawn at the same positions
				pm.reset()
				

def main(argv=None):
	window = GameWindow(width=width, height=height, fps=fps)
	game_engine = window.game_engine

	end_screen_manager = EndScreenManager()
	game_engine.add_object(end_screen_manager)

	player_x_pos = 100

	pipes = PipesManager(player_x_pos)
	loaded_img = None
	for fn in ("images/nr2.png", "images/pipe.png"):
		try:
			loaded_img = pygame.image.load(fn).convert_alpha()
			print(f"Loaded pipe image: {fn}")
			break
		except Exception:
			loaded_img = None

	# attach the loaded image (or None) to the pipes manager
	pipes.pipe_image = loaded_img

	# game_engine.add_object(pipes)
	end_screen_manager.add_child(pipes)

	player = PlayerObject(player_x_pos, 100, 50, 50, (255,0,0), pipes_manager=pipes, on_lose=end_screen_manager.show_game_over)
	# game_engine.add_object(player)
	end_screen_manager.add_child(player)

	try:
		player_img = pygame.image.load("images/saila1.png").convert_alpha()
		player.sprite.set_image(player_img)
		print("Loaded player image: images/saila1.png")
	except Exception:
		print("Player image images/saila1.png not found; using colored rect")


	window.run()


if __name__ == "__main__":
	main(sys.argv[1:])