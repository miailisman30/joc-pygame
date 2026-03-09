from src.engine.game_engine import GameEnvironment, SpriteObject, GameObject, RectObject, load_image
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

		self.jump_requested = False

	def update(self, dt):
		super().update(dt) 
		keys = pygame.key.get_pressed()

		if keys[pygame.K_SPACE]:
			self.jump_requested = True
		
		if self.jump_requested:
			self.velocity_y = -300
			self.jump_requested = False

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

		gap_x, gap_y = self.pipes_manager.next_pipe_pos(self.x)
		if gap_y is not None:
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

		self.pipes: list[tuple[PipeObject, PipeObject]] = []
		self.pipe_image: pygame.Surface | None = None
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

		for pair in self.pipes[:]:
			top, bottom = pair
			top.x -= self.pipe_speed * dt
			bottom.x -= self.pipe_speed * dt

			top.update(dt)
			bottom.update(dt)

			if not getattr(top, 'passed_score', False) and top.x + top.width < self.x_pos_score:
				top.passed_score = True
				print("Score!")
				self.score += 1

			if top.x + top.width < 0:
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
		if self.pipe_image:
			top_img = pygame.transform.flip(self.pipe_image, False, True)
			top_pipe.image = top_img
			bottom_pipe.image = self.pipe_image

		self.pipes.append((top_pipe, bottom_pipe))
		self.add_child(top_pipe)
		self.add_child(bottom_pipe)

	def reset(self):
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
		if not self.pipes:
			return (None, None)


		effective_x = x_player_pos - offset

		for pair in self.pipes:
			top, bottom = pair
			pipe_center_x = top.x + top.width / 2
			if pipe_center_x >= effective_x:
				gap_center_x = pipe_center_x
				gap_center_y = top.height + self.gap_height / 2
				return (gap_center_x, gap_center_y)

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



	def update(self, dt):
		if self.is_game_over:
			keys = pygame.key.get_pressed()
			if keys[pygame.K_r]:
				print("Restarting game...")
				self.is_game_over = False
				self.set_enable_children(True)
				pl = self.get_child_of_type(PlayerObject)
				if pl:
					pl.x = 100
					pl.y = 100
					pl.velocity_y = 0
				pm = self.get_child_of_type(PipesManager)
				if pm:
					pm.reset()


class FlappySailaGame(GameEnvironment):
	def __init__(self, width=800, height=600, headless=False, fps=60):
		super().__init__(width, height, headless, fps)

		end_screen_manager = EndScreenManager()
		self.game_engine.add_object(end_screen_manager)
		self.end_screen_manager = end_screen_manager

		player_x_pos = 100

		pipes = PipesManager(player_x_pos)
		pipes.pipe_image = load_image("images/nr2.png")
		end_screen_manager.add_child(pipes)

		self.player = PlayerObject(player_x_pos, 100, 50, 50, (255,0,0), pipes_manager=pipes, on_lose=end_screen_manager.show_game_over)
		end_screen_manager.add_child(self.player)
		player_img = load_image("images/saila1.png")
		self.player.sprite.set_image(player_img)

	def run(self):
		super().run()

	def step_ai(self, dt, jump: bool):
		if jump:
			self.player.jump_requested = True
		super().step(dt)
		state = {}
		state['player_y'] = self.player.y
		state['player_velocity_y'] = self.player.velocity_y
		gap_x, gap_y = self.player.pipes_manager.next_pipe_pos(self.player.x)
		if gap_x is None:
			state['next_pipe_gap_x_distance'] = width - self.player.x
			state['next_pipe_gap_y'] = height / 2
		else:
			state['next_pipe_gap_x_distance'] = gap_x - self.player.x
			state['next_pipe_gap_y'] = gap_y
		return state

	def is_game_over(self) -> bool:
		return getattr(self, 'end_screen_manager', None) is not None and self.end_screen_manager.is_game_over

	def current_score(self) -> int:
		pm = self.player.pipes_manager if hasattr(self, 'player') else None
		return pm.score if pm is not None else 0

	def reset_game(self):
		if hasattr(self, 'end_screen_manager') and self.end_screen_manager is not None:
			self.end_screen_manager.is_game_over = False
			self.end_screen_manager.set_enable_children(True)
		if hasattr(self, 'player') and self.player is not None:
			self.player.x = 100
			self.player.y = 100
			self.player.velocity_y = 0
		pm = self.player.pipes_manager if hasattr(self, 'player') else None
		if pm:
			pm.reset()
