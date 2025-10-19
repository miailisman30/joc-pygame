"""Simple Pygame boilerplate starter.

Features:
- Window creation with configurable size and caption
- Main loop with fixed FPS via pygame.time.Clock
- Basic event handling (quit, escape)
- Simple movable player rectangle using arrow keys
- FPS counter drawn on the screen

Run:
	python main.py

Install:
	pip install -r requirements.txt

"""


import argparse
import pygame
from pygame import gfxdraw

from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, cast

# Type variable for get_child_of_type so the return type matches the requested class
# Put it here so it's defined before the GameObject class that uses it.
T = TypeVar('T', bound='GameObject')
			
class GameObject():
	# has position, draw recursively
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y
		self.children: list[GameObject] = []
		# reference to parent object (None for root)
		self.parent_obj: GameObject | None = None
		self.enabled = True

	# tell pylance that return type is of type cls_type or None
	def get_child_of_type(self, cls_type: Type[T]) -> Optional[T]:
		"""Return the first child that is an instance of ``cls_type``.

		The return type is generic so static checkers (Pylance/mypy)
		can infer the concrete subclass type when calling e.g.
		get_child_of_type(SpriteObject) -> Optional[SpriteObject].
		"""
		for child in self.children:
			if isinstance(child, cls_type):
				# cast so the type-checker knows this is T, not plain GameObject
				return cast(T, child)
		return None

	def add_child(self, child):
		# set parent reference on child, then add to children list
		child.parent_obj = self
		self.children.append(child)
		
	def draw(self, surface):
		pass

	def update(self, dt):
		pass

	def update_all(self, dt):
		if self.enabled:
			self.update(dt)
			for child in self.children:
				child.update_all(dt)

	def set_enable_children(self, enabled):
		for child in self.children:
			child.enabled = enabled

	def draw_all(self, surface):
		self.draw(surface)
		for child in self.children:
			child.draw_all(surface)

	def get_abs_pos(self):
		"""Return the absolute (x, y) position by walking up the parent chain.

		If this object has no parent (parent_obj is None), its absolute
		position is simply its local (self.x, self.y). Otherwise, add
		this object's local position to its parent's absolute position.
		"""
		if self.parent_obj is None:
			return (self.x, self.y)
		# recursively obtain parent's absolute position
		parent_x, parent_y = self.parent_obj.get_abs_pos()
		return (parent_x + self.x, parent_y + self.y)


class RectObject(GameObject):
	def __init__(self, x=0, y=0, width=50, height=50):
		super().__init__(x, y)
		self.width = width
		self.height = height
		ax, ay = self.get_abs_pos()
		self.rect = pygame.Rect(ax, ay, self.width, self.height)

	def update(self, dt):
		self.rect.topleft = self.get_abs_pos()

	def collide_list(self, other_rects):
		return pygame.Rect.collidelist(self.rect, other_rects)
		
class SpriteObject(RectObject):	
	def __init__(self, x=0, y=0, width=50, height=50, color=(255,0,0)):
		super().__init__(x, y, width, height)
		self.color = color
		self.image = None

	def draw(self, surface):
		if self.image:
			img = pygame.transform.scale(self.image, (int(self.width), int(self.height)))
			surface.blit(img, self.rect.topleft)
		else:
			pygame.draw.rect(surface, self.color, self.rect)

	def set_image(self, surface):
		"""Assign a pygame.Surface to this sprite. The surface will be
		scaled during draw to the sprite's width/height.
		"""
		self.image = surface

def load_image(filename):
	try:
		img = pygame.image.load(filename).convert_alpha()
		return img
	except Exception as e:
		print(f"Error loading image {filename}: {e}")
		return None

class GameEngine:
	def __init__(self, surface):
		self.rootObject = GameObject()
		self.surface = surface
		
	def add_object(self, obj):
		self.rootObject.add_child(obj)

	#draw 
	def draw(self):
		self.rootObject.draw_all(self.surface)
		
	def handle_quit(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.running = False

	def update(self, dt):
		# dt is seconds since last frame
		# keys = pygame.key.get_pressed()
		for obj in self.rootObject.children:
			obj.update_all(dt)

	def stop_game(self):
		self.running = False
		


class GameEnvironment:
	def __init__(self, width=800, height=600, headless=False, fps=60):
		# initialize pygame modules
		pygame.init()
		self.width = width
		self.height = height
		self.fps = fps
		self.clock = pygame.time.Clock()
		self.headless = bool(headless)

		# When running headless we do not create a visible window. Instead
		# we render to an off-screen Surface. This avoids creating an OS
		# window which is useful for automated runs / CI.
		if self.headless:
			# do not call display.set_mode; create a plain Surface instead
			self.screen = pygame.Surface((self.width, self.height))
		else:
			pygame.display.init()
			pygame.display.set_caption("Pygame Boilerplate")
			self.screen = pygame.display.set_mode((self.width, self.height))

		self.bg_color = pygame.Color("#111111")
		# font works without a display surface as long as pygame.font is initialized
		self.font = pygame.font.Font(None, 24)

		# Game engine
		self.game_engine = GameEngine(self.screen)
		

	def draw_fps(self):
		fps_text = f"FPS: {int(self.clock.get_fps())}"
		surf = self.font.render(fps_text, True, pygame.Color("white"))
		self.screen.blit(surf, (8, 8))

	def draw(self):
		self.screen.fill(self.bg_color)
		self.draw_fps()
		self.game_engine.draw()
		# Only flip/update the display when not headless. In headless mode
		# we keep rendering into the off-screen Surface so callers can still
		# inspect it if needed (for screenshots/tests) without creating a window.
		if not self.headless:
			pygame.display.flip()

	def set_headless(self, headless):
		# Allow switching between headless and windowed at runtime. This
		# will (re)create the screen Surface appropriately.
		headless = bool(headless)
		if self.headless == headless:
			return
		self.headless = headless
		if headless:
			# shut down the display subsystem and create an off-screen surface
			try:
				pygame.display.quit()
			except Exception:
				pass
			self.screen = pygame.Surface((self.width, self.height))
		else:
			# re-init display and create a visible window
			pygame.display.init()
			pygame.display.set_caption("Pygame Boilerplate")
			self.screen = pygame.display.set_mode((self.width, self.height))
		# update GameEngine surface reference
		self.game_engine.surface = self.screen

	def step(self, dt):
		self.game_engine.update(dt)
		self.draw()

	def run(self):
		self.game_engine.running = True
		while self.game_engine.running:
			self.game_engine.handle_quit()
			dt = self.clock.tick(self.fps) / 1000.0  # seconds
			self.step(dt)
		pygame.quit()




