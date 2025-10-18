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
			
class GameObject():
	# has position, draw recursively
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y
		self.children = []
		# reference to parent object (None for root)
		self.parent_obj = None
		self.enabled = True

	def get_child_of_type(self, cls_type):
		for child in self.children:
			if isinstance(child, cls_type):
				return child
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


class GameEngine:
	def __init__(self, surface):
		self.rootObject = GameObject()
		self.surface = surface
		
	def add_object(self, obj):
		self.rootObject.add_child(obj)

	#draw 
	def draw(self):
		self.rootObject.draw_all(self.surface)
		
	def handle_events(self):
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
		


class GameWindow:
	def __init__(self, width=800, height=600, fps=60):
		pygame.init()
		pygame.display.set_caption("Pygame Boilerplate")
		self.width = width
		self.height = height
		self.fps = fps
		self.screen = pygame.display.set_mode((self.width, self.height))
		self.clock = pygame.time.Clock()
		self.running = False

		# Simple game stats
		self.bg_color = pygame.Color("#111111")
		self.font = pygame.font.Font(None, 24)

		# Game engine
		self.game_engine = GameEngine(self.screen)

	def draw_fps(self):
		fps_text = f"FPS: {int(self.clock.get_fps())}"
		surf = self.font.render(fps_text, True, pygame.Color("white"))
		self.screen.blit(surf, (8, 8))

	def draw(self):
		self.screen.fill(self.bg_color)

		# draw game engine objects
		self.game_engine.draw()  # Uncomment if using GameEngine

		# FPS
		self.draw_fps()

		pygame.display.flip()

	def run(self):
		self.game_engine.running = True
		while self.game_engine.running:
			dt = self.clock.tick(self.fps) / 1000.0  # seconds
			self.game_engine.handle_events()
			self.game_engine.update(dt)
			self.draw()

		pygame.quit()




