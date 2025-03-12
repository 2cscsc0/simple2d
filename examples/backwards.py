from simple2d.render import SpaceRenderer
from simple2d.shapes import Circle, RectangleBorder
from simple2d.sound import Note, SoundEffect, SoundGen
from simple2d.space import Space
from simple2d import DTYPE

import numpy as np
from numpy.typing import NDArray

from tqdm import trange

import random
random.seed(1440)

def circle_points(x: DTYPE, y: DTYPE, r: DTYPE, num_points: int) -> NDArray[DTYPE]:
  theta = np.linspace(0, 2*np.pi, num_points + 1)
  theta = theta[:-1]
  points = np.column_stack((x + r * np.cos(theta), y + r * np.sin(theta)))
  return points

def main() -> None:
  num_circles = 30
  step_size = 1200
  circles = [Circle(0, 0, 10) for _ in range(num_circles)]
  border = RectangleBorder(10, 10, 500, 500)
  space = Space(1/(step_size), sound=True)
  space.gravity = 0,0

  cx, cy = DTYPE(260), DTYPE(260)
  points = circle_points(cx, cy, DTYPE(200), num_circles)
  for c, p in zip(circles, points):

    c.x = p[0]
    c.y = p[1]
    vec = np.array((cx - c.x, cy - c.y), dtype=DTYPE)
    c.vx = (-c.y / np.linalg.norm(vec)) * 16
    c.vy = (c.x / np.linalg.norm(vec)) * 16
    space.add_body(c)
  space.add_body(border)

  sr = SpaceRenderer(space, watermark='WATER', step_size=int(step_size/30))
  effect = SoundEffect().fade_in(0.2).fade_out(0.4)
  sound_gen = SoundGen(0.2, Note('C', '', 4), effect)
  sr.set_sound(circles[0], border, sound_gen)

  steps = step_size * 45

  with space.reverse():
    for _ in (t:=trange(steps)): space.step()

  print(sum([k._collisions for k in space.kinetics + space.statics])/2)
  
  for _ in (t:=trange(steps)): space.step()

  try:
    sr.bad_live_render()
  except KeyboardInterrupt: pass
  return
  sr.render(8 * 30)

if __name__ == "__main__":
  main()
