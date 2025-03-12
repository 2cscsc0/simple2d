from simple2d.shapes import Circle, CircleBorder, RectangleBorder
from simple2d.space import Space
from simple2d.render import SpaceRenderer
from simple2d.sound import SoundGen, Note, SoundEffect

from tqdm import trange
import numpy as np

def main() -> None:
  frame_rate = 30.0
  c = Circle(120, 150, 5)
  cb = CircleBorder(150, 150, 150) # RectangleBorder(150, 150, 100, 100)
  cb.angle = 45
  s = Space(1/(500), sound=True)
  s.add_body(c)
  s.add_body(cb)
  sr = SpaceRenderer(s, step_size=100, watermark='TEST')
  notes = [Note('C', '#', 4), Note('A', '', 4), Note('F', 'b', 5)]
  effect = SoundEffect().fade_in(0.2).fade_out(0.4).reverse()
  sg = SoundGen(duration=0.3, notes=notes, sound_effect=effect)
  sr.set_sound(c, cb, sg)
  sr.render(12*30, frame_rate=frame_rate)

if __name__ == "__main__":
  main()
