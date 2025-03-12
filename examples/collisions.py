from simple2d.shapes import Rectangle
from simple2d.space import Space
from simple2d.render import SpaceRenderer

def main() -> None:
  r1 = Rectangle(10, 10, 10, 10)
  r1.velocity = 10, 0
  r2 = Rectangle(30, 10, 10, 10)
  r2.velocity = -10, 0

  space = Space(1/120)
  space.gravity = 0, 0

  space.add_body(r1)
  space.add_body(r2)

  sr = SpaceRenderer(space)

  sr.bad_live_render()

if __name__ == "__main__":
  main()
