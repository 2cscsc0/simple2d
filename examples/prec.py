from simple2d.shapes import Circle, CircleBorder, RectangleBorder
from simple2d.space import Space
import numpy as np

from tqdm import trange


def main() -> None:
  duration = 75
  step_size = 5_000
  c = Circle(305, 305, 10)
  c.velocity = (35, 65)

  cb = CircleBorder(300, 300, 200)
  #cb = RectangleBorder(25, 25, 350, 350)
  space = Space(1/(step_size))
  print(f'SPACE-CONFIG: gravity={space.gravity} | dt={space.dt} | {space.gravity * space.dt}')
  #space.gravity = (0.0, 0.0)

  space.add_body(c)
  space.add_body(cb)

  steps = step_size * duration

  print('STARTINGPOINT')
  scx, scy, scvx, scvy, scv = c.x, c.y, c.vx, c.vy, np.linalg.norm(c.velocity)
  senergy = scv**2 * 0.5 + 10.0 * c.y
  print(f'CIRCLE: x={c.x:<14.6f} | y={c.y:<14.6f} | vx={c.vx:<14.6f} | vy={c.vy:<14.6f} | len(v)={np.linalg.norm(c.velocity):<14.6f} | energy={senergy:<14.6f}')

  with space.reverse():
    for _ in trange(steps):
      space.step()

  menergy = np.linalg.norm(c.velocity)**2 * 0.5 + 10.0 * c.y
  print('REVERSE')
  print(f'CIRCLE: x={c.x:<14.6f} | y={c.y:<14.6f} | vx={c.vx:<14.6f} | vy={c.vy:<14.6f} | len(v)={np.linalg.norm(c.velocity):<14.6f} | energy={menergy:<14.6f}')

  for _ in trange(steps):
    space.step()

  ecx, ecy, ecvx, ecvy, ecv = c.x, c.y, c.vx, c.vy, np.linalg.norm(c.velocity)
  eenergy = ecv**2 * 0.5 + 10.0 * c.y
  print('ENDPOINT')
  print(f'CIRCLE: x={c.x:<14.6f} | y={c.y:<14.6f} | vx={c.vx:<14.6f} | vy={c.vy:<14.6f} | len(v)={np.linalg.norm(c.velocity):<14.6f} | energy={eenergy:<14.6f}')

  print(f'COLLISIONS: {c._collisions}')
  print(f'VELOCITY  : {scv:9.4f} -> {ecv:9.4f}: {(cve:=abs((scv - ecv) / scv * 100)):7.4f}%')
  print(f'VELOCITY_X: {scvx:9.4f} -> {ecvx:9.4f}: {(cvxe:=abs((scvx - ecvx) / scvx * 100)):7.4f}%')
  print(f'VELOCITY_Y: {scvy:9.4f} -> {ecvy:9.4f}: {(cvye:=abs((scvy - ecvy) / scvy * 100)):7.4f}%')
  print(f'POSITION_X: {scx:9.4f} -> {ecx:9.4f}: {(cxe:=abs((scx - ecx) / scx * 100)):7.4f}%')
  print(f'POSITION_Y: {scy:9.4f} -> {ecy:9.4f}: {(cye:=abs((scy - ecy) / scy * 100)):7.4f}%')
  print(f'ENERGY    : {senergy:9.4f} -> {eenergy:9.4f}: {(cee:=abs((senergy - eenergy) / senergy * 100)):7.4f}%')
  print(f'MEAN_ERROR: {(cve + cvxe + cvye + cxe + cye + cee) / 6:7.4f}%')

if __name__ == "__main__":
  main()
