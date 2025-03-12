import numpy as np
from simple2d.shapes import Circle, RectangleBorder
from simple2d.handler import get_handler
from simple2d import DTYPE

circle = Circle(150, 150, 9)
border = RectangleBorder(150, 150, 100, 100)
border.angle = 45

handler = get_handler(circle, border)

print(handler.check(circle, border))

circle.y -= (50-circle.radius) * np.sqrt(2)
circle.y = 91.9815

print(circle.pos)

print(handler.check(circle, border))
print(handler.closest_side(circle, border)) # type: ignore
