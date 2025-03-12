import numpy as np
import scipy.special as spec  # type: ignore[import-untyped]
from numpy.typing import NDArray

from simple2d import DTYPE


def _rotate_point(rotate: NDArray[DTYPE], point: NDArray[DTYPE], angle: DTYPE) -> NDArray[DTYPE]:
  """Rotates a point around another point by a given angle.

  Args:
      rotate (DTYPE): coordinates of the point to rotate
      point (NDArray[DTYPE]): coordinates of the point to rotate around
      angle (DTYPE): angle in degrees

  Returns:
      numpy.ndarray[DTYPE]: x and y coordinates of the rotated point
  """
  cos_angle = spec.cosdg(angle)
  sin_angle = spec.sindg(angle)

  translated = rotate - point

  return np.array((
    translated[0] * cos_angle - translated[1] * sin_angle + point[0],
    translated[0] * sin_angle + translated[1] * cos_angle + point[1]),
  dtype=DTYPE)


def rotate_point(x: DTYPE, y: DTYPE, point: NDArray[DTYPE], angle: DTYPE) -> tuple[DTYPE, DTYPE]:
  """Rotates a point around another point by a given angle.

  Args:
      x (DTYPE): x-coordinate of the point to rotate
      y (DTYPE): y-coordinate of the point to rotate
      point (NDArray[DTYPE]): coordinates of the point to rotate around
      angle (DTYPE): angle in degrees

  Returns:
      tuple[DTYPE, DTYPE]: x and y coordinates of the rotated point
  """
  cos_angle = spec.cosdg(angle)
  sin_angle = spec.sindg(angle)

  translated_x = x - point[0]
  translated_y = y - point[1]

  rotated_x = translated_x * cos_angle - translated_y * sin_angle
  rotated_y = translated_x * sin_angle + translated_y * cos_angle

  return rotated_x + point[0], rotated_y + point[1]

def closest_point(point: NDArray[DTYPE], points: list[NDArray[DTYPE]]) -> NDArray[DTYPE]:
  """Returns the closest point to the given point from a list of points.

  Args:
      point (NDArray[DTYPE]): reference point
      points (list[NDArray[DTYPE]]): list of points to check

  Raises:
      RuntimeError: List only contains one point

  Returns:
      NDArray[DTYPE]: Point closest to the reference point
  """
  if len(points) == 0: raise RuntimeError('Only one point')
  if len(points) == 1:
    return points[0]
  dist1 = np.linalg.norm(point - points[0])
  dist2 = np.linalg.norm(point - points[1])
  if dist1 < dist2:
    return points[0]
  return points[1]

def line_line_intersection(
  p1: NDArray[DTYPE], v1: NDArray[DTYPE],
  p2: NDArray[DTYPE], v2: NDArray[DTYPE],
) -> NDArray[DTYPE]:
  """Returns the intersection point of two lines.

  Args:
      p1 (NDArray[DTYPE]): point on the first line
      v1 (NDArray[DTYPE]): vector of the first line
      p2 (NDArray[DTYPE]): point on the second line
      v2 (NDArray[DTYPE]): vector of the second line

  Raises:
      RuntimeError: two lines are parallel

  Returns:
      NDArray[DTYPE]: intersection point
  """
  x1, y1 = p1
  x2, y2 = p2
  vx1, vy1 = v1
  vx2, vy2 = v2

  det = vx1 * vy2 - vx2 * vy1
  if det == 0: raise RuntimeError("BUG")
  t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
  it = np.empty(shape=(2,), dtype=DTYPE)
  it[0] = x1 + t * vx1; it[1] = y1 + t * vy1
  return it

def line_circle_intersection(
  circle_center: NDArray[DTYPE],
  circle_radius: DTYPE,
  linepoint: NDArray[DTYPE],
  direction_vector: NDArray[DTYPE],
) -> list[NDArray[DTYPE]]:
  """Returns the intersection points of a line and a circle.

  Args:
      circle_center (NDArray[DTYPE]): center of the circle
      circle_radius (DTYPE): radius of the circle
      linepoint (NDArray[DTYPE]): point on the line
      direction_vector (NDArray[DTYPE]): vector of the line

  Raises:
      RuntimeError: when circle and line do not intersect

  Returns:
      list[NDArray[DTYPE]]: intersection points
  """
  a = direction_vector[0]**2 + direction_vector[1]**2
  b = 2 * (direction_vector[0] * (linepoint[0] - circle_center[0]) + direction_vector[1] * (linepoint[1] - circle_center[1]))
  c = (linepoint[0] - circle_center[0])**2 + (linepoint[1] - circle_center[1])**2 - circle_radius**2

  discriminant = b**2 - 4*a*c

  if discriminant < 0:
    print(f'{circle_center=}')
    print(f'{circle_radius=}')
    print(f'{linepoint=}')
    print(f'{direction_vector=}')
    raise RuntimeError('Missing intersection point!')
  if discriminant == 0:
    t = -b / (2*a)
    p = linepoint + t * direction_vector
    return [p]
  t1 = (-b + np.sqrt(discriminant)) / (2*a)
  t2 = (-b - np.sqrt(discriminant)) / (2*a)
  point1 = linepoint + t1 * direction_vector
  point2 = linepoint + t2 * direction_vector
  return [point1, point2]
