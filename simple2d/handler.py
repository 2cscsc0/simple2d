from __future__ import annotations

from abc import abstractmethod
from typing import Any, Union

import numpy as np
import scipy.special as spec # type: ignore[import-untyped]

from simple2d import DTYPE, helpers, shapes
from simple2d.sound import SoundGen


def get_handler(X: Union[shapes.Body, shapes.Border], Y: Union[shapes.Body, shapes.Border]) -> CollisionHandler:
  """ Returns the correct collision handler for the given Bodies/Border objects.

  Args:
      X (Union[shapes.Body, shapes.Border]): first object
      Y (Union[shapes.Body, shapes.Border]): second object

  Raises:
      NotImplementedError: in case the handler for the given types is not implemented

  Returns:
      CollsionHandler: the correct CollisionHandler
  """
  match type(X), type(Y):
    case (shapes.Circle, shapes.CircleBorder) | (shapes.CircleBorder, shapes.Circle):
      return CircleCircleBorderHandler((X.collision_type, Y.collision_type))
    case (shapes.Circle, shapes.Circle):
      return CircleCircleHandler((X.collision_type, Y.collision_type))
    case (shapes.Circle, shapes.RectangleBorder) | (shapes.RectangleBorder, shapes.Circle):
      return CircleRectangleBorderHandler((X.collision_type, Y.collision_type))
    case (shapes.Rectangle, shapes.Rectangle):
      raise NotImplementedError()
    case (shapes.Rectangle, shapes.Circle) | (shapes.Circle, shapes.Rectangle):
      raise NotImplementedError()
    case (shapes.Rectangle, shapes.RectangleBorder) | (shapes.RectangleBorder, shapes.Rectangle):
      raise NotImplementedError()
    case (shapes.Rectangle, shapes.CircleBorder) | (shapes.CircleBorder, shapes.Rectangle):
      raise NotImplementedError()

  raise NotImplementedError()

class CollisionHandler:
  """Abstract class for collision handlers."""
  def __init__(self, types):
    self.types = types
    self.sound_gen = SoundGen(0.0)

  def checktypes(
    self,
    type1: Union[int, type[shapes.Body], type[shapes.Border]],
    type2: Union[int, type[shapes.Body], type[shapes.Border]],
  ) -> bool:
    """Returns True if the given types match the handler's types.

    Args:
        type1 (Union[int, type[shapes.Body], type[shapes.Border]]): first type
        type2 (Union[int, type[shapes.Body], type[shapes.Border]]): second type

    Returns:
        bool: True if the given types match the handler's types
    """
    return self.types in ((type1, type2), (type2, type1))

  def post(self, X, Y) -> bool: return True # pylint: disable=unused-argument

  @abstractmethod
  def check(self, X: Any, Y: Any) -> bool: pass

  @abstractmethod
  def resolve(self, X: Any, Y: Any, dt: DTYPE) -> bool: pass

class CircleCircleHandler(CollisionHandler):
  """Collision handler for two Circle objects."""
  def __init__(self, types=(shapes.Circle, shapes.Circle)):
    super().__init__(types)

  def check(self, X: shapes.Circle, Y: shapes.Circle) -> bool:
    """Checks if the two Circle objects are colliding.

    Args:
        X (shapes.Circle): first Circle object
        Y (shapes.Circle): second Circle object

    Returns:
        bool: Returns True if the two Circle objects are colliding
    """
    return bool(np.linalg.norm(X.pos - Y.pos) < X.radius + Y.radius)

  def resolve(self, X: shapes.Circle, Y: shapes.Circle, dt: DTYPE) -> bool:
    """Resolves the collision between two Circle objects.

    Args:
        X (shapes.Circle): first Circle object
        Y (shapes.Circle): second Circle object
        dt (DTYPE): size of timestep used for the simulation

    Raises:
        RuntimeError: when a bug occurs

    Returns:
        bool: returns the output off the self.post method
    """
    pre_c1_loc = X.pos - (X.velocity * dt)
    pre_c2_loc = Y.pos - (Y.velocity * dt)

    g1 = pre_c1_loc[0] - pre_c2_loc[0]
    g2 = pre_c1_loc[1] - pre_c2_loc[1]
    h1 = (X.vx - Y.vx) * dt
    h2 = (X.vy - Y.vy) * dt
    r = X.radius + Y.radius

    a = h1**2 + h2**2
    b = ((g1*h1) + (g2*h2))*2
    c = g1**2 + g2**2 - r**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0: raise RuntimeError('BUGGG')

    if discriminant == 0:
      t = (-b) / (2*a)
    else:
      t1 = (-b + np.sqrt(discriminant)) / (2*a)
      t2 = (-b - np.sqrt(discriminant)) / (2*a)
      t = t1 if t1 < t2 else t2

    assert t < np.linalg.norm(1), f'percentage moved must be < 0, but is {t}'

    X.pos = pre_c1_loc + (X.velocity * dt) * t
    Y.pos = pre_c2_loc + (Y.velocity * dt) * t

    tmp_v1 = X.velocity - (2*Y.mass/(X.mass+Y.mass)) * (np.dot(X.velocity-Y.velocity, X.pos-Y.pos) / np.linalg.norm(X.pos-Y.pos)**2) * (X.pos-Y.pos)
    tmp_v2 = Y.velocity - (2*X.mass/(Y.mass+X.mass)) * (np.dot(Y.velocity-X.velocity, Y.pos-X.pos) / np.linalg.norm(Y.pos-X.pos)**2) * (Y.pos-X.pos)

    tmp_v1[np.isnan(tmp_v1) | np.isinf(tmp_v1)] = DTYPE(0)
    tmp_v2[np.isnan(tmp_v2) | np.isinf(tmp_v2)] = DTYPE(0)

    X.velocity = tmp_v1 # * Y.elasticity
    Y.velocity = tmp_v2 # * Y.elasticity

    X.pos += X.velocity * dt * (1-t)
    Y.pos += Y.velocity * dt * (1-t)

    X._collisions += 1
    Y._collisions += 1

    return self.post(X, Y)

class CircleCircleBorderHandler(CollisionHandler):
  """Collision handler for a Circle and a CircleBorder object."""
  def __init__(self, types=(shapes.Circle, shapes.CircleBorder)):
    super().__init__(types)

  def check(self, X: shapes.Circle, Y: shapes.CircleBorder) -> bool:
    """Checks if the Circle and CircleBorder objects are colliding.

    Args:
        X (shapes.Circle): Circle object
        Y (shapes.CircleBorder): CircleBorder object

    Returns:
        bool: Returns True if the Circle and CircleBorder objects are colliding
    """
    return bool(np.linalg.norm(X.pos - Y.pos) >= Y.radius - X.radius)

  def resolve(self, X: shapes.Circle, Y: shapes.CircleBorder, dt: DTYPE) -> bool:
    """Resolves the collision between a Circle and a CircleBorder object.

    Args:
        X (shapes.Circle): Circle object
        Y (shapes.CircleBorder): CircleBorder object
        dt (DTYPE): size of timestep used for the simulation

    Returns:
        bool: returns the output off the self.post method
    """
    pre_circle_loc = X.pos - (X.velocity * dt)

    X.pos = helpers.closest_point(
      X.pos,
      helpers.line_circle_intersection(Y.pos, Y.radius - X.radius, X.pos, X.velocity),
    )

    p_mov = np.linalg.norm(pre_circle_loc - X.pos) / np.linalg.norm(X.velocity * dt)

    assert p_mov < np.linalg.norm(1), f'percentage moved must be < 0, but is {p_mov}'

    mirror_vec = Y.pos - X.pos
    mirror_vec = mirror_vec / np.linalg.norm(mirror_vec)

    X.velocity -= 2 * np.dot(X.velocity, mirror_vec) * mirror_vec
    #X.velocity *= X.elasticity

    X.pos += (1-p_mov) * X.velocity * dt

    X._collisions += 1
    Y._collisions += 1

    return self.post(X, Y)

class CircleRectangleBorderHandler(CollisionHandler):
  """Collision handler for a Circle and a RectangleBorder object."""
  def __init__(self, types=(shapes.Circle, shapes.RectangleBorder)):
    super().__init__(types)

  def closest_side(self, X: shapes.Circle, Y: shapes.RectangleBorder) -> tuple[np.ndarray, np.ndarray]:
    """Returns the closest side of the RectangleBorder object to the Circle object.

    Args:
        X (shapes.Circle): Circle object
        Y (shapes.RectangleBorder): CircleBorder object

    Returns:
        tuple[np.ndarray, np.ndarray]: Vector and point of the closest side of the RectangleBorder object to the Circle object
    """
    circ = X.pos - Y.pos
    circ = np.array([
      circ[0]*spec.cosdg(-Y.angle) - circ[1]*spec.sindg(-Y.angle),
      circ[0]*spec.sindg(-Y.angle) + circ[1]*spec.cosdg(-Y.angle),
    ], dtype=DTYPE)

    distances = np.array([
      np.abs(-Y.width/2 - circ[0]),
      Y.width/2 - circ[0],
      np.abs(-Y.height/2 - circ[1]),
      Y.height/2 - circ[1],
    ], dtype=DTYPE)

    closest_side = np.argmin(distances)

    if closest_side == 0:
      return (
        np.array([spec.cosdg(Y.angle + 90), spec.sindg(Y.angle + 90)], dtype=DTYPE),
        np.array(
          helpers.rotate_point(Y.x - Y.width/2 + X.radius, Y.y - Y.height/2 + X.radius, Y.pos, Y.angle),
          dtype=DTYPE,
        ),
      )
    if closest_side == 1:
      return (
        np.array([spec.cosdg(Y.angle + 90), spec.sindg(Y.angle + 90)], dtype=DTYPE),
        np.array(
          helpers.rotate_point(Y.x + Y.width/2 - X.radius, Y.y + Y.height/2 - X.radius, Y.pos, Y.angle),
          dtype=DTYPE,
        ),
      )
    if closest_side == 2:
      return (
        np.array([spec.cosdg(Y.angle), spec.sindg(Y.angle)], dtype=DTYPE),
        np.array(
          helpers.rotate_point(Y.x - Y.width/2 + X.radius, Y.y - Y.height/2 + X.radius, Y.pos, Y.angle),
          dtype=DTYPE,
        ),
      )
    if closest_side == 3:
      return (
        np.array([spec.cosdg(Y.angle), spec.sindg(Y.angle)], dtype=DTYPE),
        np.array(
          helpers.rotate_point(Y.x + Y.width/2 - X.radius, Y.y + Y.height/2 - X.radius, Y.pos, Y.angle),
          dtype=DTYPE,
        ),
      )

    raise RuntimeError("BUGG")

  def check(self, X: shapes.Circle, Y: shapes.RectangleBorder) -> bool:
    """Checks if the Circle and RectangleBorder objects are colliding.

    Args:
        X (shapes.Circle): Circle object
        Y (shapes.RectangleBorder): RectangleBorder object

    Returns:
        bool: Returns True if the Circle and RectangleBorder objects are colliding
    """
    circ = X.pos - Y.pos
    circ = np.array([
      circ[0]*spec.cosdg(-Y.angle) - circ[1]*spec.sindg(-Y.angle),
      circ[0]*spec.sindg(-Y.angle) + circ[1]*spec.cosdg(-Y.angle),
    ], dtype=DTYPE)

    if circ[0] + X.radius > Y.width/2:
      return True
    if circ[1] + X.radius > Y.height/2:
      return True
    if circ[0] - X.radius < -Y.width/2:
      return True
    if circ[1] - X.radius < -Y.height/2:
      return True

    return False

  def resolve(self, X: shapes.Circle, Y: shapes.RectangleBorder, dt: DTYPE) -> bool:
    """Resolves the collision between a Circle and a RectangleBorder object.

    Args:
        X (shapes.Circle): Circle object
        Y (shapes.RectangleBorder): RectangleBorder object
        dt (DTYPE): size of timestep used for the simulation

    Returns:
        bool: returns the output off the self.post method
    """
    pre_circle_loc = X.pos - (X.velocity * dt)

    mirror_vec, mirror_line_point = self.closest_side(X, Y)

    X.pos = helpers.line_line_intersection(
      X.pos,
      X.velocity,
      mirror_line_point,
      mirror_vec,
    )

    p_mov = np.linalg.norm(pre_circle_loc - X.pos)/np.linalg.norm(X.velocity * dt)

    assert p_mov < np.linalg.norm(1), f'percentage moved must be < 0, but is {p_mov}'

    orth_mirror_vec = mirror_vec
    orth_mirror_vec[0], orth_mirror_vec[1] = -orth_mirror_vec[1], orth_mirror_vec[0]

    X.velocity -= 2 * np.dot(X.velocity, orth_mirror_vec) * orth_mirror_vec
    # X.velocity *= X.elasticity

    X.pos += (1-p_mov) * X.velocity * dt

    X._collisions += 1
    Y._collisions += 1

    return self.post(X, Y)
