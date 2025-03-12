from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

from simple2d import DTYPE, shapes
from simple2d.handler import CollisionHandler, get_handler


class Space:
  """Class for the physics space."""
  def __init__(self, dt: float, sound:bool=False):
    self.kinetics: list[shapes.Body] = []
    self.statics: list[shapes.Border] = []
    self._gravity: NDArray[DTYPE] = np.array((0, -10.0), dtype=DTYPE)
    self.dt: DTYPE = DTYPE(dt)
    self.sound = sound
    self.damp: DTYPE = DTYPE(1) # DTYPE(1 - 1e-3)
    self.reversed: bool = False

    self.collision_handlers: list[CollisionHandler] = []

  class Rev:
    """Helper class to reverse the space."""
    def __init__(self, space: Space) -> None: self.space = space
    def __enter__(self): self.rev()
    def __exit__(self, exc_type, exc_value, traceback): self.rev()
    def rev(self) -> None:
      self.space.reversed = not self.space.reversed
      for kin in self.space.kinetics:
        kin.velocity = -kin.velocity
        #kin.elasticity = 1/kin.elasticity

  def reverse(self) -> Rev:
    """Reverse the all velocities in the space.

    Returns:
        Rev: Context manager to reverse the velocities
    """
    return Space.Rev(self)

  @property
  def gravity(self) -> NDArray[DTYPE]:
    return self._gravity

  @gravity.setter
  def gravity(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), "gravity must be a tuple or list or numpy.ndarray"
    if isinstance(val, np.ndarray): self._gravity = val
    else: self._gravity = np.array(val, dtype=DTYPE)

  def add_body(self, body: Union[shapes.Body, shapes.Border]) -> None:
    """Add a body to the space.

    Args:
        body (Union[shapes.Body, shapes.Border]): Body to add
    """
    if body in self.kinetics or body in self.statics: return

    if isinstance(body, shapes.Body):
      for stat in self.statics:
        handler = self.get_collision_handler(body, stat)
        if handler is None:
          self.add_collision_handler(body, stat)

    for kin in self.kinetics:
      handler = self.get_collision_handler(body, kin)
      if handler is None:
        self.add_collision_handler(body, kin)

    match type(body):
      case shapes.Circle | shapes.Rectangle:
        self.kinetics.append(body)  # type: ignore[arg-type]
      case shapes.CircleBorder | shapes.RectangleBorder:
        self.statics.append(body)  # type: ignore[arg-type]

  def remove_body(self, body: Union[shapes.Body, shapes.Border]) -> None:
    """Remove a body from the space.

    Args:
        body (Union[shapes.Body, shapes.Border]): Body to remove
    """
    if body not in self.kinetics and body not in self.statics: return
    match type(body):
      case shapes.Circle:
        self.kinetics.remove(body)  # type: ignore[arg-type]
      case shapes.CircleBorder:
        self.statics.remove(body)  # type: ignore[arg-type]

  def add_collision_handler(self, x: Union[shapes.Body, shapes.Border], y: Union[shapes.Body, shapes.Border]) -> CollisionHandler:
    """Add a collision handler to the space for given objects.

    Args:
        x (Union[shapes.Body, shapes.Border]): first object
        y (Union[shapes.Body, shapes.Border]): second object

    Returns:
        CollisionHandler: Matching CollisionHandler
    """
    handler = get_handler(x, y)
    if handler not in self.collision_handlers:
      self.collision_handlers.append(handler)
    return handler

  def get_collision_handler(
    self, x: Union[shapes.Body,  shapes.Border],
    y: Union[shapes.Body, shapes.Border],
  ) -> CollisionHandler | None:
    """Returns a matching collision handler for the given objects.

    Args:
        x (Union[shapes.Body,  shapes.Border]): first object
        y (Union[shapes.Body, shapes.Border]): second object

    Returns:
        CollisionHandler | None: Matching CollisionHandler if found
    """
    for handler in self.collision_handlers:
      if handler.checktypes(x.collision_type, y.collision_type):
        return handler
      if handler.checktypes(type(y), x.collision_type):
        return handler
      if handler.checktypes(type(x), y.collision_type):
        return handler
    return None

  def step(self) -> list[CollisionHandler]:
    """Simulate one step in the space.

    Raises:
        RuntimeError: Unknown Handler
        RuntimeError: Unknown Handler

    Returns:
        list[CollisionHandler]: list of all used collision handlers
    """
    collisions: list[tuple[shapes.Body | shapes.Border, shapes.Body | shapes.Border, CollisionHandler]] = []
    for kin in self.kinetics:
      k1 = self.gravity * self.dt
      k2 = (self.gravity + (k1 * 0.5)) * self.dt
      k3 = (self.gravity + (k2 * 0.5)) * self.dt
      k4 = (self.gravity + (k3 * 0.5)) * self.dt

      kin.velocity += ((k1 + 2*k2 + 2*k3 + k4) / 6.0) * self.damp

      kin.pos += kin.velocity * self.dt

      kin.angle += kin.angular_velocity * self.dt

    for i, kin in enumerate(self.kinetics):
      for o_kin in self.kinetics[i+1:]:
        handler = self.get_collision_handler(kin, o_kin)
        if handler is None: raise RuntimeError("Unknown Handler")

        if handler.check(kin, o_kin):
          collisions.append((kin, o_kin, handler))
          # if self.sound: collisions.append(handler)
          # handler.resolve(kin, o_kin, self.dt)

      for stat in self.statics:
        handler = self.get_collision_handler(kin, stat)
        if handler is None: raise RuntimeError("Unknown Handler")

        if handler.check(kin, stat):
          collisions.append((kin, stat, handler))
          # if self.sound: collisions.append(handler)
          # handler.resolve(kin, stat, self.dt)

    if self.reversed: collisions = collisions[::-1]

    for collision in collisions:
      collision[2].resolve(collision[0], collision[1], self.dt)

    return [e[-1] for e in collisions] if self.sound else []
