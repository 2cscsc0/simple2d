from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

from simple2d import DTYPE


class Body:
  """Base class for all moving objects."""
  def __init__(self, x: float, y: float):
    self._pos: NDArray[DTYPE] = np.array((x, y), dtype=DTYPE)
    self._velocity: NDArray[DTYPE] = np.array((0, 0), dtype=DTYPE)
    self._mass: DTYPE = DTYPE(1)
    self._angle: DTYPE = DTYPE(0)
    self._angular_velocity: DTYPE = DTYPE(0)
    self._collisions: int = 0
    self._collision_type: Union[type[Body], int]
    #self._acceleration: NDArray[DTYPE] = np.array((0, 0), dtype=DTYPE)
    #self._elasticity = DTYPE(1)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(x={self.x:.3f}, y={self.y:.3f}, vx={self.vx:.3f}, vy={self.vx:.3f}, collisiontype={(self.collision_type) if isinstance(self.collision_type, int) else self.collision_type.__name__})' # pylint: disable=line-too-long

  @property
  def x(self) -> DTYPE:
    return self.pos[0]

  @x.setter
  def x(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"x must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._pos[0] = DTYPE(val)

  @property
  def y(self) -> DTYPE:
    return self.pos[1]

  @y.setter
  def y(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"y must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._pos[1] = DTYPE(val)

  @property
  def pos(self) -> NDArray[DTYPE]:
    return self._pos

  @pos.setter
  def pos(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), "pos must be a tuple or list or numpy.ndarray"
    if isinstance(val, np.ndarray): self._pos = val
    else: self._pos = np.array(val, dtype=DTYPE)

  @property
  def velocity(self) -> NDArray[DTYPE]:
    return self._velocity

  @velocity.setter
  def velocity(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), "velocity must be a tuple or list or numpy.ndarray"
    if isinstance(val, np.ndarray): self._velocity = val
    else: self._velocity = np.array(val, dtype=DTYPE)

  @property
  def vx(self) -> DTYPE:
    return self.velocity[0]

  @vx.setter
  def vx(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"vx must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._velocity[0] = DTYPE(val)

  @property
  def vy(self) -> DTYPE:
    return self.velocity[1]

  @vy.setter
  def vy(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"vy must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._velocity[1] = DTYPE(val)

  @property
  def mass(self) -> DTYPE:
    return self._mass

  @mass.setter
  def mass(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"mass must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._mass = DTYPE(val)

  @property
  def angle(self) -> DTYPE:
    return self._angle

  @angle.setter
  def angle(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"angle must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._angle = DTYPE(val)

  @property
  def angular_velocity(self) -> DTYPE:
    return self._angular_velocity

  @angular_velocity.setter
  def angular_velocity(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"angular_velocity must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._angular_velocity = DTYPE(val)

  @property
  def collision_type(self) -> Union[type[Body], int]:
    return self._collision_type

  @collision_type.setter
  def collision_type(self, val) -> None:
    assert isinstance(val, int), "collision_type must be an int"
    self._collision_type = val

class Circle(Body):
  """A moving circle object."""
  def __init__(self, x: float, y: float, r: float) -> None:
    super().__init__(x, y)
    self._radius: DTYPE = DTYPE(r)
    self._collision_type: Union[type[Body], int] = Circle

  @property
  def radius(self) -> DTYPE:
    return self._radius

  @radius.setter
  def radius(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"radius must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._radius = DTYPE(val)

class Rectangle(Body):
  """A moving Rectangle object."""
  def __init__(self, x: float, y: float, width: float, height: float) -> None:
    super().__init__(x, y)
    self._width: DTYPE = DTYPE(width)
    self._height: DTYPE = DTYPE(height)
    self._collision_type: Union[type[Body], int] = Rectangle

  @property
  def width(self) -> DTYPE:
    return self._width

  @width.setter
  def width(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"radius must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._width = DTYPE(val)

  @property
  def height(self) -> DTYPE:
    return self._height

  @height.setter
  def height(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"radius must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._height = DTYPE(val)

class Border:
  """Base class for all static objects."""
  def __init__(self, x: float, y: float):
    self._pos: NDArray[DTYPE] = np.array((x, y), dtype=DTYPE)
    self._angle: DTYPE = DTYPE(0)
    self._collisions: int = 0
    self._collision_type: Union[type[Border], int]

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(x={self.x:.3f}, y={self.y:.3f})'

  @property
  def x(self) -> DTYPE:
    return self.pos[0]

  @x.setter
  def x(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"x must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._pos[0] = DTYPE(val)

  @property
  def y(self) -> DTYPE:
    return self.pos[1]

  @y.setter
  def y(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"y must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._pos[1] = DTYPE(val)

  @property
  def pos(self) -> NDArray[DTYPE]:
    return self._pos

  @pos.setter
  def pos(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), "pos must be a tuple or list or numpy.ndarray"
    if isinstance(val, np.ndarray): self._pos = val
    else: self._pos = np.ndarray(val, dtype=DTYPE)

  @property
  def angle(self) -> DTYPE:
    return self._angle

  @angle.setter
  def angle(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"angle must be an int or float or {str(DTYPE)}"  # type: ignore[misc]
    self._angle = DTYPE(val)

  @property
  def collision_type(self) -> Union[type[Border], int]:
    return self._collision_type

  @collision_type.setter
  def collision_type(self, val) -> None:
    assert isinstance(val, int), "collision_type must be an int"
    self._collision_type = val

class CircleBorder(Border):
  """A static circle object."""
  def __init__(self, x: float, y: float, r: float):
    super().__init__(x, y)
    self._radius: DTYPE = DTYPE(r)
    self._collision_type: Union[type[Border], int] = CircleBorder

  @property
  def radius(self) -> DTYPE:
    return self._radius

  @radius.setter
  def radius(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"radius must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._radius = DTYPE(val)

class RectangleBorder(Border):
  """A static rectangle object."""
  def __init__(self, x: float, y: float, w: float, h: float):
    super().__init__(x,y)
    self._dims = np.array((w, h), dtype=DTYPE)
    self._collision_type: Union[type[Border], int] = RectangleBorder

  @property
  def dims(self) -> np.ndarray:
    return self._dims

  @dims.setter
  def dims(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), "dims must be a tuple or list or numpy.ndarray"
    if isinstance(val, np.ndarray): self._dims = val
    else: self._dims = np.array(val, dtype=DTYPE)

  @property
  def width(self) -> DTYPE:
    return self.dims[0]

  @width.setter
  def width(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"width must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._dims[0] = DTYPE(val)

  @property
  def height(self) -> DTYPE:
    return self._dims[1]

  @height.setter
  def height(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), f"height must be an int or float or {str(DTYPE)}" # type: ignore[misc]
    self._dims[1] = DTYPE(val)
