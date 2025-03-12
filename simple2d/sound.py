from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Literal, Union

import numpy as np
from numpy.typing import NDArray

from simple2d import DTYPE


class Note:
  """Class for notes."""
  def __init__(
    self,
    note: str,
    signature: Union[Literal['b'], Literal['#'], Literal[''], None],
    octave: int
  ) -> None:
    assert note in ('A', 'B', 'C', 'D', 'E', 'F', 'G'), f'{note} is not a valid note'
    assert signature in ('b', '#', '', None), f'{signature} is not a valid signature'

    self.note = note
    self.signature = signature if signature else ''
    self.octave = octave
    if self.note in ('A', 'B'): self.octave += 1

  @staticmethod
  def from_str(s: str) -> Note:
    """create a Note from a string.

    Args:
        s (str): String representation of a Note

    Returns:
        Note: Matching Note object
    """
    l = list(
      filter(
        lambda x: x != '',
        re.split(
          r'(^[A-G])([0-9])(#|b)?$',
          s.strip(),
        )
      )
    )
    assert len(l) == 3, f'{s} is not a valid String representation of a Note'
    _note, _octave, _signature = l
    return Note(_note, _signature, int(_octave))  # type: ignore[arg-type]

  @property
  def frequency(self) -> float:
    return self.to_frequency()

  @property
  def index(self) -> float:
    return self.to_index()

  def to_index(self) -> int:
    """Returns the index of the note.

    Raises:
        ValueError: Not a valid note

    Returns:
        int: Index of the note
    """
    match self.note:
      case 'A': idx = 1
      case 'B': idx = 3
      case 'C': idx = 4
      case 'D': idx = 6
      case 'E': idx = 8
      case 'F': idx = 9
      case 'G': idx = 11
      case _: raise ValueError(f'{self.note} is not a valid note')

    match self.signature:
      case 'b':
        if self.note not in ['F', 'C']: idx -= 1
      case '#':
        if self.note not in ['E', 'B']: idx += 1
      case _: pass
    return idx + (self.octave - 1) * 12

  def to_frequency(self) -> float:
    """Returns the frequency of the note.

    Returns:
        float: Frequency of the note
    """
    return 2**((self.to_index()-49)/12) * 440

class SoundGen:
  """Class for generating sounds."""
  def __init__(
    self,
    duration:float,
    notes:Union[list[Note], Path, str, Note]=Note('A', '', 4),
    sound_effect: Union[SoundEffect, None]=None,
    sr:int=44100,
  ) -> None:
    self.duration = duration
    self.sr = sr
    self.sound = self.sine
    self.effect = sound_effect if sound_effect else SoundEffect()

    if isinstance(notes, (Path, str)):
      with open(notes, 'r', encoding="UTF-8") as file: raw_sounds = file.read()
      self.notes = [Note.from_str(note) for note in raw_sounds.strip().split('\n')]
    elif isinstance(notes, Note):
      self.notes = [notes]
    elif isinstance(notes, list):
      self.notes = notes
    else:
      raise ValueError('notes must be a list of Notes or a Path to a file with notes')

  def __next__(self) -> NDArray[DTYPE]:
    n = self.notes.pop(0)
    self.notes.append(n)
    return self.effect(self.sound(n, self.duration, self.sr))

  @staticmethod
  def sine(note: Note, duration:float, sr:int=44100) -> NDArray[DTYPE]:
    sound = np.sin(
      np.pi * note.frequency * np.linspace(0, duration, int(duration * sr)),
      dtype=DTYPE,
    )
    return sound

  @staticmethod
  def cosine(note: Note, duration:float, sr:int=44100) -> NDArray[DTYPE]:
    sound = np.cos(
      np.pi * note.frequency * np.linspace(0, duration, int(duration * sr)),
      dtype=DTYPE,
    )
    return sound

  @staticmethod
  def triangle(note: Note, duration: float, sr:int=44100) -> NDArray[DTYPE]: # pylint: disable=unused-argument
    return np.zeros((6,2), dtype=DTYPE)

class SoundEffect:
  """Class for sound effects."""
  def __init__(self,
    sample_rate:int=44100,
    link:Union[SoundEffect, None]=None,
    effect:Callable[..., NDArray[DTYPE]]=lambda x: x,
    kw:Union[dict[str, Any],None]=None
  ) -> None:
    self.sample_rate = link.sample_rate if link is not None else sample_rate  # type: ignore[has-type]
    self.link = link
    self.effect = effect
    self.kw = kw if kw is not None else {}

  def __call__(self, sound: NDArray[DTYPE]) -> NDArray[DTYPE]:
    assert len(sound.shape) == 1 or (len(sound.shape) == 2 and sound.shape[1] == 2), f'sound: array has wrong shape {sound.shape} -> expected shape: (x,) or (x,2)' # pylint: disable=line-too-long
    if self.link is None: return sound
    self.link(sound)
    return self.effect(sound=sound, **self.kw)

  def fade_in(self, fade_duration: float) -> SoundEffect:
    return SoundEffect(
      link=self,
      effect=SoundEffect._fade_in,
      kw={'fade_duration': fade_duration},
    )

  def fade_out(self, fade_duration: float) -> SoundEffect:
    return SoundEffect(
      link=self,
      effect=SoundEffect._fade_out,
      kw={'fade_duration': fade_duration},
    )

  def reverse(self) -> SoundEffect:
    return SoundEffect(
      link=self,
      effect=SoundEffect._reverse,
      kw={},
    )

  def echo(self, delay: int, loudness: float) -> SoundEffect:
    return SoundEffect(
      link=self,
      effect=SoundEffect._echo,
      kw={'delay': delay, 'loudness': loudness, 'sr': self.sample_rate},
    )

  @staticmethod
  def _fade_in(sound: NDArray[DTYPE], fade_duration: float) -> NDArray[DTYPE]:
    fade_in = np.logspace(1e-9, 1, int(fade_duration * len(sound)))
    fade_in = (fade_in - fade_in.min()) / (fade_in.max() - fade_in.min())
    sound[:len(fade_in)] *= fade_in
    return sound

  @staticmethod
  def _fade_out(sound: NDArray[DTYPE], fade_duration: float) -> NDArray[DTYPE]:
    fade_out = -np.logspace(1e-9, 1, int(fade_duration * len(sound)))
    fade_out = (fade_out - fade_out.min()) / (fade_out.max() - fade_out.min())
    sound[-len(fade_out):] *= fade_out
    return sound

  @staticmethod
  def _reverse(sound: NDArray[DTYPE]) -> NDArray[DTYPE]:
    return sound[::-1]

  @staticmethod
  def _echo(sound: NDArray[DTYPE], delay: int, loudness: float, sr: int) -> NDArray[DTYPE]: # delay in ms
    assert delay >= 0, 'delay can not be negative'
    echo = sound * loudness
    arr_delay = int((delay / 1000) * sr)

    for i in range(len(sound) - arr_delay):
      sound[i + arr_delay] += echo[i]
    return sound
