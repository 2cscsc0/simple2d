import platform
import random
import subprocess
import time
from abc import abstractmethod
from pathlib import Path
from typing import Union, cast

import ffmpeg  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile  # type: ignore[import-untyped]
from tqdm import tqdm, trange

from simple2d import DTYPE, helpers, shapes
from simple2d.handler import CollisionHandler
from simple2d.shapes import Body, Border
from simple2d.space import Space
from simple2d.sound import SoundGen


class Renderer:
  """Renderer class used to render simulations."""
  def __init__(self, width: int, height: int, scale: float) -> None:
    self.width = width
    self.height = height
    self.scale = DTYPE(scale)

  @abstractmethod
  def render(self, frame_count: int, frame_rate:float=30.0, path: str='output') -> None: pass

  def hex_to_tuple(self, clr: bytes):
    """Converts a bytes-hex-string to a tuple of integers

    Args:
        clr (bytes): _description_

    Returns:
        _type_: _description_
    """
    return (
      int(clr[1:3], 16),
      int(clr[3:5], 16),
      int(clr[5:], 16),
    )

  def flipy(self, height, y) -> DTYPE:
    """Flips the y-coordinate."""
    return height - y

class SpaceRenderer(Renderer):
  """Renderer class to render a space."""
  def __init__(self, space: Space, watermark:str='', step_size:int=1):
    super().__init__(0, 0, 3.0)
    self.space = space
    self.width, self.height = self._dims()
    self.step_size = step_size
    self.watermark = watermark

    self.f_palette: list[bytes] = [
      b'#64E619',
      b'#E6B419',
      b'#E6DA19',
      b'#19B8E6',
      b'#E6191C',
      b'#E6DA19',
      b'#1D19E6',
    ]
    self.border_clr: bytes = b'#B4B4B4'
    self.watermark_clr: bytes = b'#C3C3C3'
    self.background_clr: bytes = b'#000000'

  def _dims(self):
    w = h = 0
    for _stat in self.space.statics:
      match type(_stat):
        case shapes.CircleBorder:
          stat = cast(shapes.CircleBorder, _stat)
          w = max(w, stat.x * 2)
          h = max(h, stat.y * 2)
        case shapes.RectangleBorder:
          stat = cast(shapes.RectangleBorder, _stat)
          w = max(w, stat.width/2 + stat.x)
          h = max(h, stat.height/2 + stat.y)
    if w == 0: w = 100
    if h == 0: h = 100
    return int(w), int(h)

  def set_sound(self, x: Union[Body, Border], y: Union[Body, Border], sound_gen: SoundGen) -> None:
    """Sets the sound generator for a specific collision handler.

    Args:
        x (Union[Body, Border]): first object
        y (Union[Body, Border]): second object
        sound_gen (SoundGen): desired sound generator
    """
    ch = self.space.get_collision_handler(x, y)
    if ch: ch.sound_gen = sound_gen

  def render_current_frame(self, watermark: Union[tuple[str, int, int], None]=None) -> Image.Image:
    """Render current frame.

    Args:
        watermark (Union[tuple[str, int, int], None], optional): Optional watermark. Defaults to None.

    Returns:
        Image.Image: Rendered frame
    """
    r = random.Random(x=1440)
    img = Image.new(
      mode='RGB',
      size=(
        int(self.width * self.scale),
        int(self.height * self.scale)
      ),
      color=self.hex_to_tuple(self.background_clr),
    )
    draw = ImageDraw.Draw(img)

    for _stat in self.space.statics:
      match type(_stat):
        case shapes.CircleBorder:
          cborder = cast(shapes.CircleBorder, _stat)
          draw.ellipse(
            xy=(  # type: ignore[arg-type]
              (cborder.x - cborder.radius) * self.scale,
              self.flipy(self.height, cborder.y + cborder.radius) * self.scale,
              (cborder.x + cborder.radius) * self.scale,
              self.flipy(self.height, cborder.y - cborder.radius) * self.scale,
            ),
            fill=self.hex_to_tuple(self.border_clr),
          )
        case shapes.RectangleBorder:
          rborder = cast(shapes.RectangleBorder, _stat)
          points = [
            helpers.rotate_point(x, y, rborder.pos, rborder.angle)
            for x, y in
            [
              (rborder.x - rborder.width/2, rborder.y - rborder.height/2),
              ((rborder.x + rborder.width/2), rborder.y - rborder.height/2),
              ((rborder.x + rborder.width/2), (rborder.y + rborder.height/2)),
              (rborder.x - rborder.width/2, (rborder.y + rborder.height/2)),
            ]
          ]
          draw.polygon(
            xy=[(x * self.scale, self.flipy(self.height * self.scale, y * self.scale)) for (x, y) in points], # type: ignore
            fill=self.hex_to_tuple(self.border_clr),
          )
    if watermark:
      assert len(watermark) == 3, f'{watermark=} is not of form ("watermark", x, y)'
      text, x, y = watermark
      if platform.system() == 'Windows':
        font = ImageFont.truetype('ariblk.ttf', self.height // 4)
      else:
        font = ImageFont.truetype('Arial Black.ttf', self.height // 4)
      draw.text(xy=(x,y), text=text, align='left', anchor='mm', fill=self.hex_to_tuple(self.watermark_clr), font=font)

    for _kin in self.space.kinetics:
      match type(_kin):
        case shapes.Circle:
          circle = cast(shapes.Circle, _kin)
          draw.ellipse(
            xy=(  # type: ignore[arg-type]
              (circle.x - circle.radius) * self.scale,
              self.flipy(self.height, circle.y + circle.radius) * self.scale,
              (circle.x + circle.radius) * self.scale,
              self.flipy(self.height, circle.y - circle.radius) * self.scale,
            ),
            fill=self.hex_to_tuple(r.choice(self.f_palette)),
          )
        case shapes.Rectangle:
          rect = cast(shapes.Rectangle, _kin)
          points = [
            helpers.rotate_point(x, y, rect.pos, rect.angle)
            for x, y in
            [
              (rect.x - rect.width/2, rect.y - rect.height/2),
              ((rect.x + rect.width/2), rect.y - rect.height/2),
              ((rect.x + rect.width/2), (rect.y + rect.height/2)),
              (rect.x - rect.width/2, (rect.y + rect.height/2)),
            ]
          ]
          draw.polygon(
            xy=[(x * self.scale, self.flipy(self.height * self.scale, y * self.scale)) for (x, y) in points], # type: ignore
            fill=self.hex_to_tuple(r.choice(self.f_palette)),
          )

    return img

  def bad_live_render(self) -> None:
    """Simple live rendering function."""
    plt.ion()
    _, ax = plt.subplots()

    while 1:
      ax.imshow(np.array(self.render_current_frame()))
      for _ in range(self.step_size):
        self.space.step()
      plt.draw()
      plt.pause(0.01)
      ax.cla()

  def render_sound(self, collisions: dict[CollisionHandler, list[int]], duration: float, frame_rate: float, sr:int=44100) -> NDArray[DTYPE]:
    """Renders the sound of a simulation based on the collisions.

    Args:
        collisions (dict[CollisionHandler, list[int]]): each handler contains a list of frames where they handled a collision
        duration (float): duration of the sound TODO: should be replaced by max(collisions.values())
        frame_rate (float): frame rate of the simulation
        sr (int, optional): Sample Rate. Defaults to 44100.

    Returns:
        NDArray[DTYPE]: numpy array containing the sound data
    """
    sound = np.zeros((int(sr * duration), 2), dtype=DTYPE)
    for k, v in (t:=tqdm(collisions.items())):
      for c, i in enumerate(v):
        s = next(k.sound_gen)
        if len(s.shape) == 1: s = np.stack((s, s), axis=1)
        if len(s) > len(sound) - int(i*sr/frame_rate): s = s[:len(sound) - int(i*sr/frame_rate)]
        sound[int(i*sr/frame_rate): int(i*sr/frame_rate) + int(k.sound_gen.duration * sr)] += s
        t.set_description(f'{c}/{len(v)}')
      t.set_description('Rendering sound...')
    return sound

  def render(self, frame_count: int, frame_rate:float=30.0, path: str='output') -> None:
    """Renders a simulation to a video.

    Args:
        frame_count (int): number of frames to render
        frame_rate (float, optional): frame rate. Defaults to 30.0.
        path (str, optional): Save files to path. Defaults to 'output'.
    """
    otp = Path(path)
    if not otp.exists():
      otp.mkdir()
    elif not otp.is_dir():
      print(f'{otp.absolute()} is no valid directory')
      return

    proj = int(time.time())
    proj_path = otp / Path(str(proj))
    if not proj_path.exists():
      proj_path.mkdir()
    elif not proj_path.is_dir():
      print(f'{proj_path.absolute()} is no valid directory')
      return

    frames_path = Path(proj_path / 'frames')
    if not frames_path.exists():
      frames_path.mkdir()
    elif not proj_path.is_dir():
      print(f'{frames_path.absolute()} is no valid directory')
      return

    frame_name = 'frame'
    extension = 'jpg'

    print("self.dims: ", self.width, self.height)
    subprocess.call(['open', str(proj_path)])

    collision_sounds: dict[CollisionHandler, list[int]] = {}
    for i in (t:=trange(frame_count)):
      for _ in range(self.step_size):
        for ch in self.space.step():
          if not collision_sounds.get(ch, None): collision_sounds[ch] = []
          collision_sounds[ch].append((i))

      self.render_current_frame(
        (self.watermark, int(self.width * self.scale / 2), int(self.height * self.scale / 2)),
      ).save(
        f'{str(proj_path)}/frames/{frame_name}-{i:08d}.{extension}',
        quality=95,
        subsampling=0,
      )
      t.set_description('Rendering video...')

    audio = self.render_sound(collisions=collision_sounds, duration=(frame_count / frame_rate), frame_rate=frame_rate, sr=44100)

    print(f'Saved all frames in {str(otp / proj_path)}')

    audio_name = 'audio.wav'
    print(f'Saving {proj_path / audio_name}.')
    wavfile.write(proj_path / audio_name, 44100, audio)

    video_name = 'video'
    print(f'Saving {proj_path / video_name} ...')
    video = ffmpeg.input(f'{str(proj_path)}/frames/{frame_name}-%08d.{extension}', framerate=frame_rate)
    audio = ffmpeg.input(proj_path / audio_name)

    ffmpeg.concat(video, audio, v=1, a=1).output(f'{proj_path / video_name}.mp4', loglevel='quiet', crf=18).run()

    #if input(f'Remove {extension}-Frames [y/N]: ') in ['y', 'Y']:
    #  for file in (proj_path / 'frames').glob(f'*.{extension}'): file.unlink()
    #  (proj_path / 'frames').rmdir()
