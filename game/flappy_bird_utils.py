import pygame
import sys
import os

# simple wrapper to enforce a minimum delay between plays
class ThrottledSound:
    def __init__(self, sound: pygame.mixer.Sound, cooldown_ms: int):
        self._sound = sound
        self._cooldown = cooldown_ms
        self._last_play = 0

    def play(self, *args, **kwargs):
        now = pygame.time.get_ticks()
        if now - self._last_play >= self._cooldown:
            self._sound.play(*args, **kwargs)
            self._last_play = now

def load():
    base_path   = os.path.join(os.path.dirname(__file__), '..')  # go up to project root
    assets_path = os.path.join(base_path, 'assets')
    sprites_path= os.path.join(assets_path, 'sprites')
    audio_path  = os.path.join(assets_path, 'audio')

    PLAYER_PATH = (
        os.path.join(sprites_path, 'redbird-upflap.png'),
        os.path.join(sprites_path, 'redbird-midflap.png'),
        os.path.join(sprites_path, 'redbird-downflap.png')
    )

    BACKGROUND_PATH = os.path.join(sprites_path, 'background-black.png')
    PIPE_PATH       = os.path.join(sprites_path, 'BLUE PIPE.png')

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    IMAGES['numbers'] = tuple(
        pygame.image.load(os.path.join(sprites_path, f'{i}.png')).convert_alpha()
        for i in range(10)
    )

    IMAGES['base'] = pygame.image.load(os.path.join(sprites_path, 'base.png')).convert_alpha()

    # choose sound extension
    soundExt = '.wav' if 'win' in sys.platform else '.ogg'

    # load raw sounds
    raw_die    = pygame.mixer.Sound(os.path.join(audio_path, 'die'   + soundExt))
    raw_hit    = pygame.mixer.Sound(os.path.join(audio_path, 'hit'   + soundExt))
    raw_point  = pygame.mixer.Sound(os.path.join(audio_path, 'point' + soundExt))
    raw_swoosh = pygame.mixer.Sound(os.path.join(audio_path, 'swoosh'+ soundExt))
    raw_wing   = pygame.mixer.Sound(os.path.join(audio_path, 'wing'  + soundExt))

    # assign (wrap wing in a throttle)
    SOUNDS['die']    = raw_die
    SOUNDS['hit']    = raw_hit
    SOUNDS['point']  = raw_point
    SOUNDS['swoosh'] = raw_swoosh
    # only allow one wing sound every 300ms:
    SOUNDS['wing']   = ThrottledSound(raw_wing, cooldown_ms=300)

    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    IMAGES['player'] = tuple(
        pygame.image.load(path).convert_alpha() for path in PLAYER_PATH
    )

    IMAGES['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    HITMASKS['player'] = tuple(
        getHitmask(img) for img in IMAGES['player']
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
