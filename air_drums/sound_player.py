"""Play drums sounds."""
from threading import Thread
from pathlib import Path
import simpleaudio as sa

# Drum sounds
CLAP = sa.WaveObject.from_wave_file(str(Path("./assets/audio/Clap-1.wav")))
BASS_DRUM = sa.WaveObject.from_wave_file(str(Path("./assets/audio/Bass-Drum-1.wav")))
ELECTRONIC_TOM = sa.WaveObject.from_wave_file(
    str(Path("./assets/audio/Electronic-Tom-3.wav"))
)
CRASH_CYMBAL = sa.WaveObject.from_wave_file(
    str(Path("./assets/audio/Crash-Cymbal-8.wav"))
)
BOOM_KICK = sa.WaveObject.from_wave_file(str(Path("./assets/audio/Boom-Kick.wav")))
SNARE_DRUM = sa.WaveObject.from_wave_file(
    str(Path("./assets/audio/Ensoniq-ESQ-1-Snare-2.wav"))
)


def play_drum(drum: sa.WaveObject):
    """Play a  drum sound."""
    drum_play = drum.play()
    drum_play.wait_done()


def play_drum_sound(drum_sound: sa.WaveObject):
    """Run play_drum in another thread."""
    thread = Thread(target=play_drum, args=(drum_sound,))
    thread.start()
