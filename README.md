# Vaporizer

### An automatic Vaporwave generator by Yaakov Schectman

##### Example usage:
For a given audio file "music.wav", and an image "background.png", call the following from CLI:

`python vaporizer music.wav background.png vaporwave.mp4`

to produce the newly vaporized "vaporwave.mp4".
Alternatively, run the following Python code:

```python
import vaporizer

vaporizer.make_vid('music.wav', 'background.png', 'vaporwave.mp4')
```

Using the `make_vid` function requires having `ffmpeg`. Use `-p` or `--ffmpeg=` to specify the path to the ffmpeg executable.
