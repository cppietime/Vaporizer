# Vaporizer

### An automatic Vaporwave generator by Yaakov Schectman

##### Example usage:
For a given audio file "music.wav", and an image "background.png", call the following from CLI:

`python vaporizer music.wav background.png vaporwave.mp4`

to produce the newly vaporized "vaporwave.mp4".
Alternatively, run the following Python code:

```python
import vaporizer

vaporizer.make_vids('music.wav', 'background.png', 'vaporwave.mp4')
```