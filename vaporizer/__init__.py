from scipy.io import wavfile
from scipy import signal, ndimage
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import random
import math
import os
import ffmpeg
import argparse
import sys
import glob

class VapeArgs(argparse.ArgumentParser):
	'''The CLI args for vaporization'''
	def __init__(self, args = sys.argv[1:]):
		super().__init__(description = 'Vaporize media')
		self.add_argument('--width', '-w', type=int, default=720, help='Width of image')
		self.add_argument('--height', '-v', type=int, default=480, help='Height of image')
		self.add_argument('--stretch', '-f', type=float, default=2, help='Slowdown rate')
		self.add_argument('--noise_strength', '-x', type=float, default=0.005, help='Tape hiss')
		self.add_argument('--noise_order', '-y', type=int, default=2, help='Noise butterworth order')
		self.add_argument('--noise_freq', '-z', type=float, default=2000, help='Noise hipass cutoff')
		self.add_argument('--filter_order', '-O', type=int, default=2, help='Bandpass filter order')
		self.add_argument('--lowpass', '-0', type=float, default=500, help='Bandpass low frequency')
		self.add_argument('--hipass', '-1', type=float, default=4000, help='Bandpass high frequency')
		self.add_argument('--combs', '-c', action='append', default=[1500, 1700], help='Comb filter frequencies')
		self.add_argument('--alls', '-a', action='append', default=[400, 500], help='All-pass filter frequencies')
		self.add_argument('--skips', '-r', type=int, default=2, help='Number of random tape skips')
		self.add_argument('--blur_width', '-W', type=int, default=11, help='Gaussian blur width')
		self.add_argument('--blur_height', '-H', type=int, default=1, help='Gaussian blur height')
		#TODO self.red; self.cyan
		self.add_argument('--crt_translate', '-t', type=int, default=5, help='CRT effect width')
		self.add_argument('--crt_alpha', '-C', type=int, default=64, help='CRT alpha out of 255')
		self.add_argument('--scan_spacing', '-s', type=int, default=2, help='Scanline spacing')
		self.add_argument('--off_height', '-D', type=int, default=150, help='Offset height')
		self.add_argument('--off_offset', '-E', type=int, default=-10, help='Offset translation')
		self.add_argument('--static', '-S', type=int, default=64, help='Static strength out of 255')
		self.add_argument('--framerate', '-F', type=int, default=30, help='Framerate')
		self.add_argument('--looplen', '-l', type=int, default=120, help='Loop length in frames')
		self.add_argument('--ffmpeg', '-p', type=str, default='ffmpeg', help='Path to ffmpeg')
		self.add_argument('--frame_path', type=str, default='frames', help='Frames path')
		self.add_argument('--temp_name', type=str, default='vaporized.wav', help='Audio name')
		self.add_argument('audio', type=str, help='Source audio WAV file')
		self.add_argument('image', type=str, help='Source image file')
		self.add_argument('output', type=str, help='Output file')
		# self.width = 720
		# self.height = 480
		# self.factor = 2
		# self.noise_strength = lambda x: x.max() * 0.005
		# self.noise_order = 2
		# self.noise_freq = 2000
		# args.filter_order = 2
		# args.lowpass = 500
		# args.hipass = 4000
		# self.combs = (1500, 1700)
		# self.alls = (400, 500)
		# self.skips = 3
		# self.blur_width = 11
		# self.blur_height = 1
		self.red = (255, 0, 0)
		self.cyan = (0, 255, 255)
		# self.crt_translate = 5
		# self.crt_alpha = 64
		# self.scan_color = (0, 0, 0)
		# self.scan_spacing = 2
		# self.off_height = 150
		# self.off_offset = -10
		# self.static = 64
		# self.framerate = 30
		# self.looplen = 300
		self.args = self.parse_args(args)
	
	def __getattr__(self, name):
		return getattr(self.args, name)

def stretch(data, factor):
	'''Resample w/ linear interpolation to slow or speed up a signal'''
	return np.interp(np.arange(0, len(data), 1.0 / factor), np.arange(len(data)), data).astype(data.dtype)

def noisy(data, strength = 1, ba = None):
	'''Add noise to a signal'''
	noise = np.random.rand(len(data))
	if ba is not None:
		noise = signal.lfilter(*ba, noise)
	return (noise * strength + data).astype(data.dtype)

def comb(data, combs = ( (.1, 1500), (.1, 1700) ), alls = ( (.1, 400), (.1, 500) )):
	'''Apply a reverb effect'''
	combed = [ np.roll(data, f[1]) * f[0] for f in combs ]
	combed = (sum(combed) + data).astype(data.dtype)
	for all in alls:
		num = [-all[0]] + [0] * (all[1] - 1) + [1]
		den = [1] + [0] * (all[1] - 1) + [-all[0]]
		combed = signal.lfilter(num, den, combed)
	return combed.astype(data.dtype)
	
def skips(data, num = 3):
	'''Apply random 'track skips' '''
	for _ in range(num):
		start = random.randint(0, len(data) - 5000)
		end = start + random.randint(5000, 100000)
		end = min(end, len(data) - 1)
		data = np.concatenate((data[:end], data[start:end], data[end:]))
	return data
	
def vaporize(in_file, out_file, args):
	'''Transform audio into vaporized'''
	orig = wavfile.read(in_file)
	slow = stretch(orig[1], args.stretch)
	filter = signal.butter(args.filter_order, (args.lowpass * 2 / orig[0], args.hipass * 2 / orig[0]), 'bandpass')
	high = signal.lfilter(*filter, slow)
	delayed = comb(high, [(.1, x) for x in args.combs], [(.1, x) for x in args.alls])
	hiss = noisy(delayed, delayed.max() * args.noise_strength, signal.butter(args.noise_order, args.noise_freq * 2 / orig[0], 'highpass'))
	skip = skips(hiss, args.skips)
	wavfile.write(out_file, orig[0], skip.astype(orig[1].dtype))
	
def blur(img, w, h):
	'''Apply 2D custom Gaussian Blur'''
	arr = np.asarray(img)
	kernel = np.ones((h, w, 1))
	sigx = w / 6
	sigy = h / 6
	funx = [math.exp(-(x - w//2)**2 / (2 * sigx**2)) / sigx for x in range(w)]
	funy = [math.exp(-(y - h//2)**2 / (2 * sigy**2)) / sigy for y in range(h)]
	for x in range(w):
		for y in range(h):
			kernel[y][x][0] = funx[x] * funy[y]
	kernel /= kernel.sum()
	arr = ndimage.convolve(arr, kernel, mode = 'nearest')
	return Image.fromarray(arr, 'RGBA')
	
def crt(img, args):
	'''Apply CRT effect'''
	size = img.size
	gray = img.convert('L')
	red = ImageOps.colorize(gray, (0, 0, 0), args.red)
	cyan = ImageOps.colorize(gray, (0, 0, 0), args.cyan)
	red.putalpha(args.crt_alpha)
	cyan.putalpha(args.crt_alpha)
	redmsk = Image.new(('RGBA'), size)
	redmsk.paste(red, (-args.crt_translate, 0))
	cyanmsk = Image.new(('RGBA'), size)
	cyanmsk.paste(cyan, (args.crt_translate, 0))
	cyanmsk = Image.alpha_composite(cyanmsk, redmsk)
	img = Image.alpha_composite(img, cyanmsk)
	img = blur(img, args.blur_width, args.blur_height)
	return img
	
def scanlines(img, args):
	'''Add scanlines'''
	draw = ImageDraw.Draw(img)
	for y in range(0, img.size[1], args.scan_spacing):
		draw.line((0, y, img.size[0], y), (0, 0, 0), 1)
	return img
	
def off_frame(img, start, args):
	'''Translate part of image horizontally'''
	frame = img.crop((0, start, img.size[0] - 1, start + args.off_height))
	copy = img.copy()
	copy.paste(frame, ((args.off_offset, start)))
	return copy
	
def static(img, alpha = 64):
	'''Add grayscale white noise'''
	noise = np.random.rand(img.height, img.width) * 255
	noise = np.repeat(noise, 3, axis=1).reshape((img.height, img.width, 3))
	noise = np.concatenate((noise.reshape(img.height, img.width, 3), np.ones((img.height, img.width, 1)) * alpha), axis = 2)
	nimg = Image.fromarray(noise.astype('uint8'), 'RGBA')
	return Image.alpha_composite(img, nimg)
	
def base_img(path, args):
	'''Load and distort an iamge'''
	img = Image.open(path).convert('RGBA').resize((args.width, args.height))
	scanlines(img, args)
	colors = crt(img, args)
	return colors

def save_frames(base, template, args):
	'''Save frames image frames'''
	for i in range(args.looplen):
		frame = static(off_frame(base, i, args))
		frame.save(template % i)

def make_vid(*args, **kwargs):
	'''Make the video with a given base image and sound file'''
	pars = [item for sublist in map(lambda i: ('--'+i[0], i[1]), kwargs.items()) for item in sublist] + list(args)
	print(pars)
	
	args = VapeArgs(args = pars)
	frames = os.path.join(args.frame_path, 'frame%03d.png')
	for frame in glob.glob(os.path.join(args.frame_path, 'frame*.png')):
		try:
			os.remove(frame)
		except Exception as e:
			print(e, file=sys.err)
	base = base_img(args.image, args)
	save_frames(base, frames, args)
	vaporize(args.audio, args.temp_name, args)
	os.system("%s -framerate %d -stream_loop -1 -i %s -i %s -shortest %s" % (args.ffmpeg, args.framerate, frames, args.temp_name, args.output))

def main():
	make_vid()
	