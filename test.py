import os
import glob


PATH = '/Users/vietnd/Wallpapers/Gais/'
os.chdir(PATH)
glob = glob.glob('*.jpg')
print(glob)
