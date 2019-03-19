
from tkinter import *
import numpy as np

import network

'''
TODO:
- display where the user is drawing
- average pixels to create greyscale values
- increase "width" of user's cursors when drawing
'''

root = Tk()

K = 3

canvas = Canvas(root, width=28*K, height=28*K)

arr = np.zeros((28, 28), dtype=np.float32)

net = network.load('network.json')

def click_move(event):
	#print(event.x, event.y)
	#print(canvas.winfo_width(), canvas.winfo_height()
	arr[event.y//K][event.x//K] = 1
	

def reset(event):
	global arr
	arr = np.zeros((28, 28), dtype=np.float32)

def display(event):
	print(arr)
	print(arr.flatten().reshape(28*28, 1).shape)
	print(net.feedforward(arr.flatten().reshape(28*28, 1)))


root.bind('<B1-Motion>', click_move)
root.bind('<2>', reset)
root.bind('<1>', display)

canvas.pack()

root.mainloop()