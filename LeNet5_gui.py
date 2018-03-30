#!/usr/bin/python
# -*- coding: UTF-8 -*-

import Tkinter as tk           # 导入 Tkinter 库
import Image
import ImageTk
import tkFileDialog
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_operate
import numpy as np
#导入tk模块

path = None
label_pic = None
inum = None
tnum = None
canvas = None
buffpic = None
buffcan = None
buffmnist = None

#0:文件识别，1:手写识别鼠标松开,2:手写识别鼠标按下
state = 1

def selectPath():
	global path,label_pic,inum, tnum, buffpic,state
	
	path_ = tkFileDialog.askopenfilename()
	
	path.set(path_)
	im2 = Image.open(path_)
	if im2 != None:
		im2 = ImageTk.PhotoImage(im2)
		label_pic.bm=im2
		label_pic.configure(image=im2)
		label_pic.place(x=5,y=0,width=280,height=280)
		f = open(path_,'rb+')
		f.seek(54)
		buffpic = f.read(280*280*3)
		f.close()
		inum.delete(0,'end')
		tnum.delete(0,'end')

		canvas.place_forget()
		buffcan = None
		state = 0

def mouse_press_event(evt):
	global canvas, state, buffcan
	if state == 0:
		return
	state = 2
	canvas.create_rectangle(evt.x+10,evt.y+10,evt.x-10,evt.y-10,fill = 'black')	
	for i in np.arange(max(0,evt.y-13),min(280,evt.y+14)):
		for j in np.arange(max(0,evt.x-13),min(280,evt.x+14)):
			buffcan[i][j] = 1

def mouse_release_event(evt):
	global canvas, state, buffcan
	if state == 0:
		return
	state = 1

def mouse_movie_event(evt):
	global canvas, state, buffcan
	if state != 2:
		return
	canvas.create_rectangle(evt.x+10,evt.y+12,evt.x-10,evt.y-12,fill = 'black')	
	for i in np.arange(max(0,evt.y-13),min(280,evt.y+14)):
		for j in np.arange(max(0,evt.x-13),min(280,evt.x+14)):
			buffcan[i][j] = 1

def writeNum():
	global label_pic,inum,tnum,path,state,buffcan
	canvas.delete('all')
	canvas.place(x=5,y=0,width=280,height=280)
	inum.delete(0,'end')
	tnum.delete(0,'end')
	label_pic.place_forget()
	buffcan = np.zeros([280,280], 'int8')
	buffmnist = None
	path.set('')
	state = 1

def greyPic():
	global buffcan,buffmnist,state
	buffmnist = np.zeros([28,28],dtype=float)
	if state == 0:
		label_pic.place_forget()
		canvas.delete('all')
		canvas.place(x=5,y=0,width=280,height=280)
		
		for ibase in np.arange(28):
			for jbase in np.arange(28):
				sum = 0
				for ii in np.arange(10):
					for jj in np.arange(10):
						for kk in np.arange(3):
							sum += ord(buffpic[(279-ibase*10-ii)*280*3+(jbase*10+jj)*3+kk])
				buffmnist[ibase][jbase] = 1-float(sum)/300/255
	elif state == 1 or state == 2:
		xx = np.split(buffcan,28,axis=0)
		for ibase in np.arange(28):
			t = np.split(xx[ibase],28,axis=1)
			for jbase in np.arange(28):
				buffmnist[ibase][jbase] = float(t[jbase].sum())/100
	else:
		return
	
	state = 3
	canvas.delete('all')
	for ibase in np.arange(28):
		for jbase in np.arange(28):
			col = 255-int(buffmnist[ibase][jbase]*255)
			if col == 0:
				col = '00'
			elif col < 16:
				col = '0'+hex(col)[2]
			else:
				col = hex(col)[2:3]
			col = '#'+col+col+col
			canvas.create_rectangle(jbase*10+10,ibase*10+10,jbase*10,ibase*10, outline=col,fill =col)

def identifyNum():
	global inum,tnum,buffpic,buffcan, state
	if state == 0:
		r1, r2, r3 = LeNet5_operate.prediction_fast(buffpic,shape='bmp')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f 备选%d'%(r1,r2,r3))
		tnum.delete(0,'end')
		tnum.insert(0,'%d'%r1)
	elif state == 3:
		r1, r2, r3 = LeNet5_operate.prediction_fast(buffmnist, shape='mnist')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f 备选%d'%(r1,r2,r3))
		tnum.delete(0,'end')
		tnum.insert(0,'%d'%r1)
	else:
		r1, r2, r3 = LeNet5_operate.prediction_fast(buffcan, shape='280X280')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f 备选%d'%(r1,r2,r3))
		tnum.delete(0,'end')
		tnum.insert(0,'%d'%r1)

def addTrain():
	tnum = 0

def main():
	global path,label_pic,inum,tnum,canvas,buffcan,state
	LeNet5_operate.prediction_init()
	
	master = tk.Tk()
	master.geometry('280x380+20+20')
	
	label_pic = tk.Label(master)
	label_pic.place(x=5,y=0,width=280,height=280)
	label_pic.place_forget()

	canvas = tk.Canvas(master, width=280, height=280, bg='white')
	canvas.place(x=5,y=0,width=280,height=280)
	canvas.bind("<ButtonPress-1>",mouse_press_event)
	canvas.bind("<ButtonRelease-1>",mouse_release_event)
	canvas.bind("<Motion>",mouse_movie_event)
	buffcan = np.zeros([280,280], 'int8')
	state = 1

	path = tk.StringVar()
	tk.Label(master,text = "图片路径:").place(x=5, y=300, width=60, height=20)
	tk.Entry(master, textvariable = path).place(x=65, y=300, width=200, height=20)
	tk.Button(master, text = "选择图片", command = selectPath).place(x=5,y=320, width=60, height=20)

	tk.Button(master, text = "清手写版", command = writeNum).place(x=65,y=320, width=60, height=20)
	tk.Button(master, text = "转MNIST灰度图", command = greyPic).place(x=130,y=320, width=110, height=20)
	tk.Button(master, text = "识别数字", command = identifyNum).place(x=5,y=340, width=60, height=20)
	inum = tk.Entry(master)
	inum.place(x=65, y=340, width=200, height=20)
	
	tk.Button(master, text = "加入训练", command = addTrain).place(x=5,y=360, width=60, height=20)
	tnum = tk.Entry(master)
	tnum.place(x=65, y=360, width=20, height=20)

	master.update_idletasks()
	master.mainloop()

if __name__ == '__main__':
	main()
