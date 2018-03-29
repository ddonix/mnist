#!/usr/bin/python
# -*- coding: UTF-8 -*-

import Tkinter as tk           # 导入 Tkinter 库
import Image
import ImageTk
import tkFileDialog
import LeNet5_operate
import numpy as np
#导入tk模块

path = None
label_pic = None
inum = None
canvas = None
buffpic = None
buffcan = None

def selectPath():
	global path,label_pic,inum,buffpic
	
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

		canvas.place_forget()
		buffcan = None
		

def writeNum():
	global label_pic,inum,path
	canvas.place(x=5,y=0,width=280,height=280)
	
	inum.delete(0,'end')
	label_pic.place_forget()
	buffpic = None
	path.set('')

def identifyNum():
	global inum,buffpic
	if buffpic != None:
		r1, r2 = LeNet5_operate.prediction_fast(buffpic,shape='bmp')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f'%(r1,r2))
	else:
		r1, r2 = LeNet5_operate.prediction_fast(buffcan, shape='280X280')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f'%(r1,r2))

def main():
	global path,label_pic,inum,canvas,buffcan
	LeNet5_operate.prediction_init()
	
	master = tk.Tk()
	master.geometry('280x380+20+20')
	
	label_pic = tk.Label(master)
	label_pic.place(x=5,y=0,width=280,height=280)
	label_pic.place_forget()

	canvas = tk.Canvas(master, width=280, height=280, bg='white')
	canvas.create_line(0,10,10,10,fill='#ff0000')
	canvas.place(x=5,y=0,width=280,height=280)
	buffcan = np.zeros([280,280], 'int8')

	path = tk.StringVar()
	tk.Label(master,text = "目标路径:").place(x=5, y=300, width=60, height=20)
	tk.Entry(master, textvariable = path).place(x=65, y=300, width=200, height=20)
	tk.Button(master, text = "路径选择", command = selectPath).place(x=5,y=320, width=60, height=20)

	tk.Button(master, text = "手写", command = writeNum).place(x=5,y=340, width=60, height=20)
	tk.Button(master, text = "识别数字", command = identifyNum).place(x=5,y=360, width=60, height=20)
	inum = tk.Entry(master)
	inum.place(x=65, y=360, width=200, height=20)

	master.update_idletasks()
	master.mainloop()

if __name__ == '__main__':
	main()
