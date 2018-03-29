#!/usr/bin/python
# -*- coding: UTF-8 -*-

import Tkinter as tk           # 导入 Tkinter 库
import Image
import ImageTk
import tkFileDialog
import LeNet5_operate
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
		f = open(path_,'rb+')
		f.seek(54)
		buffpic = f.read(280*280*3)
		f.close()
		inum.delete(0,'end')
		

def writeNum():
	global label_pic,inum
	im2 = Image.open('./pic/white.bmp')
	im2 = ImageTk.PhotoImage(im2)
	label_pic.bm=im2
	label_pic.configure(image=im2)
	inum.delete(0,'end')
	label_pic.place_forget()
	canvas.place(x=5,y=0,width=280,height=280)

def identifyNum():
	global inum,buffpic
	if buffpic != None:
		r1, r2 = LeNet5_operate.prediction_fast(buffpic,shape='280X280X3')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f'%(r1,r2))
	elif buffcan != None:
		r1, r2 = LeNet5_operate.prediction_fast(buffcan, shape='280X280')
		inum.delete(0,'end')
		inum.insert(0,'%d 可信度%f'%(r1,r2))
	else:
		inum.delete(0,'end')
		inum.insert(0,'没有图片供识别')

def main():
	global path,label_pic,inum,canvas
	LeNet5_operate.prediction_init()
	
	master = tk.Tk()
	master.geometry('280x380+20+20')
	im = Image.open('./pic/n.bmp')
	im = ImageTk.PhotoImage(im)
	label_pic = tk.Label(master, image = im)
	label_pic.bm = im
	label_pic.place(x=5,y=0,width=280,height=280)

	canvas = tk.Canvas(master, width=280, height=280)
	canvas.create_line(0,10,10,10,fill='#476042')

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
