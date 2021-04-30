import tkinter as tk
from tkinter import ttk
from PIL import ImageTk,Image
from tkinter import messagebox
import mysql.connector



class LoginPage:
	def __init__(self,parent):
		self.root=parent
		self.root.title("Login")
		self.root.geometry('950x600+200+50')
		self.root.resizable(0,0)
# resizing images
		ima=Image.open('project_image/images.png')
		newsize=(140,140)
		ima=ima.resize(newsize)
		self.images=ImageTk.PhotoImage(ima)

		# image=Image.open('project_image/logo.png')
		# newsize=(100,90)
		# imag=image.resize(newsize)
		# self.image=ImageTk.PhotoImage(imag)

		image_userL=Image.open('project_image/userlogo.png')
		newsize=(20,20)
		imag_userL=image_userL.resize(newsize)
		self.image_uL=ImageTk.PhotoImage(imag_userL)

		image_passL=Image.open('project_image/passlogo.png')
		newsize=(20,20)
		imag_passL=image_passL.resize(newsize)
		self.image_pL=ImageTk.PhotoImage(imag_passL)


		# All images
		# self.bg_icon=ImageTk.PhotoImage(file='../project_image/bg.jpg')


		# tk.Label(self.root,image=self.bg_icon).pack()

		title=tk.Label(self.root,text="Stock Prediction Login System",font=("times new roman",30,"bold"),bg="white",fg="black",bd=5,relief=tk.GROOVE)
		title.place(x=0,y=0,relwidth=1)

		self.Login_frame=tk.Frame(self.root,bg='white')
		self.Login_frame.place(x=180,y=100,height=450,width=600)

		logolbl=tk.Label(self.Login_frame,image=self.images,bd=0).grid(row=0,columnspan=2,pady=10)
		user_lbl=tk.Label(self.Login_frame,text="Username",image=self.image_uL,compound=tk.LEFT,font=('times new roman,',15,'bold'),bg='white')
		user_lbl.grid(row=1,column=0,padx=2,pady=2,sticky="w")
		pass_lbl=tk.Label(self.Login_frame,text="Password",image=self.image_pL,compound=tk.LEFT,font=('times new roman,',15,'bold'),bg='white')
		pass_lbl.grid(row=2,column=0,padx=2,pady=2,sticky="w")

		# button collection frame
		btnColl_frame=tk.Frame(self.root)
		btnColl_frame.place(x=655,y=100,height=450)

		btn1=tk.Button(btnColl_frame,text="LogIn",font=('times new roman',15,'bold'),bg='skyblue',fg='red',bd=1,command=self.login,cursor="hand2")
		btn1.grid(row=1,column=0,padx=12,pady=12,ipadx=30)
		btn2=tk.Button(btnColl_frame,text="SignUp",font=('times new roman',15,'bold'),bg='green',fg='white',bd=1,command=self.register,cursor="hand2")
		btn2.grid(row=2,column=0,padx=12,pady=2,ipadx=25)


		# variables
		self.uservar=tk.StringVar()
		self.passvar=tk.StringVar()
		self.repassvar=tk.StringVar()

		# Entry
		self.user_text=tk.Entry(self.Login_frame,textvariable=self.uservar,bd=5,font=("",20),relief=tk.GROOVE)
		self.user_text.grid(row=1,column=1)
		pass_text=tk.Entry(self.Login_frame,show="*",textvariable=self.passvar,bd=5,font=("",20),relief=tk.GROOVE)
		pass_text.grid(row=2,column=1)

		
		# button
		self.btn=tk.Button(self.Login_frame,text="Login",font=('times new roman',15,'bold'),bg='yellow',fg='red',bd=1,command=self.action,cursor="hand2")
		self.btn.grid(row=3,columnspan=3,pady=10,ipadx=10)
		self.user_text.focus()
		
	def action(self):
		if(self.uservar.get()=="" or self.passvar.get()==""):
			messagebox.showerror("Error","username and password are required..!")
		else:
			con=mysql.connector.connect(host="localhost",user="root",password="",database="stockData")
			cur=con.cursor()
			sql="select * from registration where username =%s and password=%s"
			var=(self.uservar.get(),self.passvar.get())
			cur.execute(sql,var)
			result=cur.fetchone()
			if(result !=None):
				self.root.destroy()
				import xyz
				
			else:
				messagebox.showwarning("Warning",'Invalid  characters....!')
			

	def register(self):
		self.root.title('register')
		self.btn.grid_forget()
		self.repass_lbl=tk.Label(self.Login_frame,text="Re-Password",image=self.image_pL,compound=tk.LEFT,font=('times new roman,',15,'bold'),bg='white')
		self.repass_lbl.grid(row=3,column=0,padx=2,pady=2,sticky="w")
		self.repass_text=tk.Entry(self.Login_frame,show="*",textvariable=self.repassvar,bd=5,font=("",20),relief=tk.GROOVE)
		self.repass_text.grid(row=3,column=1)
		self.btnn=tk.Button(self.Login_frame,text="Signup",font=('times new roman',15,'bold'),bg='yellow',fg='red',bd=1,command=self.SignUp,cursor="hand2")
		self.btnn.grid(row=4,columnspan=3,pady=10,ipadx=10)
		self.uservar.set("")
		self.passvar.set("")
		self.repassvar.set("")

	def SignUp(self):
		if(self.uservar.get()=="" or self.passvar.get()=="" or self.repassvar.get()==""):
			messagebox.showerror("Error","username and password are required..!")
		else:

			if(self.passvar.get() != self.repassvar.get()):
				messagebox.showwarning("Warning","password doesnot match !")
			else:
				con=mysql.connector.connect(host="localhost",user="root",password="",database="stockData")
				cur=con.cursor()
				sql="select * from registration where username =%s"
				var=(self.uservar.get(),)
				cur.execute(sql,var)
				result=cur.fetchone()
				
				if result !=None:
				    messagebox.showwarning("Warning",f"{self.uservar.get()}already exits ...")
				else:
					sql="insert into `registration`(`username`,`password`,`re-password`) values(%s,%s,%s)"
					val=(self.uservar.get(),self.passvar.get(),self.repassvar.get())
					cur.execute(sql,val)
					con.commit()
					con.close()
					messagebox.showinfo("success","Successfully registered.")

	def login(self):
		self.root.title("Login")
		self.uservar.set("")
		self.passvar.set("")
		try:
			self.btnn.grid_forget()
			self.repass_lbl.grid_forget()
			self.repass_text.grid_forget()
			self.btn.grid(row=3,columnspan=3,pady=10,ipadx=10)
		except:
			pass
		

root=tk.Tk()

obj=LoginPage(root)
root.mainloop()
