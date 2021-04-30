import pandas as pd
# import datetime
import math
import numpy as np
from sklearn import preprocessing, svm #scale, regresions, cross shuffle stats sepeareate data
from sklearn.linear_model import LinearRegression

# import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from tkinter import messagebox
import pickle
import joblib
# from sklearn import *
# from matplotlib import style
# import csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score


import tkinter as tk
from tkinter import ttk 

class StockMarket:
    def __init__(self,root):
        self.parent=root
        self.parent.title('Stock Market Prediction System')
        self.parent.geometry('1600x700+0+0')

        self.dataset=tk.StringVar()
        self.algo=tk.StringVar()
        self.algos=tk.StringVar()

# header_
        title=tk.Label(self.parent,text="Stock Market Prediction System",font=("times new roman",40,'bold'),bd=10,relief=tk.GROOVE)
        title.place(x=0,y=0,relwidth=1)

# Training frame
        Manage_Frame=tk.Frame(self.parent,bd=4,bg="white",relief=tk.RIDGE)
        Manage_Frame.place(x=20,y=80,width=500,height=300)
 # Title of project
        m_title=tk.Label(Manage_Frame,text="Train the Data",bg="white",fg="green",font=("helvatica",20,"bold"))
        m_title.grid(row=0,columnspan=2)


 # dataset label and entry
        lbl_faculty=tk.Label(Manage_Frame,text='Dataset:',bg='white',fg='black',font=('times new roman',15,'bold'))
        lbl_faculty.grid(row=1,column=0,pady=2,padx=20,sticky='w')
        self.combo_faculty=ttk.Combobox(Manage_Frame,font=('times new roman ',15,'bold'),textvariable=self.dataset,state='readonly',width=17)
        self.combo_faculty.grid(row=1,column=1,pady=2,sticky="w")

        lbl_algo=tk.Label(Manage_Frame,text='Algorithm:',bg='white',fg='black',font=('times new roman',15,'bold'))
        lbl_algo.grid(row=2,column=0,pady=2,padx=20,sticky='w')
        self.combo_algos=ttk.Combobox(Manage_Frame,font=('times new roman ',15,'bold'),textvariable=self.algo,state='readonly',width=17)
        self.combo_algos.grid(row=2,column=1,pady=2,sticky="w")

# button 
        Button=tk.Button(text="Done",command=self.predict,width=10,height=2,bg='yellow',fg='red',font=('helvatica',10),bd=2)
        Button.place(x=150,y=250,width=200,height=50)

#Default values
        self.open=tk.StringVar()
        self.high=tk.StringVar()
        self.low=tk.StringVar()
        self.close=tk.StringVar()


        

        # =========Predict Frame===========
        predict_frame=tk.Frame(self.parent,bd=4,relief=tk.RIDGE,bg='white')
        predict_frame.place(x=550,y=80,width=600,height=250)

        # Adj Close     High_Low_per  Per_change  Volume
        _title=tk.Label(predict_frame,text="Predict the Dataset ",bg="white",fg="green",font=("helvatica",20,"bold"))
        _title.grid(row=0,columnspan=2)
        # open
        opens=tk.Label(predict_frame,text='Open:',bg='white',fg='black',font=('times new roman',15,'bold'))
        opens.grid(row=1,column=0,pady=2,padx=20,sticky='w')
        open_=tk.Entry(predict_frame,textvariable =self.open,font=('times new roman',15,'bold'),bd=3)
        open_.grid(row=1,column=1,pady=2,sticky="w")
        # high
        high=tk.Label(predict_frame,text='High:',bg='white',fg='black',font=('times new roman',15,'bold'))
        high.grid(row=2,column=0,pady=2,padx=20,sticky='w')
        high_=tk.Entry(predict_frame,textvariable =self.high,font=('times new roman',15,'bold'),bd=3)
        high_.grid(row=2,column=1,pady=2,sticky="w")
        #low
        low=tk.Label(predict_frame,text='Low:',bg='white',fg='black',font=('times new roman',15,'bold'))
        low.grid(row=3,column=0,pady=2,padx=20,sticky='w')
        low_=tk.Entry(predict_frame,textvariable =self.low,font=('times new roman',15,'bold'),bd=3)
        low_.grid(row=3,column=1,pady=2,sticky="w")
        # close
        close=tk.Label(predict_frame,text='CLose:',bg='white',fg='black',font=('times new roman',15,'bold'))
        close.grid(row=4,column=0,pady=2,padx=20,sticky='w')
        close_=tk.Entry(predict_frame,textvariable =self.close,font=('times new roman',15,'bold'),bd=3)
        close_.grid(row=4,column=1,pady=2,sticky="w")

        algo=tk.Label(predict_frame,text='Algorithm:',bg='white',fg='black',font=('times new roman',15,'bold'))
        algo.grid(row=5,column=0,pady=2,padx=20,sticky='w')
        self.combo_algo=ttk.Combobox(predict_frame,font=('times new roman ',15,'bold'),textvariable=self.algos,state='readonly',width=17)
        self.combo_algo.grid(row=5,column=1,pady=2,sticky="w")


        Button_predict=tk.Button(text="Predict",command=self.predicted,width=10,height=2,bg='yellow',fg='red',font=('helvatica',10),bd=2)
        Button_predict.place(x=950,y=160,width=100,height=100)


        self.accuracy=tk.Label(self.parent,text="",fg='black',font=("helvatica",20,"bold"))
        self.accuracy.place(x=550,y=300,width=600,height=50)

        # predicted vol
        self.vol=tk.Label(self.parent,text="",fg='black',font=("helvatica",20,"bold"))
        self.vol.place(x=1200,y=170,width=400,height=200)


# values 
        self.dataset.set("Select Dataset")
        self.algo.set("Select Algorithm")
        self.algos.set("Choose Algorithm")

        
        self.combo_algo['values']=('Linear','SVM')
        self.combo_faculty['values']=('TSLA','AAPL')
        self.combo_algos['values']=('Linear','SVM')


    def predicted(self):

            if(self.open.get()=="" or self.close.get()=="" or self.high.get()=="" or self.low.get()=="" or self.algos.get()== "Choose Algorithm"):
                messagebox.showerror("Error", "Please Provide Valid Data")
            else:
                opens=self.open.get()
                close=self.close.get()
                high=self.high.get()
                low=self.low.get()
                value=[[int(opens),int(close),int(high),int(low)]]

                if self.algos.get()=="Linear":
                        df=pd.read_csv(f'dataset/AAPL.csv',parse_dates = True, index_col=0)
                        df=df.dropna()
                        df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*100
                        df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100
                        df = df[['Adj Close','High_Low_per','Per_change','Volume']]
                        label_col = 'Adj Close'
                        forecast_ceil = int(math.ceil(0.001*len(df)))
                        df['label'] = df[label_col].shift(-forecast_ceil)

                        #feaures X, labels Y
                        X = np.array(df.drop(['label'],1))
                        X = preprocessing.scale(X)
                        X = X[:-forecast_ceil:]
                        X_lately = X[-forecast_ceil:] #no y value
                        df.dropna(inplace=True)

                        y = np.array(df['label'])

                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                        lr=LinearRegression()
                        lr.fit(X_train,y_train)

                        self.vol.config(text="Predicted using  \n Linear is {}".format(lr.predict(value)[0]))
        

                if self.algos.get()=="SVM":
                        df=pd.read_csv(f'dataset/TSLA.csv',parse_dates = True, index_col=0)
                        df = df.dropna()
                        df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*10
                        df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100
                        df = df[['Adj Close','High_Low_per','Per_change','Volume']]
                        label_col = 'Adj Close'
                        forecast_ceil = int(math.ceil(0.001*len(df)))
                        df['label'] = df[label_col].shift(-forecast_ceil)

                        X = np.array(df.drop(['label'],1))
                        X = preprocessing.scale(X)
                        X = X[:-forecast_ceil:]
                        y = np.array(df[:-forecast_ceil]['label'])
                
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                        clf = svm.SVR(kernel='rbf') #svm.SVR()
                        clf.fit(X_train, y_train) 
                        self.vol.config(text="Predicted using \n SVM is {}".format(clf.predict(value)[0]))
        #clear screen after providing value
                self.open.set('')
                self.close.set('')
                self.low.set('')
                self.high.set('')
                self.algos.set("Choose Algorithm")


    def predict(self):
        dataSet=self.dataset.get()
        algorithm=self.algo.get()
        if dataSet=='Select Dataset' or algorithm=='Select Algorithm':
                messagebox.showerror("Error", "Please Provide Valid Data")

        elif(dataSet=="TSLA" and algorithm=="SVM"):
                df=pd.read_csv(f'dataset/{dataSet}.csv',parse_dates = True, index_col=0)
                df = df.dropna()
                df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*10
                df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100
                df = df[['Adj Close','High_Low_per','Per_change','Volume']]
                label_col = 'Adj Close'
                forecast_ceil = int(math.ceil(0.001*len(df)))
                df['label'] = df[label_col].shift(-forecast_ceil)

                X = np.array(df.drop(['label'],1))
                X = preprocessing.scale(X)
                X = X[:-forecast_ceil:]
                X_lately = X[-forecast_ceil:] #no y value
                
                y = np.array(df[:-forecast_ceil]['label'])

                
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                clf = svm.SVR(kernel='rbf') #svm.SVR()
                clf.fit(X_train, y_train) 
                accuracy = clf.score(X_test, y_test) 
                
                self.accuracy.config(text="Accuracy using {} is {}".format(algorithm,accuracy))
                predicted=clf.predict(X_test)
                Actual=(y_test)
               
                df['Forecast'] = np.nan
                self.imageShow(predicted,Actual,df)

        elif(dataSet=="AAPL" and algorithm=="Linear"):
                df=pd.read_csv(f'dataset/{dataSet}.csv',parse_dates = True, index_col=0)
                df=df.dropna()
                df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*100
                df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100
                df = df[['Adj Close','High_Low_per','Per_change','Volume']]
                label_col = 'Adj Close'
                forecast_ceil = int(math.ceil(0.001*len(df)))
                df['label'] = df[label_col].shift(-forecast_ceil)

                #feaures X, labels Y
                X = np.array(df.drop(['label'],1))
                X = preprocessing.scale(X)
                X = X[:-forecast_ceil:]
                X_lately = X[-forecast_ceil:] #no y value
                df.dropna(inplace=True)

                y = np.array(df['label'])

                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                # train_f1 = X_train[:,0]
                # train_f2 = X_train[:,1]
                # train_f1 = train_f1.reshape(200,1)
                # train_f2 = train_f2.reshape(200,1)
                # w1 = np.zeros((200,1))
                # w2 = np.zeros((200,1))
                # epochs = 1
                # alpha = 0.0001

                # while(epochs < 10000):
                #         y = w1 * train_f1 + w2 * train_f2
                #         prod = y * y_train
                #         count = 0
                #         for val in prod:
                #                 if(val.any() >= 1):
                #                         cost = 0
                #                         w1 = w1 - alpha * (2 * 1/epochs * w1)
                #                         w2 = w2 - alpha * (2 * 1/epochs * w2)
                                
                #                 else:
                #                         cost = 1 - val 
                #                         w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
                #                         w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
                #                         count += 1
                #         epochs += 1



                lr=LinearRegression()
                lr.fit(X_train,y_train)
                lr_confidence=lr.score(X_test,y_test)

                df['Forecast'] = np.nan

                
                self.accuracy.config(text="Accuracy using {} is {}".format(algorithm,lr_confidence))
                predicted=lr.predict(X_test)
                Actual=(y_test)
                self.imageShow(predicted,Actual,df)

        
        else:
                df=pd.read_csv(f'dataset/{dataSet}.csv',parse_dates = True, index_col=0)
                df = df.dropna()
                df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*10
                df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100
                df = df[['Adj Close','High_Low_per','Per_change','Volume']]
                label_col = 'Adj Close'
                forecast_ceil = int(math.ceil(0.001*len(df)))
                df['label'] = df[label_col].shift(-forecast_ceil)

                X = np.array(df.drop(['label'],1))
                X = preprocessing.scale(X)
                X = X[:-forecast_ceil:]
                X_lately = X[-forecast_ceil:] #no y value
                
                y = np.array(df[:-forecast_ceil]['label'])

                
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                if(algorithm=="Linear"):
                        clf=LinearRegression()
                else:
                        clf = svm.SVR(kernel='rbf') #svm.SVR()
                clf.fit(X_train, y_train) 
                accuracy = clf.score(X_test, y_test) 
                
                self.accuracy.config(text="Accuracy using {} is {}".format(algorithm,accuracy))
                predicted=clf.predict(X_test)
                Actual=(y_test)
               
                df['Forecast'] = np.nan
                self.imageShow(predicted,Actual,df)

    def imageShow(self,predicted,Actual,df):
        f3=plt.Figure(figsize=(3,2),dpi=80)
        ax3=f3.add_subplot(111)
        scatter3=FigureCanvasTkAgg(f3,self.parent)
        scatter3.get_tk_widget().place(x=30,y=390,width=350,height=350)
        ax3.plot(Actual,color='red',label='Real Stock Price',linewidth=1)
        ax3.plot(predicted,color='blue',label='Predicted Stock Price',linewidth=2)
        ax3.set_title("Stock Price Prediction (Actual vs Predicted)")
        ax3.set_xlabel('Time in days')
        ax3.set_ylabel('Stock Price')


        f4=plt.Figure(figsize=(3,2),dpi=80)
        ax4=f4.add_subplot(111)
        scatter4=FigureCanvasTkAgg(f4,self.parent)
        scatter4.get_tk_widget().place(x=400,y=390,width=350,height=350)
        ax4.plot(Actual,color='red',label='Real Stock Price',linewidth=2)
        ax4.set_title("Stock Price Prediction(Actual)")
        ax4.set_xlabel('Time in days')
        ax4.set_ylabel('Stock Price')

        f5=plt.Figure(figsize=(3,2),dpi=80, facecolor='w', edgecolor='k')
        ax5=f5.add_subplot(111)
        scatter5=FigureCanvasTkAgg(f5,self.parent)
        scatter5.get_tk_widget().place(x=740,y=390,width=350,height=350)
        ax5.plot(predicted,color='green',label='Real Stock Price',linewidth=2)
        ax5.set_title("Stock Price Prediction(predicted)")
        ax5.set_xlabel('Time in days')
        ax5.set_ylabel('Stock Price')

        f6=plt.Figure(figsize=(3,2),dpi=80, facecolor='w', edgecolor='k')
        ax6=f6.add_subplot(111)
        scatter6=FigureCanvasTkAgg(f6,self.parent)
        scatter6.get_tk_widget().place(x=1100,y=390,width=350,height=350)
        ax6.plot(df['Adj Close'],color='red',label='Adj Close',linewidth=2)
        ax6.plot(df['Forecast'],color='blue',label='Forecast',linewidth=2)
        ax6.set_title("Closing Price")
        ax6.set_title("Closing Price")
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Price')

        
                

root=tk.Tk()
obj=StockMarket(root)
root.mainloop()
