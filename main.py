
import pandas as pd
from sklearn import linear_model
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

stock_market={
    'Year':[2017,2017,2017,2017,2017,2017,2017,2017,2017,2017],
    'Month':[12,11,10,9,8,7,6,5,4,3],
     'Interest_Rate':[2.3,1.2,3,4,5,2,1,3,3,4],
    'Unemployment_Rate':[5,3,4,2,3,4,1,2,3,4],
    'Stock_index_Price':[1234,2892,3234,542,344,989,87838,3884,8844,48383]
}

df=pd.DataFrame(stock_market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_index_Price'])


X=df[['Interest_Rate','Unemployment_Rate']].astype(float)
Y=df['Stock_index_Price'].astype(float)

#with sklearn
regr=linear_model.LinearRegression()
regr.fit(X,Y)

print('Intercept: \n',regr.intercept_)
print('Coefficients:\n',regr.coef_)

#tkinter GUI




root=tk.Tk()

canvas1= tk.Canvas(root,width=500,height=300)
canvas1.pack()

#with sklearn
Intercept_result=('Intercept:',regr.intercept_)
label_Interecept=tk.Label(root,text=Intercept_result,justify='center')
canvas1.create_window(260,220,window=label_Interecept)
label_Interecept.config(font=("Times",14))

#with sklearn
Coefficients_result=('Coefficients:',regr.coef_)
label_Coefficients=tk.Label(root,text=Coefficients_result,justify='center')
canvas1.create_window(260,245, window=label_Coefficients)
label_Coefficients.config(font=("Times", 14))

# New Interest Rate label and input box
label1=tk.Label(root,text='type interest rate')
canvas1.create_window(100,100,window=label1)
label1.config(font=('Times',18))

e1=tk.Entry(root,text='create 1st entry box')
canvas1.create_window(380,100,window=e1)
e1.config(font=('Times',18))

l2=tk.Label(root,text='type unemployment rate')
canvas1.create_window(120,130,window=l2)
l2.config(font=('Times',18))

e2=tk.Entry(root)
canvas1.create_window(380,130,window=e2)
e2.config(font=('Times',18))

def values():
    global New_Interest_Rate
    New_Interest_Rate=float(e1.get())

    global New_Unemployment_Rate
    New_Unemployment_Rate=float(e2.get())

    Prediction_result=('Predicted stock price',regr.predict([[New_Interest_Rate,New_Unemployment_Rate]]))
    label_Prediction=tk.Label(root,text=Prediction_result,bg='Red')
    canvas1.create_window(260,280,window=label_Prediction)
    label_Prediction.config(font=("Times",18))

b1=tk.Button(root,text='predict stock index price',command=values,bg='yellow')
canvas1.create_window(270,180,window=b1)
b1.config(font=('Times',18))

# plot 1st scatter
f3=plt.Figure(figsize=(5,4),dpi=100)
ax3=f3.add_subplot(111)
ax3.scatter(df['Interest_Rate'].astype(float),df['Stock_index_Price'].astype(float),color='r')
scatter3=FigureCanvasTkAgg(f3,root)
scatter3.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax3.legend(['stock index rate'])
ax3.set_xlabel('Interest_rate')
ax3.set_title('interest rate vs stock index price')

# plot 2nd scatter
f4=plt.Figure(figsize=(5,4),dpi=100)
ax4=f4.add_subplot(111)
ax4.scatter(df['Unemployment_Rate'].astype(float),df['Stock_index_Price'].astype(float),color='g')
scatter4=FigureCanvasTkAgg(f4,root)
scatter4.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax4.legend(['stock_index price'])
ax4.set_xlabel(['unemployment_rate'])
ax4.set_title('unemployment_rate vs stock_index_price')

root.mainloop()