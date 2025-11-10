import matplotlib.pyplot as plt
import pandas as pd
fig, ax = plt.subplots() # 创建图实例
dd = pd.read_csv("360.csv",sep=',')
df = pd.read_csv("480.csv",sep=',')
d = pd.read_csv("metric.csv",sep=',')


x =[]
y=[]
for i in range(240):
    if(i % 3 ==0):
        x.append(dd["Step"][i])
        y.append(dd["MRR"][i])

x1 = x
y1 = y

ax.plot(x1, y1, label='Dimension = 360') # 作y1 = x 图，并标记此线名为linear

x =[]
y=[]
for i in range(240):
    if (i % 5 == 0):
        x.append(df["Step"][i])
        y.append(df["MRR"][i])
y3 = y
x3 = x

ax.plot(x3, y3, label='Dimension = 480')

x =[]
y=[]
for i in range(240):
    if (i % 5 == 0):
        x.append(d["Step"][i])
        y.append(d["MRR"][i])
y2 = y
x2 = x

ax.plot(x2, y2, label='Dimension = 50')
ax.set_xlabel('Training Step') #设置x轴名称 x label
ax.set_ylabel('MMD') #设置y轴名称 y label
#ax.set_title('Simple Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

#plt.show() #图形可视化
plt.savefig('dimension.png', dpi=350)