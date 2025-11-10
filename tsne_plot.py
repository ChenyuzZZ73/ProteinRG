import numpy as np
from embed.spectrum import spectrum_map
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import torch
from utill.fasta import fasta_file_to_df
import esm
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.to(device)
model.eval()  # disables dropout for deterministic results

#dd = fasta_file_to_df("data/split_5/generated_.fasta")
dd = pd.read_csv("data/split_3/val.csv",sep=',')
df = fasta_file_to_df("data/split_3/generated_Re.fasta")
#seq = df['sequence'].values
seq_1 = df['sequence'].values
seq_2 = dd['sequence'].values
#seq_all = seq[0:2000]
seq_1  = seq_1[0:500]
seq_2 = seq_2[0:500]


sequence_representations=[]
#sequence_representations.append(spectrum_map(seq_1))

#sequence_representations.append(spectrum_map(seq_2))
for seq in seq_1.tolist():
    results = model(batch_converter([("1", seq)])[2].to(device), repr_layers=[12], return_contacts=True)["representations"][12]
    re = results[0, 1: ((batch_converter([("1", seq)])[2] != alphabet.padding_idx).sum(1)) - 1].mean(0).to('cpu')
    re = re.detach().numpy()
    sequence_representations.append(re)

#for seq in seq_2.tolist():
#    results = model(batch_converter([("1", seq)])[2].to(device), repr_layers=[12], return_contacts=True)["representations"][12]
#    re = results[0, 1: ((batch_converter([("1", seq)])[2] != alphabet.padding_idx).sum(1)) - 1].mean(0).to('cpu')
#    re = re.detach().numpy()
#    sequence_representations.append(re)


re = np.array(sequence_representations)
print(re.shape)



'''t-SNE'''
tsne = TSNE(n_components=2)
#pca = PCA(n_components=2)
X_tsne_1 = tsne.fit_transform(re)

sequence_representations=[]
for seq in seq_2.tolist():
    results = model(batch_converter([("1", seq)])[2].to(device), repr_layers=[12], return_contacts=True)["representations"][12]
    re = results[0, 1: ((batch_converter([("1", seq)])[2] != alphabet.padding_idx).sum(1)) - 1].mean(0).to('cpu')
    re = re.detach().numpy()
    sequence_representations.append(re)

re = np.array(sequence_representations)
print(re.shape)
'''t-SNE'''
tsne = TSNE(n_components=2)
#pca = PCA(n_components=2)
X_tsne_2 = tsne.fit_transform(re)
#X_tsne = pca.fit_transform(re)
#print(pca.explained_variance_ratio_)
#y = dd['label'].values
#y_1 = np.array(y_1,dtype=int)
#y_2 =  d['label'].values
#y_2 = np.array(y_2,dtype=int)
#y = np.concatenate((y_1,y_2))
#y = df['name'].values
#y=[0]
#y_1 = df['label'].values
y_1 = np.zeros(500)
y_2 = np.ones(500)
#y_3 = np.ones(258)
#y_4 = np.zeros(258)
y = np.concatenate((y_1,y_2))
print(y.shape)
#为6个点配置颜色


#为6个点配置颜色
def get_color(labels):
    means = [5, 15, 25, 35, 45]
    colormap = plt.get_cmap('coolwarm')
    colors = ['r',"b"]
    color =[]
    for i in range(len(labels)):
        print(labels[i])
        color.append(colors[int(labels[i])])
    return color

from matplotlib.pyplot import MultipleLocator

figure=plt.figure(figsize=(5,5),dpi=80)
color=get_color(y)#为6个点配置颜色
x=np.append(X_tsne_1[:,0],X_tsne_2[:,0])#横坐标
y=np.append(X_tsne_1[:,1],X_tsne_2[:,1])#纵坐标
#x=X_tsne_1[:,0]#横坐标
#y=X_tsne_1[:,1]#纵坐标
plt.scatter(x,y,edgecolors='k',color=color)#绘制散点图。
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

x_major_locator=MultipleLocator(10)

y_major_locator=MultipleLocator(10)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.savefig('tsne.png', dpi=350)
plt.close()
