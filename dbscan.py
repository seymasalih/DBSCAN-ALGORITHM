from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

"""Algoritma rastgele bir noktadan başlar. Noktaları ziyaret eder ve core pointi belirler.
 DR olan başka noktayı aynı sınıfa ekler. Başka nokta kalmayınca kadar"""

#centers = [(0, 4), (5, 5) , (8,2)]
#cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200,cluster_std=1.0,center_box=(-10.0, 10.0), n_features=2, random_state=1)
#eğer merkezlerin random oluşturulmasını isteseydik center_box indexinin alıp center boş bırakırdık

def determine_core_point(eps,minPts, df, index):
    
    x, y = df.iloc[index]['X']  ,  df.iloc[index]['Y']
    
    #yarıçap içindeki noktaları kontrol ederiz uzaklıklarının mutlak değerini alarak
    temp =  df[((np.abs(x - df['X']) <= eps) & (np.abs(y - df['Y']) <= eps)) & (df.index != index)]
    
    #core , border, noise belirle
    if len(temp) >= minPts:
        
        return (temp.index , True, False, False)
    
    elif (len(temp) < minPts) and len(temp) > 0:
       
        return (temp.index , False, True, False)
    
    elif len(temp) == 0:
        
        return (temp.index , False, False, True)
    
def cluster_with_stack(eps, minPts, df):
    
    #küme numarasını başlatma
    C = 1

    current_stack = set() #set tipinde
    unvisited = list(df.index) #df nin değerlerini alır
    clusters = []
    
    
    while (len(unvisited) != 0): #ziyaret edilmeyen nokta varsa

        #identifier for first point of a cluster
        first_point = True
        
        #ziyaret edilmeyenlerden random seç
        current_stack.add(random.choice(unvisited))
        
        while len(current_stack) != 0: #current_stack boş değilse
            
            #current stackin son elemanı atanır
            curr_idx = current_stack.pop()
            
            #format core ya da border mi kontrol et
            neigh_indexes, iscore, isborder, isnoise = determine_core_point(eps, minPts, df, curr_idx)
            
            #ilk nokta border ise
            if (isborder & first_point):
                #ilk border pointin çevresindekileri noise veya komşu olarak işaretliyoruz
                clusters.append((curr_idx, 0))
                clusters.extend(list(zip(neigh_indexes,[0 for _ in range(len(neigh_indexes))])))
                
                #label as visited
                unvisited.remove(curr_idx)
                unvisited = [e for e in unvisited if e not in neigh_indexes]
    
                continue
                
            unvisited.remove(curr_idx) #ziyaret edilen nokta curr_idx ten kaldırılır
            
            
            neigh_indexes = set(neigh_indexes) & set(unvisited) #look at only unvisited points
            
            if iscore: #point core ise
                first_point = False
                
                clusters.append((curr_idx,C)) #C kümesine ekle
                current_stack.update(neigh_indexes)

            elif isborder: #border ise
                clusters.append((curr_idx,C))
                
                continue

            elif isnoise: #noise ise
                clusters.append((curr_idx, 0))
                
                continue
                
        if not first_point:
            #küme numarası arttırılır
            C+=1
            
    
    return clusters

#Seçilen bir noktanın yakınında olması istenilen sayıda komşu nokta sayısı belirlenir ve bunu sağlayan en küçük 𝜀 saptanır.
eps = 0.6

minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] ) #data , etiket

clustered = cluster_with_stack(eps, minPts, data) #fonksiyona belli girdiler gönderilir ve atanır

idx , cluster = list(zip(*clustered))#yinelenebilir bağımsız değişkenleri list olarak alır.
cluster_df = pd.DataFrame(clustered, columns = ["idx", "cluster"]) #data , etiket

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=20, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
   

