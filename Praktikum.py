import numpy as np
import matplotlib.pyplot as plt
A=np.arange(1,13).reshape(3,4)
print(A)
B=np.random.randn(3,4)
print(B)
# 1.3: Führen Sie die folgenden Operationen aus:
#### Warum lässt sich das Punktprodukt von A und B nicht Bilden?
####### Weil Shap von A passt beim Punktprodukt shape von B ,
#A.dot(B)
###### Überscheiben Sie B mit den elementweisen Quadratwurzeln von A.######### Berechnen Sie eine Matrix C aus dem Punktprodukt von A und der Transponierten von B
C=A.dot(B.T)
print(C)
######### Geben Sie die Shape (Dimensionen) von C aus
print(C.shape)
########## Geben Sie die erste Spalte von C aus.
print(C[:,0])
######### Geben Sie die Diagonale von C aus.
print(C.diagonal())
########## Geben Sie das Ergebn"is der Inversion von C aus.
print(np.linalg.inv(C))
########### Multiplizieren Sie A und B elementweise.
print(A*B)
######### Berechnen Sie den Mittelwert von B.
print (B.mean())
############ Summieren Sie alle Spalten von B auf.
print(B.sum())
###########Erstellen Sie eine Matrix D, die aus der 3, 1, 2 spalte
D=A[:,(3,1,2)]
print(D)
#########Führen Sie die Matrixmultipikationen A B.T und B A.T aus, geben Sie die Dimensionen der Ergebnisse aus.
#########print(B.dot(A.T))
print(B.dot(A.T).shape)
print(A.dot(B.T))
print(A.dot(B.T).shape)
############# Überscheiben Sie B mit den elementweisen Quadratwurzeln von A.
B=np.sqrt(A)
print(B)
########Überscheiben Sie B mit B*B und vergleichen Sie danach elementweise A mit B. Was stellen Sie fest?
B=B*B
print(B==A)
#####################die Geauigkeit ist nicht perfekt !
########## Geben Sie den kleinsten Wert von B aus.

print(B.min())
B=np.random.randn(3,4)
########### Geben Sie den Index des Wertes von B aus, welcher am weitesten von 0 entfernt ist.
ind = np.unravel_index(np.argmax(np.abs(B), axis=None), B.shape)
print(B)
print(ind)
############# Geben Sie alle ungeraden Zahlen aus A in umgekehrter Reihenfolge aus.
print((A.flatten()[A.flatten()%2==1])[::-1])

########### Erweitern sie A um [13 , 14 , 15, 16], so das A nun die Dimension 4x4 hat."
A=np.vstack((A,[13,14,15,16]))
print(A)
#########Generieren Sie die Vektoren v und b aus Einsen, sodass die Operation Av + b möglich ist und einen Vektor ergibt. 
v=b=np.ones((4,1))
print(A*v+b)
############# Berechnen Sie die L1 und L2 Distanzen zwischen A und B.
B=np.random.rand(4,4)
print(np.linalg.norm(A-B))
print(np.linspace(0,1,5))
# 1.4: MatplotLib
def myGrid(a,b,n):
    x=np.linspace(a,b,n)
    xx,yy=np.meshgrid(x,x)
    return np.vstack((xx.flatten(),yy.flatten())).T
xy=myGrid(0,1,5)
plt.scatter(xy[:,0],xy[:,1])   
xy=myGrid(-1,1,3)
print(xy)
print(type(xy))
print (xy.dtype)
from timeit import default_timer as timer
start=timer()
myGrid(0,1,1000)
end=timer()
print(end-start)
from matplotlib.image import imread

###Bild als Matrix
n="a.jpeg"
im=imread(n)
im.shape
R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
import matplotlib.colors as colors
HSV=colors.rgb_to_hsv(im)
H=HSV[:,:,0]


plt.close()
hist=plt.hist(H.flatten())
