import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

plt.style.use("seaborn")


##-------------------------------------------------KNN ALGO------------------------
def distance(x,point):
	d = np.sqrt(sum((x-point)**2))
	return d

def knn(X,Y,point,k=5):
	dist = []
	m = X.shape[0]

	for i in range(m):
		d = distance(X[i],point)
		dist.append((d,Y[i]))

	dist = sorted(dist,key= lambda d: d[0])
	dist = np.array(dist[:k])

	#adding voting part
	classes = np.unique(np.array(Y))
	# print(classes)

	votes = np.zeros(len(classes))

	for d in dist:#farther points will contribute less in voting part
		votes[int(d[1])]+= 1/(d[0])

	# print(votes)
	pred = np.argmax(votes)

	return pred



#-----------------------------------------------------------------------------------------------


pdx = pd.read_csv("Diabetes_XTrain.csv")
# print(pdx.describe())

pdy = pd.read_csv("Diabetes_YTrain.csv")
# print(pdy.shape)

no_of_classes = np.unique(np.array(pdy))
# print(no_of_classes)

decider = np.unique(np.array(pdy),return_counts=True)
suffering = decider[1][1]
not_suffering = decider[1][0]

#since mean and stds are not 0 and 1 so normalization  required

#task 1: plotting bar graph
x = np.arange(2)
plt.bar(x,[suffering,not_suffering])
plt.xticks(x,['suffering','not suffering'])
plt.xlabel("classes in dataset")
plt.ylabel("patient counts")
plt.show()

#normalise the data
pdx = (pdx - np.mean(pdx,axis=0))/np.std(pdx,axis=0)
# print(pdx.describe())

X = np.array(pdx)
Y = np.array(pdy)
Y = Y.reshape(-1)

X_test = pd.read_csv('Diabetes_Xtest.csv').values
X_test = (X_test-np.mean(X_test,axis=0))/np.std(X_test,axis=0)
# print(X_test.describe())

y_pred = []
m = X_test.shape[0]
for i in range(m):
	y_pred.append(knn(X,Y,X_test[i],5))

print(y_pred)

pd_test = pd.DataFrame(y_pred)
pd_test.to_csv('ans.csv',index_label='Outcome',index=False)





