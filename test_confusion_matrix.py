from sklearn.metrics import confusion_matrix
import numpy as np

a = np.array([1,1,0,0,0],dtype=np.int64)
b = np.ones_like(a,dtype=np.float32)
c = confusion_matrix(a,b)
tp = c[1,1]
np = c[0,1]
tn = c[0,0]
fn = c[1,0]
print(c)

