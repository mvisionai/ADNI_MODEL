
import  numpy as np
data = np.array([[1, 2, 3],[6,56,9],[4,10,45]])

di=np.take(data, np.random.permutation(len(data)), axis=0, out=data)

print(di)