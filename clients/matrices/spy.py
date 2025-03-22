import scipy.sparse as sprs
import matplotlib.pyplot as plt
Matrix=sprs.rand(10,10, density=0.1, format='csr')
plt.spy(Matrix)
plt.show()