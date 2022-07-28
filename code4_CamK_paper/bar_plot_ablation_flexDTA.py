import numpy as np
g1=[111,
74,
197,
153,
55,
43,
85,
70,
51,
71,
43,
98,
51,
40,
69,
50,
58,
68,
83,
27,
24,
53,
65,
107,
70,
67,
55,
99,
62,
85,
74,
84,
86,
72,
85,
40,
59]

g2=[21,
7,
38,
3,
10,
10,
49,
51,
42,
42,
43,
64,
45,
41,
15,
9,
9,
17,
6,
8,
15,
20]

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind 
s,p = ttest_ind(g1,g2)
print(s,p)
xpos=np.array([1,2])
fig, ax = plt.subplots()
g1=np.array(g1)/(0.3*0.3)
g2=np.array(g2)/(0.3*0.3)
m1=np.mean(g1)
m2=np.mean(g2)
e1=np.std(g1)/(len(g1)**0.5)
e2=np.std(g2)/(len(g2)**0.5)
ax.bar(xpos,np.array([m1,m2]), yerr=[e1,e2], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean cell density ($cells/mm^2$)',fontsize=15)
ax.set_xticks(xpos)
print('Mean1:',m1)
print('Mean2:',m2)
ax.set_xticklabels(['Hm3Dq', 'Hm3Dq + Flex-DTA'])
ax.set_title('Densitity of Inhibitory neurons',fontsize=19)
ax.yaxis.grid(True)
plt.show()
