import re
from sklearn import tree
from sklearn.cross_validation import train_test_split
from scipy import *

#put all files in one big string
big_string=''
files=[]
for j in range(1,6):
    f=open("/Users/nuh-ufmg/Downloads/lp"+str(j)+".data")
    for line in f:
        big_string += ''.join(line)
    #the following step is important to keep the template when concatenating the files
    big_string += '\n\n\n'
big_string = big_string[:-6]


#divide the instances according with the divisor (3 newlines), for each instance split its lines, so finally split
#the cells in order to transform each instance in a matrix
instance_matrices = [[re.split('\t',i) for i in re.split('\n\t',j)[1:]] for j in re.split('\n\n\n',big_string)]

#define the output values
Y = re.findall('[a-z_]+',big_string)

#determine the output classes as normal or not normal states
Y = ['normal' if (_ == 'normal' or _=='ok') else 'not_normal' for _ in Y ]

# transform the matrix in order to get better results
def define_transformation(values,flag):
    if flag==1:

        sorted_values = [(int(v)) for v in values ]
        sorted_values.sort()

        abs_sum = sum([ abs(i) for i in values])
        abs_range = abs(min(sorted_values) - max(sorted_values))

        return [abs_sum,abs_range]
    else:
        return values

X=[]
#create the entry of the classification algorithm transforming the matrix, if the flag of the function define_transformation
#is set to one

for matrix in instance_matrices:
    entry = []
    for pz in range(0,5):
        entry += define_transformation([j[pz] for j in matrix],1)
    X.append(entry)


###########################################################################
#imported code of the first excercise

#split the data in train and test values
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=10)

#create a decision tree to classify the waveforms. The tree was pruned in order to keep only the eight more significant
#leafs
clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)

#train the decision tree and show the score of the process in order to judge the quality of the prediction.
print(clf.fit(X_train, Y_train).score(X_test, Y_test))

#export tree in a representation form
tree.export_graphviz(clf, out_file='tree2.dot',feature_names=None,class_names=['normal','not_normal'])

#print the graph of the tree
os.system("dot -Tpng -otree2.png -v tree2.dot")
