##################################################
## This is Assignment II project of EH2745 Computer Applications in Power Systems
## K-Nearest Neighbour algorithm to analyze a
# database from a sample power system 
##################################################
## {2019, MIT License}
##################################################
## Author: Fahrur Rozzi
## Copyright: Copyright 2019, Assignment 2
## Credits: ["Lars Nordström", "Oscar Utterbäck", "Fahrur Rozzi"]
## License: MIT License
## Version: 1.0.1
## Maintainer: Fahrur Rozzi
## Email: fahrurrozzi18@gmail.com
## Status: Production
##################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import random

from tkinter import *
from tkinter import ttk
import tkinter
from tkinter import filedialog


# GUI of the application
root = Tk()
root.title("Assignment 2")
content = ttk.Frame(root)
frame = ttk.Frame(content, borderwidth=5, relief="sunken", width=100, height=100)
img = PhotoImage(file='KTH.png')
panel = ttk.Label(content, image=img)
content.grid(column=1, row=0)
frame.grid(column=1, row=0, columnspan=3, rowspan=2)
panel.grid(column=1, row=0, columnspan=3, rowspan=2)

# Insert File
def Measurement_file():
    Measurement_file = filedialog.askopenfilename()
    global Measurement_files, Message_Measurement_files
    Measurement_files = Measurement_file
    Message_Measurement_files = "Successfully added Measurement File"

def Analog_file():
    Analog_file = filedialog.askopenfilename()
    global Analog_files, Message_Analog_file
    Analog_files = Analog_file
    Message_Analog_file = "Successfully added Analog File"

# GUI Layout
Button_EQ = Button(root, text="Import Measurement File", command=Measurement_file)
Button_EQ.grid(column=1, row=3, sticky='nesw')
Button_SSH = Button(root, text="Import Analog File", command=Analog_file)
Button_SSH.grid(column=1, row=4, sticky='nesw')
Button_Run = Button(root, text="Run", command=root.destroy)
Button_Run.grid(column=1, row=5,sticky='nesw')

root.mainloop()




# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import random

def import_data(dataset):
#    X = dataset.groupby('name')['time'].nunique()
    attribute_list = ['CLAR','AMHE','WINL','BOWM','TROY','MAPL','GRAN','WAUT','CROSS']
    attributes_list = ['TIME']
    for atr in attribute_list:
        attributes_list.append(atr + '_VOLT')
        attributes_list.append(atr + '_ANG')
    train_data = []

    train_data.append(dataset.loc[dataset['name']=='CROSS_VOLT', 'time'].values)
    
    for atr_index in range(len(attributes_list)-1) :
        train_data.append(dataset.loc[dataset['name']==attributes_list[atr_index+1], 'value'].values)
    
    train_array = np.transpose(np.array([train_data[0]]))
    
    for train_index in range(len(train_data)-1):
        train_array = np.column_stack((train_array, np.transpose(np.array([train_data[train_index+1]]))))
    
    train_arrays = train_array[:,1:]
    train_array_table = pd.DataFrame(train_array[:,:], columns=attributes_list)
    
    return train_arrays, train_array_table


# Find Min Value
def minValue (list):
    min_value = 1
    for i in range(1,len(list),2):
        if list[i] < min_value :
            min_value = list[i]            
    return min_value

#K-Label
def klabel(attributes):
    state = ""
    sum579 = attributes[8]+attributes[12]+attributes[16]
#    print(sum579, attributes[8], attributes[12], attributes[16]  )
    min_value = minValue(attributes)
#    buses 5, 7 and 9 have low voltages
    if sum579<2.92 and sum579>2.60:
        state = "High Load"
#    buses 6, 7 and 8 have high voltages
    elif sum579>2.97:
        state = "Low Load"
#    if one of these flows is too low, it means that the generator is offline
    elif attributes[1]-attributes[7]<0.01 or attributes[3]-attributes[15]<0.01 or attributes[5]-attributes[11]<0.01:
        state = "Shut Down"
    elif min_value<0.85:
        state = "Disconnect"    
    return state


# Initialize Centroid
def RandomList(train_array_set):
#   Random points generator
    
#    random_list = np.zeros((1, 4))
#    random_list.fill(1000)
    
    global random_list 
    
#    condition = True
    
    for i in range(len(train_array_set)):
        temp = random.randint(0,len(train_array_set)-1)
        random_values = train_array_set[temp]
        temp_state = klabel(random_values)
        if temp_state == "High Load":
            random_list[0] = temp
        elif temp_state == "Shut Down":
            random_list[1] = temp
        elif temp_state == "Low Load":
            random_list[2] = temp 
        elif temp_state == "Disconnect":
            random_list[3] = temp
        
#        print(random_list)
        if 1000 not in random_list:
            break
        
    return random_list

        

# Calculates the distances to be later compared with the tolerance       
def toleranceCalculator(old_centroids, new_centroids):
    cluster_distances1 = np.zeros((k_number, 18))

    for i in range(k_number):
        for j in range(18):
            temp_val = (old_centroids[i][j]-new_centroids[i][j])*(old_centroids[i][j]-new_centroids[i][j])
            cluster_distances1[i][j] = math.sqrt((temp_val*temp_val))
    
    return cluster_distances1

# Finds the maximum element in double[][]
def maxValue(list):
    max = list[0][0]
    
    for i in range(len(list)):
        for j in range(len(list[i])):
            if list[i][j]>max:
                max = list[i][j]
    
    return max

# Initizialize
def initialize():
#    Assign random centroids
    global random_list
    random_list = RandomList(train_array)
#    random_list = [128, 199, 80, 27]
#    random_list = [129, 149, 65, 103]
    
#    print("Random List", random_list)
    
    for i in range(18) :
        centroids[0][i]=train_array[random_list[0]][i]
        centroids[1][i]=train_array[random_list[1]][i]
        centroids[2][i]=train_array[random_list[2]][i]
        centroids[3][i]=train_array[random_list[3]][i]
        
    cluster_definition = []
    
    for i in range (len(train_array)):
        dist1 = 0
        dist2 = 0
        dist3 = 0
        dist4 = 0
        
#        Calculate distance
        for j in range(18):
            dist1 += ((centroids[0][j]-train_array[i][j])*(centroids[0][j]-train_array[i][j]))
            dist2 += ((centroids[1][j]-train_array[i][j])*(centroids[1][j]-train_array[i][j]))
            dist3 += ((centroids[2][j]-train_array[i][j])*(centroids[2][j]-train_array[i][j]))
            dist4 += ((centroids[3][j]-train_array[i][j])*(centroids[3][j]-train_array[i][j]))
        
        dist1 = math.sqrt(dist1)
        dist2 = math.sqrt(dist2)
        dist3 = math.sqrt(dist3)
        dist4 = math.sqrt(dist4)
        
        distArray = [dist1, dist2, dist3, dist4]
        min_index = distArray.index(min(distArray))
#        print(i, min_index)
        cluster_definition.append(min_index)
        
    
    
    for i in range(len(cluster_definition)):
        if cluster_definition[i] == 0 :
            cluster1.append(train_array[i].tolist())
        elif cluster_definition[i] == 1 :
            cluster2.append(train_array[i].tolist())
        elif cluster_definition[i] == 2 :
            cluster3.append(train_array[i].tolist())
        elif cluster_definition[i] == 3 :
            cluster4.append(train_array[i].tolist())       
        

#  Calculates the new centroids
def calculateCentroids():
    global centroids
    temp_centroids = np.zeros((k_number, 18))
    
    for i in range(len(cluster1)):
        for j in range(18):
            temp_centroids[0][j] += cluster1[i][j]/len(cluster1)
            
    for i in range(len(cluster2)):
        for j in range(18):
            temp_centroids[1][j] += cluster2[i][j]/len(cluster2)
            
    for i in range(len(cluster3)):
        for j in range(18):
            temp_centroids[2][j] += cluster3[i][j]/len(cluster3)
    
    for i in range(len(cluster4)):
        for j in range(18):
            temp_centroids[3][j] += cluster4[i][j]/len(cluster4)
    
    centroids = temp_centroids

# k-means Clustering Algorithm
def kClusters():
    
    global centroids
#    print("falam", centroids)
    counter = 0
    maxChange = 1000
    
    while maxChange > tolerance :
#        print("start while", centroids)
#        temp_centroids = np.zeros((k_number, 19))
        temp_centroids = centroids.copy()
        
#        print("cluster 1", cluster1)
        cluster1.clear()
        cluster2.clear()
        cluster3.clear()
        cluster4.clear()
        data_wcluster.clear()
#        print("cluster 1", cluster1)
#        print("################")
#        print(train_array)
             
        
        cluster_definition = []
    
        for i in range (len(train_array)):
            dist1 = 0
            dist2 = 0
            dist3 = 0
            dist4 = 0
            
    #        Calculate distance
            for j in range(18):
                
                dist1 += ((centroids[0][j]-train_array[i][j])*(centroids[0][j]-train_array[i][j]))
                dist2 += ((centroids[1][j]-train_array[i][j])*(centroids[1][j]-train_array[i][j]))
                dist3 += ((centroids[2][j]-train_array[i][j])*(centroids[2][j]-train_array[i][j]))
                dist4 += ((centroids[3][j]-train_array[i][j])*(centroids[3][j]-train_array[i][j]))
            
            dist1 = math.sqrt(dist1)
            dist2 = math.sqrt(dist2)
            dist3 = math.sqrt(dist3)
            dist4 = math.sqrt(dist4)
            
            distArray = [dist1, dist2, dist3, dist4]
            min_index = distArray.index(min(distArray))
#            print(i, min_index)
            cluster_definition.append(min_index)
            
        
        for i in range(len(cluster_definition)):
            if cluster_definition[i] == 0 :
                cluster1.append(train_array[i].tolist())
                data_wcluster.append([train_array[i].tolist(), "1", klabel(centroids[0])])
            elif cluster_definition[i] == 1 :
                cluster2.append(train_array[i].tolist())
                data_wcluster.append([train_array[i].tolist(), "2", klabel(centroids[1])])
            elif cluster_definition[i] == 2 :
                cluster3.append(train_array[i].tolist())
                data_wcluster.append([train_array[i].tolist(), "3", klabel(centroids[2])])
            elif cluster_definition[i] == 3 :
                cluster4.append(train_array[i].tolist())
                data_wcluster.append([train_array[i].tolist(), "4", klabel(centroids[3])])
        
#        Calculates the new centroids, again based on the cluster information
        calculateCentroids()
#        print("after calculate", centroids)
        
#        Calculates the distance between centroids
        centroid_distances = toleranceCalculator(temp_centroids, centroids)
        
        maxChange = maxValue(centroid_distances)
        
#        print(maxChange, tolerance)
        
        counter += 1
    
#    print("iterated", counter, "times")
        
#Cluster frequency counter
def clusterFrequenccy(list, string_target):
    counter = 0
    for i in range(len(list)):
        if list[i][1][1] == string_target :
            counter += 1
    return counter
    

#--------------- METHODS for K-NN    ----------------------



#	 kNN algorithm
def knnAlgorithm():
    for n in range(len(test_array)):
#        For data storage
        distance_cluster_knn = []
        
#        Distance between one value of test set and all training set, including the cluster
        
        for i in range(len(train_array)):
#            calculate the distance
            distance_testset = 0
            
            for j in range(18):
                distance_testset += ((test_array[n][j]-train_array[i][j])*(test_array[n][j]-train_array[i][j]))
            
#            store the cluster of the training set value
            distance_cluster_knn.append([math.sqrt(distance_testset), data_wcluster[i]])
		
				
			
#		Find the k nearest objects in the training set
#								
#		Step 1. Sort the data (distance, cluster) from smallest distance to largest
#		Collections.sort(distance_cluster, new DistanceComparator());
						
#		Step 3. Find the k nearest objects (k smallest distances)
#		OBS: Here k is not 4
#		OBS: k for kNN is defined by k=n^(1/2) where n is the amount of values in training set
			
        temp_val = math.sqrt(len(train_array))
        k_nn = round(temp_val)
        
        distance_cluster_knn = sorted(distance_cluster_knn)
        
#       KNN nearest cluster
        nearest_cluster = []
        
        for i in range(k_nn):
            nearest_cluster.append(distance_cluster_knn[i])
          
#       Step 4. Analyze which is the one with most incidence			
#		Calculate the frequency of each cluster in k nearest objects
        
        freq1 = clusterFrequenccy(nearest_cluster, "1")
        freq2 = clusterFrequenccy(nearest_cluster, "2")
        freq3 = clusterFrequenccy(nearest_cluster, "3")
        freq4 = clusterFrequenccy(nearest_cluster, "4")
        
#       Define which cluster is the one with most frequency
        freq_vector = [freq1, freq2, freq3, freq4]
        max_index = freq_vector.index(max(freq_vector))
        
        if max_index == 0 :
            test_wcluster.append([test_array[n], "1", klabel(centroids[0])])
        elif max_index == 1 :
            test_wcluster.append([test_array[n], "1", klabel(centroids[1])])
        elif max_index == 2 :
            test_wcluster.append([test_array[n], "1", klabel(centroids[2])])
        elif max_index == 3 :
            test_wcluster.append([test_array[n], "1", klabel(centroids[3])])
    
if __name__ == "__main__":
    # Importing the measurements dataset
    dataset = pd.read_csv('measurements.csv')
    dataset1 = pd.read_csv('analog_values.csv')
    
    train_array, train_array_table = import_data(dataset)
    test_array, test_array_table = import_data(dataset1)
    
    # Initialize 
    k_number = 4;
    
    centroid_distance = np.zeros((k_number, 18))
    centroids = np.zeros((k_number, 18))
    
    # Distances
    dist1 = 0
    dist2 = 0
    dist3 = 0
    dist4 = 0
    
    # Clusters
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    
    # Data with cluster
    data_wcluster = []
    test_wcluster = []
    
    # Tolerance and random value
    tolerance = 0.00000000000001   
    random_list = [1000] * k_number
    
#    KMeans Clustering
    initialize()
    calculateCentroids()       
    kClusters()

#   KNN
    knnAlgorithm() 
    
#    Result
    kmeans_list = []
    knn_list = []
    for i in range(len(train_array)):
        kmeans_list.append(data_wcluster[i][2])    
    for i in range(len(test_array)):
        knn_list.append(test_wcluster[i][2])
    train_array_table['Condition'] = kmeans_list
    test_array_table['Condition'] = knn_list 
  


# GUI for displaying result
# root = Tk()
# root.title("Table Result")
# height = len(train_array)
# width = 18


# Label(root, text="Y-BUS MATRIX").grid(row=0, column=math.floor(width/2))
# Label(root, text="").grid(row=1, column=math.floor(width/2))

# scrollbar = Scrollbar(root, orient= VERTICAL)


# for i in range(height): #Rows
#     for j in range(width): #Columns
#         b = Label(root, text=str(train_array[i][j]).replace("(","").replace(")",""))
#         b.grid(row=i+2, column=j, pady=5 ,padx=10, sticky="n")



# b.config(yscrollcommand=scrollbar.set)
# scrollbar.config(command=listbox.yview)

# Label(root, text="").grid(row=height+3, column=math.floor(width/2))
# yscroll.grid(row=0, column = 1, rowspan = 2)
# Button(root, text="Quit", command=root.destroy).grid(row=height+4, column=math.floor(width/2))


# mainloop()	

'''
Table Result
'''
try:
    import Tkinter as tk
    import tkFont
    import ttk
except ImportError:  # Python 3
    import tkinter as tk
    import tkinter.font as tkFont
    import tkinter.ttk as ttk


class MultiColumnListbox(object):
    """use a ttk.TreeView as a multicolumn ListBox"""

    def __init__(self, list1, list2):
        self.tree = None
        self._setup_widgets()
        self._build_tree(list1, list2)

    def _setup_widgets(self):
        s = """TABLE RESULT"""
        msg = ttk.Label(wraplength="4i", justify="center", anchor="n",
            padding=(10, 2, 10, 6), text=s)
        msg.pack(fill='x')
        container = ttk.Frame()
        container.pack(fill='both', expand=True)
        # create a treeview with dual scrollbars
        self.tree = ttk.Treeview(columns=header, show="headings")
        vsb = ttk.Scrollbar(orient="vertical",
            command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal",
            command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky='nsew', in_=container)
        vsb.grid(column=1, row=0, sticky='ns', in_=container)
        hsb.grid(column=0, row=1, sticky='ew', in_=container)

        Button_Run = Button(root, text="Close", command=root.destroy)
        Button_Run.pack()

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

    def _build_tree(self, list1, list2):
        for col in list1:
            self.tree.heading(col, text=col.title(),
                command=lambda c=col: sortby(self.tree, c, 0))
            # adjust the column's width to the header string
            self.tree.column(col,
                width=tkFont.Font().measure(col.title()))

        for item in list2:
            self.tree.insert('', 'end', values=item)
            # adjust column's width if necessary to fit each value
            for ix, val in enumerate(item):
                col_w = tkFont.Font().measure(val)
                if self.tree.column(list1[ix],width=None)<col_w:
                    self.tree.column(list1[ix], width=col_w)

def sortby(tree, col, descending):
    """sort tree contents when a column header is clicked on"""
    # grab values to sort
    data = [(tree.set(child, col), child) \
        for child in tree.get_children('')]
    # if the data to be sorted is numeric change to float
    # data =  change_numeric(data)
    data_update = data
    # for i in range(len(data)):
    #     data_update.append((int(float(data[i][0])), data[i][1]))
    # now sort the data in place
    data_update.sort(reverse=descending)
    for ix, item in enumerate(data_update):
        tree.move(item[1], '', ix)
    # switch the heading so it will sort in the opposite direction
    tree.heading(col, command=lambda col=col: sortby(tree, col, \
        int(not descending)))

# the test data ...

header = ['TIME', 'Condition']
kmeans_list = []

knn_list = []

times = train_array_table['TIME'].tolist()
for i in range(len(train_array)):
    kmeans_list.append((times[i], data_wcluster[i][2]))

for i in range(len(test_array)):
    knn_list.append((times[i], test_wcluster[i][2]))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("K-means Clustering")
    listbox = MultiColumnListbox(header, kmeans_list)
    root.mainloop()	

    root = tk.Tk()
    root.title("K-nearest neighbors")
    listbox = MultiColumnListbox(header, knn_list)
    root.mainloop()	
	


