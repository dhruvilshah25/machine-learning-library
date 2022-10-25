#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


attribute_value = {}

def Get_Num(dataframe, value):
    dataframe.loc[dataframe['label'] == value].count()
    return

class Node:
    
    
    def __init__(self, attribute=None, values=None, label=None):
        self.attribute = attribute
        self.values = values
        self.next = {}

        self.label = label


# In[3]:


def Entropy(v, s):
    
    E = (v/s)*np.log2(v/s) * - 1
    
    
    return E


# In[4]:


def Information_Gain(total, s_size, value_numbers, c):
    a = 0
    i = 0
    
    while (i < value_numbers.size):
        a += (value_numbers[i]/s_size) * c[i]
        i += 1
        
    return total - a 


# In[5]:


def Gini_Index(p, s):
    GI = np.power(p/s,2)
    
    return GI


# In[6]:


def Majority_Error(p, s):
    ME = (s-p)/s
    
    return ME


# In[7]:


def Common_Label(data):
    return data['label'].value_counts().max()

def total_Value(label_v,num_rows,d):
    if d == 1:
        total_v = 0
        
        for v in label_v:
            total_v += Entropy(v, num_rows)
        return total_v
    
    if d == 2:
        total_v = 1
        for v in label_v:
            total_v -= Gini_Index(v, num_rows)
            
        return total_v
    
    if d == 3:
        total_v = 0
        for v in label_v:
            total_v += (num_rows-v)/num_rows
        return total_v


# In[8]:


def Get_Entropy(data, attributes, ttl_entropy):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            entropies = []

            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                label_counts = filtered_data['label'].value_counts()
                value_entropy = 0
                for label in label_counts:
                    value_entropy += Entropy(label, filtered_data.shape[0])
                entropies.append(value_entropy)
                
                
            attribute_info_gain = Information_Gain(ttl_entropy, data.shape[0], data[attribute].value_counts(), entropies)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain

    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain


# In[9]:




def Get_Gini(data,attributes, ttl_gini):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            gini_indexes = []

            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                label_counts = filtered_data['label'].value_counts()
                value_gini = 1
                for label in label_counts:
                    value_gini -= Gini_Index(label, filtered_data.shape[0])
                gini_indexes.append(value_gini)
            attribute_info_gain = Information_Gain(ttl_gini, data.shape[0], data[attribute].value_counts(), gini_indexes)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain

    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain


# In[10]:


def Get_Majority(data,attributes, ttl_majority):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            majority_errors = []

            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                label_counts = filtered_data['label'].value_counts()
                value_majority = 1
                for label in label_counts:
                    value_majority += Majority_Error(label, filtered_data.shape[0])
                majority_errors.append(value_majority)
            attribute_info_gain = Information_Gain(ttl_majority, data.shape[0], data[attribute].value_counts(), majority_errors)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain

    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain


# In[11]:






def ID3(data, attributes, ttl_entropy, defined_depth, d):
    if defined_depth == 0:
        dd = Node(label = Common_Label(data))
        return dd

    if len(pd.unique(data['label'])) == 1:
        dd1 = Node(label=pd.unique(data['label'])[0])
        return dd1

    if len(attributes) == 0:
        dd2 = Node(label= Common_Label(data))
        return dd2


    if d == 1:
        root_node, new_error = Get_Entropy(data, attributes, ttl_entropy)
    if d == 2:
        root_node, new_error = Get_Gini(data, attributes, ttl_entropy)
    if d == 3:
        root_node, new_error = Get_Majority(data, attributes, ttl_entropy)
    for value in attribute_value[root_node.attribute]:

        is_val = data[root_node.attribute] == value
        value_subset = data[is_val]

        length = len(value_subset.index)
        if length == 0:
            root_node.next[value] = Node(label= Common_Label(data))
        else:
            new_attributes = attributes[:]
            new_attributes.remove(root_node.attribute)
            new_depth = defined_depth -1
            root_node.next[value] = ID3(value_subset,new_attributes, new_error, new_depth, d)
    return root_node


# In[12]:


def Get_Accuracy(root_node, data):
    wrong_predictions = 0
    i = 0
    
    while i < data.shape[0]:
        current_node = root_node
        
        while current_node.label == None:
            current_node = current_node.next[data[current_node.attribute].iloc[i]]
            
        if current_node.label != data['label'].iloc[i]:
            wrong_predictions += 1
        i += 1
    return wrong_predictions/data.shape[0]


# In[13]:


def Get_Data(data, columns):
    for column in columns:
        attribute_value[column] = pd.unique(data[column])


# In[49]:


def main():
    print('Please enter the Depth(1-6)')
    depth = int(input())
    print('Please select the attribute, 1 = Entropy, 2 = Gini_Index, 3 = Majority_Error')
    decider = int(input())
    car_columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    data = pd.read_csv("train.csv", names=car_columns)
    test_data = pd.read_csv("test.csv", names=car_columns)
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    total_error = total_Value(total_label_values, num_rows, decider)
    car_columns.remove('label')
    Get_Data(data, car_columns)
    root_node = ID3(data, car_columns, total_error, depth, decider)
    train_error = Get_Accuracy(root_node, data)
    test_error = Get_Accuracy(root_node, test_data)
    print('The selected depth is: ' + str(depth))
    print('training error = ' + str(train_error) + ' testing error = ' + str(test_error))


# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




