#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers,matrix


# In[2]:


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.ndimage import interpolation, gaussian_filter


# In[3]:


f_digits_vec = "digits4000_txt/digits4000_digits_vec.txt"
f_digits_label = "digits4000_txt/digits4000_digits_labels.txt"
f_digits_trainset = "digits4000_txt/digits4000_trainset.txt"
f_digits_testset = "digits4000_txt/digits4000_testset.txt"


# In[4]:


digits_vec = np.loadtxt(f_digits_vec)
digits_label = np.loadtxt(f_digits_label)
digits_trainset = np.loadtxt(f_digits_trainset)
digits_testset = np.loadtxt(f_digits_testset)


# In[5]:


print(digits_vec.shape)
print(digits_label.shape)
print(digits_trainset.shape)
print(digits_testset.shape)


# In[8]:


import cv2 
SZ = 28
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


# In[9]:


#test deskew function
'''
fig1b, axs1b = plt.subplots(1,2, figsize=(10,20))
first_image2 = digits_vec[102]
pixels2 = first_image2.reshape((28, 28))
desk_image2 = deskew(pixels2)
axs1b[0].imshow(pixels2, cmap='gray')
axs1b[0].set_title('Before Deskewing')
axs1b[1].imshow(desk_image2, cmap='gray')
axs1b[1].set_title('After Deskewing')
'''

# In[11]:


import scipy
from scipy.ndimage import interpolation, gaussian_filter
def blurringGaussian(image):
        return gaussian_filter(image, 1)


# In[66]:


#test blurring
'''
fig1b, axs1b = plt.subplots(1,2, figsize=(10,20))
first_image2 = digits_vec[102]
pixels2 = first_image2.reshape((28, 28))
blurred_image2 = blurringGaussian(pixels2)
axs1b[0].imshow(pixels2, cmap='gray')
axs1b[0].set_title('Without Blurring')
axs1b[1].imshow(blurred_image2, cmap='gray')
axs1b[1].set_title('After Blurring')
'''

# In[13]:


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)

digits_vec_deskewed = deskewAll(digits_vec)


# In[67]:


#Blurring all training data and test data set 

new_blur_digits_list = []
for row in range(0, len(digits_vec_deskewed)):
    one_digit = digits_vec_deskewed[row].reshape((28, 28))
    new_blur_digits_list.append(blurringGaussian(one_digit.tolist()).flatten())

new_blur_digits = np.array(new_blur_digits_list)


# In[68]:


#Sample checkt the effect: Deskewed + Blurred
#plt.imshow(new_blur_digits[102].reshape((28, 28)), cmap='gray')
#plt.show()


# In[53]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(new_blur_digits)

# Apply transform to all
new_digit_sets = scaler.transform(new_blur_digits)


# In[72]:


from sklearn.decomposition import PCA
pca_90 = PCA(0.95)
pca_90.fit(new_digit_sets)
new_digit_sets_pca = pca_90.transform(new_digit_sets)



# In[73]:


# test_size: what proportion of original data is used for test set
#train_img, test_img, train_lbl, test_lbl = train_test_split(
#    digits_vec, digits_label, test_size=1/5.0, random_state=0)
#train_img, test_img, train_lbl, test_lbl = train_test_split(
#   new_blur_digits, digits_label, test_size=1/2.0, random_state=0)

start_index = digits_trainset[0][1]
end_index = digits_trainset[len(digits_trainset)-1][1]

train_img = new_digit_sets_pca[int(start_index)-1:int(end_index)]
train_lbl = digits_label[int(start_index)-1:int(end_index)]

start_index = digits_testset[0][1]
end_index = digits_testset[len(digits_testset)-1][1]

test_img = new_digit_sets_pca[int(start_index)-1:int(end_index)]
test_lbl = digits_label[int(start_index)-1:int(end_index)]


# In[74]:


#print(train_img.shape)
#print(test_img.shape)


# In[75]:


#Compare the performance between different kernals, and then select the best one
from sklearn.svm import SVC
#polynomial
svclassifier_poly = SVC(kernel='poly', degree=8)
svclassifier_poly.fit(train_img, train_lbl)
#Gaussian Kernel
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(train_img, train_lbl)
#sigmoid Kernel
svclassifier_sigmoid = SVC(kernel='sigmoid')
svclassifier_sigmoid.fit(train_img, train_lbl)


# In[76]:


test_pred_poly = svclassifier_poly.predict(test_img)
test_pred_rbf = svclassifier_rbf.predict(test_img)
test_pred_sigmoid = svclassifier_sigmoid.predict(test_img)


# In[77]:


print(confusion_matrix(test_lbl, test_pred_poly))
print(classification_report(test_lbl, test_pred_poly))


# In[78]:


print(confusion_matrix(test_lbl, test_pred_rbf))
print(classification_report(test_lbl, test_pred_rbf))


# In[79]:


print(confusion_matrix(test_lbl, test_pred_sigmoid))
print(classification_report(test_lbl, test_pred_sigmoid))


# Refer to the above result, the accuracy of SVM with using Gaussian Kernel is the best.
# So we start the do the parameter tunning on the SVM with Gaussian Kernal

# In[80]:


#Use GridSearchCV to do Cross Validation, select the best parameters on C, gamma for rbf
from sklearn.model_selection import GridSearchCV 

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

# fitting the model for grid search 
grid.fit(train_img, train_lbl) 


# In[62]:


# print best parameter after tuning 
print(grid.best_params_) 

# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 


# In[81]:


from datetime import datetime, time
def date_diff_in_milliseconds(dt2, dt1):
  timedelta = dt2 - dt1
  return timedelta.total_seconds() * 1000


# In[82]:


starttime = datetime.now()

test_pred_rbf_tune = grid.predict(test_img)

endtime = datetime.now()
print(date_diff_in_milliseconds(endtime, starttime))


# In[83]:


print(confusion_matrix(test_lbl, test_pred_rbf_tune))
print(classification_report(test_lbl, test_pred_rbf_tune))


# In[ ]:




