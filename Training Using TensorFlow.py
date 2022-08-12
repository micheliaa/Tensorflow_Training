#!/usr/bin/env python
# coding: utf-8

# In[34]:


import tensorflow as tf


# In[35]:


print(tf.__version__)


# In[36]:


const_a = tf.constant([[1, 2, 3, 4]], shape = [2,2], dtype=tf.float32)
print(const_a)
print("Value of the constant const_a:", tf.get_static_value(const_a))
print("Data type of the constant const_a:", const_a.dtype)
print("Shape of the constant const_a:", const_a.shape)
print("Name of the device that is to generate the constant const_a:", tf.device(const_a))


# In[37]:


zeros_b = tf.zeros(shape=[2,3], dtype=tf.int32)
print(zeros_b)


# In[38]:


zeros_a = tf.zeros_like(const_a)
print(zeros_a)


# In[39]:


fill_d = tf.fill([3,3], 8)
print(fill_d)


# In[40]:


random_e = tf.random.normal([5,6], mean=0, stddev=1.0, seed=1)
print(random_e)


# In[41]:


list_f = [1,2,3,4,5,6]
print(type(list_f))


# In[42]:


tensor_f = tf.convert_to_tensor(list_f, dtype=tf.float32)
print(tensor_f)


# In[43]:


var_1 = tf.Variable(tf.ones([2,3]))
print(var_1)


# In[44]:


print("Value Of the variable = ", var_1)


# In[45]:


var_2 = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print("Value of the variable var_1 after the assignment:" ,var_2)


# In[46]:


var_3 = tf.Variable([[2., 3., 4.], [5., 6., 7.]])
print(var_3)


# In[47]:


tensor_h = tf.random.normal([4, 100, 100, 3], dtype=tf.float32)
print(tensor_h)


# In[48]:


tensor_h[0,:,1:]


# In[49]:


tensor_h[2::]


# In[50]:


tensor_h[::-1]


# In[51]:


tensor_h[0][10][39][1]


# In[52]:


indices = [0,1,3]
tf.gather(tensor_h, axis=0, indices=indices,  batch_dims=1)


# In[53]:


indices = [0,1,1,0], [1,2,2,0]
tf.gather_nd(tensor_h, indices=indices)


# In[54]:


const_d_1 = tf.constant([[1,2,3,4]],shape=[2,2], dtype=tf.int32)
print(const_d_1.shape)
print(const_d_1.get_shape())
print(tf.shape(const_d_1))


# In[55]:


reshape_1 = tf.constant([[1,2,3,],
                        [4,5,6]])
print(reshape_1)
tf.reshape(reshape_1, (3,2))


# In[56]:


expand_sample_1 = tf.random.normal([100,100,3], seed=1)
print("size of the original data:", expand_sample_1.shape)
print('add dimension before the first dimention (axis=0): ', tf.expand_dims(expand_sample_1, axis=0).shape)
print("add a dimension before the second dimention (axis=1): ", tf.expand_dims(expand_sample_1, axis=1).shape)


# In[57]:


squeeze_sample_1 = tf.random.normal([1,100,100,3])
print("size of the original data:", squeeze_sample_1.shape)
squeezed_sample_1 = tf.squeeze(squeeze_sample_1)
print("data size after dimension squeezing: ", squeezed_sample_1.shape)


# In[58]:


trans_1 = tf.constant([1,2,3,4,5,6], shape=[2,3])
print("size of the original data:", trans_1.shape)
transposed_1 = tf.transpose(trans_1)
print("size of transposed data:", transposed_1.shape)


# In[59]:


trans_2 = tf.random.normal([4,100,100,3])
print("size of the original data:", trans_2.shape)
transposed_2 = tf.transpose(trans_2)
print("size of transposed data:", trans_2.shape)


# In[60]:


broadcast_sample_1 = tf.constant([1,2,3,4,5,6])
print("original data:", broadcast_sample_1.numpy())
broadcasted_sample = tf.broadcast_to(broadcast_sample_1, shape=[4,6])
print("broadcasted data:", broadcasted_sample.numpy())


# In[61]:


a = tf.constant([[0,0,0],
                [10,10,10],
                [20,20,20],
                [30,30,30]])
b = tf.constant([1,2,3])
print(a+b)


# In[62]:


a = tf.constant([[3,5],
               [4,8]])
b = tf.constant([[1,6],
                [2,9]])
print(tf.add(a,b))


# In[63]:


tf.matmul(a,b)


# In[64]:


argmax_sample_1 = tf.constant([[1,3,2],
                              [2,5,8],
                              [7,5,9]])
print("input tensor:", argmax_sample_1.numpy())
max_sample_1 = tf.argmax(argmax_sample_1, axis=0)
max_sample_2 = tf.argmax(argmax_sample_1, axis=1)
print("Searches for the position of the maximum value by column:", max_sample_1.numpy())
print("Searches for the position of the maximum value by row:", max_sample_2.numpy())


# In[68]:


reduce_sample_1 = tf.constant([1,2,3,4,5,6], shape=[2,3])
print("raw data:", reduce_sample_1.numpy())
print("calculate the sum (axis=None):",
      tf.reduce_sum(reduce_sample_1,axis=None).numpy())
print("calculate the sum of each column by column (axis=0):", 
      tf.reduce_sum(reduce_sample_1, 0).numpy())
print("calculate the sum of each column by row (axis=1):", 
      tf.reduce_sum(reduce_sample_1, 1).numpy())


# In[69]:


import numpy as np

split_sample1 = tf.random.normal([10,100,100,3])
print("size of the original data:",
     split_sample1.shape)

splited_sample1 = tf.split(split_sample1, num_or_size_splits=5, axis=0)
print("size of the split data when num_or_size_split is set to 10:",
     np.shape(splited_sample1))

splited_sample2 = tf.split(split_sample1, num_or_size_splits=[3,5,2], axis=0)
print("when num_or_size_splits=[3,5,2], the size of the splits data is:",
     np.shape(splited_sample2[0]), np.shape(splited_sample2[1]),
     np.shape(splited_sample2[2]))


# In[70]:


concat_sample_1 = tf.random.normal([4,100,100,3])
concat_sample_2= tf.random.normal([40,100,100,3])
print("Sizes the original data:", concat_sample_1.shape, concat_sample_2.shape)
concated_sample_1 = tf.concat([concat_sample_1,concat_sample_2], 0)
print("Sizes of the data after concatenation:", concated_sample_1.shape)


# In[71]:


stack_sample_1 = tf.random.normal([100,100,3])
stack_sample_2 = tf.random.normal([100,100,3])
print("Sizes of the original data:", stack_sample_1.shape, stack_sample_2.shape)
stacked_sample = tf.stack([stack_sample_1, stack_sample_2], 0)
print("Data size after concatenation:", stacked_sample.shape)


# In[72]:


tf.unstack(stacked_sample, axis=0)


# In[73]:


sort_sample1 = tf.random.shuffle(tf.range(10))
print("input tensor:", sort_sample1.numpy())
sorted_sample1 = tf.sort(sort_sample1, direction="ASCENDING")
sorted_sample2 = tf.argsort(sort_sample1, direction="ASCENDING")
print("tensor after sorting:", sorted_sample1.numpy())
print("The indexes of the element after sorting are as follows:", sorted_sample2.numpy())


# In[74]:


value, index = tf.nn.top_k(sort_sample1, 5)
print("input tensor:", sort_sample1.numpy())
print("The first value in ascending order are as follows:", value.numpy())
print("The first value in ascending order in ascending order:", index.numpy())


# In[75]:


x = tf.ones((2, 2), dtype=tf.dtypes.float32) 
y = tf.constant([[1, 2], [3, 4]], dtype=tf.dtypes.float32) 
z = tf.matmul(x, y) 
print(z)


# In[1]:


import tensorflow.compat.v1 as tf


# In[77]:


tf.disable_eager_execution()


# In[2]:


a = tf.ones((2, 2), dtype=tf.dtypes.float32) 
b = tf.constant([[1, 2], [3, 4]], dtype=tf.dtypes.float32) 
c = tf.matmul(a, b)
sess = tf.Session()
print(sess.run(c))


# In[1]:


import tensorflow as tf


# In[2]:


import numpy


# In[3]:


thre_1 = tf.random.uniform([], 0, 1) 
x = tf.reshape(tf.range(0, 4), [2, 2]) 
print(thre_1) 
if thre_1.numpy() > 0.5: 
    y = tf.matmul(x, x) 
else: 
    y = tf.add(x, x)


# In[4]:


@tf.function
def simple_nn_layer(w,x,b): 
    return tf.nn.relu(tf.matmul(w, x)+b) 

w = tf.random.uniform((3, 3)) 
x = tf.random.uniform((3, 3)) 
b = tf.constant(0.5, dtype='float32') 
simple_nn_layer(w,x,b)


# In[5]:


CNN_layer = tf.keras.layers.Conv2D(100, 2, strides=(1,1)) 


# In[6]:


@tf.function
def CNN_fn(image):
  return CNN_layer(image)


# In[7]:


image = tf.zeros([100, 200, 200, 3]) 


# In[8]:


CNN_layer(image)
CNN_fn(image)


# In[9]:


import timeit


# In[ ]:


print("time required for performing the computation of one convolutional neural network (CNN) layer in eager execution mode:", timeit.timeit(lambda: CNN_layer(image), number=10)) 
print("time required for performing the computation of one CNN layer in graph mode:", timeit.timeit(lambda: CNN_fn(image), number=10))


# In[ ]:




