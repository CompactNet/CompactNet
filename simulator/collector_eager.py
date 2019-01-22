import tensorflow as tf
import numpy as np
import time
import pickle

def measure_time(layer_num, num_filters, num_input_channels):
  tf.enable_eager_execution()
  delay_list = [None]*10

  if layer_num == 0:  
    #x = tf.random.uniform([1,224,224,num_input_channels], 0, 225)
    x = np.random.uniform(0,255,(1, 224, 224, num_input_channels))
    x = x.astype('float32')
    x = tf.constant(x)

    #expand_filter_val = tf.random.uniform([3,3,num_input_channels,num_filters], 1, 8)
    expand_filter_val = np.random.uniform(1, 8,(3, 3, num_input_channels, num_filters))
    expand_filter_val = expand_filter_val.astype('float32')
    expand_filter_val = tf.constant(expand_filter_val)

    #para_val = tf.random.uniform([num_filters], 10, 20)
    para_val = np.random.uniform(10, 20, (num_filters))
    para_val = para_val.astype('float32')
    para_val = tf.constant(para_val)

    #dw_filter_val = tf.random.uniform([3,3,num_filters,1], 1, 8)
    dw_filter_val = np.random.uniform(1, 8,(3, 3, num_filters, 1))
    dw_filter_val = dw_filter_val.astype('float32')
    dw_filter_val = tf.constant(dw_filter_val)
    
    with tf.device("/device:GPU:0"):
      for i in range(10):          
        start = time.perf_counter()

        expand_conv = tf.nn.conv2d(input=x, filter=expand_filter_val,
                      strides=[1,2,2,1], padding="SAME", use_cudnn_on_gpu=True)
        expand_BN = tf.nn.bias_add(expand_conv, para_val)
        expand_relu = tf.nn.relu6(expand_BN)
        dw_conv = tf.nn.depthwise_conv2d_native(expand_relu, dw_filter_val,
                  [1,1,1,1], "SAME")
        dw_BN = tf.nn.fused_batch_norm(dw_conv, para_val, para_val,
                para_val,para_val, is_training=False)
        dw_relu = tf.nn.relu6(dw_BN[0])
        #print(dw_relu)

        #sess = tf.Session()
        #start = time.perf_counter()
        #sess.run(dw_relu)
        delay_list[i] = time.perf_counter()-start
        #sess.close()
        #tf.reset_default_graph()

      #print(delay_list)
      time_delay = min(delay_list) * 1000

  elif layer_num == 17:
    #x = tf.random.uniform([1,7,7,num_input_channels], 0, 225)
    x = np.random.uniform(0,255,(1, 7, 7, num_input_channels))
    x = x.astype('float32')
    x = tf.constant(x) 

    #project_filter_val = tf.random.uniform([1,1,num_input_channels,num_filters], 1, 8)
    project_filter_val = np.random.uniform(1, 8,(1, 1, num_input_channels, num_filters))
    project_filter_val = project_filter_val.astype('float32')
    project_filter_val = tf.constant(project_filter_val)

    #para_val = tf.random.uniform([num_filters], 10, 20)
    para_val = np.random.uniform(10, 20, (num_filters))
    para_val = para_val.astype('float32')
    para_val = tf.constant(para_val)

    #expand_filter_val = tf.random.uniform([1,1,num_filters,1280], 1, 8)
    expand_filter_val = np.random.uniform(1, 8,(1, 1, num_filters, 1280))
    expand_filter_val = expand_filter_val.astype('float32')
    expand_filter_val = tf.constant(expand_filter_val)

    #expand_para_val = tf.random.uniform([1280], 10, 20)
    expand_para_val = np.random.uniform(10, 20, (1280))
    expand_para_val = expand_para_val.astype('float32')
    expand_para_val = tf.constant(expand_para_val)
    
    with tf.device("/device:GPU:0"):     
      for i in range(10):
        start = time.perf_counter()

        project_conv = tf.nn.conv2d(input=x, filter=project_filter_val,
                      strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=True)
        project_BN = tf.nn.bias_add(project_conv, para_val)
        
        expand_conv = tf.nn.conv2d(input=project_BN, filter=expand_filter_val,
                      strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=True)
        expand_BN = tf.nn.bias_add(expand_conv, expand_para_val)
        expand_relu = tf.nn.relu6(expand_BN)

        delay_list[i] = time.perf_counter()-start

      #for i in range(10): 
        #start = time.perf_counter()
        #sess.run(expand_relu)
        #delay_list[i] = time.perf_counter()-start

      time_delay = min(delay_list) * 1000

  else:
    if layer_num == 1: 
      input_dim = 112
    elif (layer_num > 1 and layer_num <= 3): 
      input_dim = 56
    elif (layer_num > 3 and layer_num <= 6): 
      input_dim = 28
    elif (layer_num > 6 and layer_num <= 13): 
      input_dim = 14
    else: 
      input_dim = 7

    if (layer_num == 1 or layer_num == 3 or layer_num == 6 or layer_num == 13):
      dw_strides_val = [1,2,2,1]
    else:
      dw_strides_val = [1,1,1,1]

    #x = tf.random.uniform([1,input_dim,input_dim,num_input_channels], 0, 225)
    x = np.random.uniform(0, 255,(1, input_dim, input_dim, num_input_channels))
    x = x.astype('float32')
    x = tf.constant(x)

    #project_filter_val = tf.random.uniform([1,1,num_input_channels,num_filters], 1, 8)
    project_filter_val = np.random.uniform(1, 8,(1, 1, num_input_channels, num_filters))
    project_filter_val = project_filter_val.astype('float32')
    project_filter_val = tf.constant(project_filter_val)

    #para_val = tf.random.uniform([num_filters], 10, 20)
    para_val = np.random.uniform(10, 20, (num_filters))
    para_val = para_val.astype('float32')
    para_val = tf.constant(para_val)

    #expand_filter_val = tf.random.uniform([1,1,num_filters,num_filters*6], 1, 8)
    expand_filter_val = np.random.uniform(1, 8,(1, 1, num_filters, num_filters*6))
    expand_filter_val = expand_filter_val.astype('float32')
    expand_filter_val = tf.constant(expand_filter_val)

    #para_val_b = tf.random.uniform([num_filters*6], 10, 20)
    para_val_b = np.random.uniform(10, 20, (num_filters*6))
    para_val_b = para_val_b.astype('float32')
    para_val_b = tf.constant(para_val_b)

    #dw_filter_val = tf.random.uniform([3,3,num_filters*6,1], 1, 8)
    dw_filter_val = np.random.uniform(1, 8,(3, 3, num_filters*6, 1))
    dw_filter_val = dw_filter_val.astype('float32')
    dw_filter_val = tf.constant(dw_filter_val)
    
    with tf.device("/device:GPU:0"):      
      for i in range(10):
        start = time.perf_counter()

        project_conv = tf.nn.conv2d(input=x, filter=project_filter_val,
                      strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=True)
        project_BN = tf.nn.bias_add(project_conv, para_val)
        
        if(layer_num != 1 and layer_num != 2 and layer_num != 4 and layer_num != 7 and layer_num != 11 and layer_num != 14):
          Add = tf.math.add(project_BN, project_BN)
          expand_conv = tf.nn.conv2d(input=Add, filter=expand_filter_val,
                        strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=True)
        else:
          expand_conv = tf.nn.conv2d(input=project_BN, filter=expand_filter_val,
                        strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=True)
                        
        expand_BN = tf.nn.bias_add(expand_conv, para_val_b)
        expand_relu = tf.nn.relu6(expand_BN)

        dw_conv = tf.nn.depthwise_conv2d_native(expand_relu, dw_filter_val,
                  dw_strides_val, "SAME")
        dw_BN = tf.nn.fused_batch_norm(expand_relu, para_val_b, para_val_b,
                para_val_b, para_val_b, is_training=False)
        dw_relu = tf.nn.relu6(dw_BN[0])

        delay_list[i] = time.perf_counter()-start

      #for i in range(10):  
        #start = time.perf_counter()
        #sess.run(dw_relu)
        #delay_list[i] = time.perf_counter()-start

      time_delay = min(delay_list) * 1000
     
  #sess.close()
  #tf.reset_default_graph()

  return time_delay


def make_table(num_layers, max_num_filters):

  lookup_table = []

  for n in range(num_layers):
    print("\n***********No.%d Layer: %d filters" % (n+1,max_num_filters[n]))

    if n == 0:  
      num_input_channels = 3
      
      print("input channel = 3\n")

      table_val = np.zeros((max_num_filters[n], 1))

      for i in range(1, max_num_filters[n]+1, 1):
        table_val[i-1][0] = measure_time(n, i, 3)
        print("***********%d filters with 3 input_channels-->time_delay: %fms" % (i,table_val[i-1][0]))
        #print(table_val[i-1][0])

      #lookup_table.append(table_val)
      #print(lookup_table)      

    else:
      if n == 1:
        num_input_channels = max_num_filters[n-1]

        print("input channel = %d\n" % (num_input_channels))

        table_val = np.zeros((max_num_filters[n], num_input_channels))

        for i in range(1, max_num_filters[n]+1):
          
          for k in range(1, num_input_channels+1):
            table_val[i-1][k-1] = measure_time(n, i, k)
            print("***********%d filters with %d input_channels-->time_delay: %fms" % (i,k,table_val[i-1][k-1]))
            #print(table_val[i-1][k-1])

      else:
        num_input_channels = max_num_filters[n-1]*6
   
        print("input channel = %d\n" % (int(num_input_channels/6)))

        table_val = np.zeros((max_num_filters[n], int(num_input_channels/6)))
      
        for i in range(1, max_num_filters[n]+1):

          for k in range(6, num_input_channels+6, 6):
            table_val[i-1][int((k-6)/6)] = measure_time(n, i, k)
            print("***********%d filters with %d input_channels-->time_delay: %fms" % (i,int(k/6),table_val[i-1][int((k-6)/6)]))
            #print(table_val[i-1][k-1])

    lookup_table.append(table_val)
    #print(lookup_table)
  
  return lookup_table


if __name__ == '__main__':
  num_layers = 18
  max_num_filters = [32,16,24,24,32,32,32,64,64,64,64,96,96,96,160,160,160,320]
  
  table_1 = make_table(num_layers, max_num_filters)
  table_2 = make_table(num_layers, max_num_filters)

  #f = open('./table_2.pickle', 'wb')
  #pickle.dump(table_1, f)
  #f.close()
  
  np.set_printoptions(threshold=np.nan)
  np_lookup_table = np.array(table_1)
  #np.savetxt("CPU_224:table_1.txt", np_lookup_table, fmt='%s', delimiter=',')

  np.set_printoptions(threshold=np.nan)
  np_lookup_table = np.array(table_2)
  #np.savetxt("CPU_224:table_2.txt", np_lookup_table, fmt='%s', delimiter=',')
  
  #f1 = open('./table_1.pickle', 'rb')
  #f2 = open('./table_2.pickle', 'rb')
  #table_1 = pickle.load(f1)
  #table_2 = pickle.load(f2)
  #f1.close()
  #f2.close()

  #print(len(table_1))

  for n in range(len(table_1)):
    #print(len(table_1[n]))
    
    for i in range(len(table_1[n])):
      #print(len(table_1[n][i]))
      
      for k in range(len(table_1[n][i])):
        print("*******%f  %f -->" % (table_1[n][i][k], table_2[n][i][k]))
        if(table_1[n][i][k] > table_2[n][i][k]):
          table_1[n][i][k] = table_2[n][i][k]
        print("-->%f" % (table_1[n][i][k]))

  f = open('./sim_data/CPU_sim_latency.pickle', 'wb')
  pickle.dump(table_1, f)
  f.close() 

  np.set_printoptions(threshold=np.nan)
  np_table_result = np.array(table_1)
  np.savetxt("./sim_data/CPU_sim_latency.txt", np_table_result, fmt='%s', delimiter=',')
