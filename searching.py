# Copyright 2019 ***. All Rights Reserved.
#
# Licensed under the GNU License, Version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.gnu.org/licenses/gpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, add, Flatten, DepthwiseConv2D
import pickle
import time

#MobileNetV2 Model: 18 layers with diff # of filters
org_num_layers = 18

#import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#1st layer of MobilNetV2
def layer_0(inputs, filters):
  channel_axis = -1

  x = Conv2D(filters[0], (3,3), padding='same', strides=(2,2))(inputs)
  #x = BatchNormalization(axis=channel_axis)(x)
  x = ReLU(max_value=6)(x)


  x = DepthwiseConv2D((3,3), strides=(1,1), depth_multiplier=1, use_bias=False, padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)
  x = ReLU(max_value=6)(x)

  x = Conv2D(filters[1], (1,1), padding='same', strides=(1,1))(x)
  #x = BatchNormalization(axis=channel_axis)(x)

  return x


#bottleneck layers of MobileNetV2
def bottleneck_layer(layer_num, inputs, filters):
  channel_axis = -1

  if (layer_num == 1 or layer_num == 3 or layer_num == 6 or layer_num == 13):
    dw_s = 2
  else:
    dw_s = 1
  
  x = Conv2D(filters[layer_num]*6, (1,1), padding='same', strides=(1,1))(inputs)
  #x = BatchNormalization(axis=channel_axis)(x)
  x = ReLU(max_value=6)(x)

  x = DepthwiseConv2D((3,3), strides=(dw_s, dw_s), depth_multiplier=1, use_bias=False, padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)
  x = ReLU(max_value=6)(x)

  x = Conv2D(filters[layer_num+1], (1, 1), strides=(1, 1), padding='same')(x)
  #x = BatchNormalization(axis=channel_axis)(x)

  if dw_s == 1 and layer_num != 16 and layer_num != 10:
    x = add([x, inputs])
  return x


#MonileNetV2 model implement
def MobileNetv2(new_filters, k):
  inputs = Input(shape=(32,32,3)) 
  x = layer_0(inputs, new_filters)
  #print(x.shape)
  for i in range(1, 17):
    x = bottleneck_layer(i, x, new_filters)
    #print(x.shape)
  x =  Conv2D(1280, (1,1), padding='same', strides=(1,1))(x)
  #x = BatchNormalization(axis=-1)(x)
  x = ReLU(max_value=6)(x)
  #print(x.shape)

  x = AveragePooling2D(pool_size=(1, 1), strides=None, padding='same')(x)
  x = Conv2D(k, (1, 1), padding='same', strides=(1,1))(x)

  x = Flatten()(x)
  output = Activation('softmax', name='softmax')(x)
  #print(output.shape)

  model = Model(inputs, output)
  #plot_model(model, to_file='images/MobileNetv2-test.png', show_shapes=True)

  return model


#get weights from a pre-trained model
def get_weights():
  GRAPH_PB_PATH = './data/mobilenet_v2/mobilenetv2_1.0_224_imagenet.pb' #path to your .pb file
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]

  wts_layer = [n for n in graph_nodes if (n.op=='Conv2D' or n.op=='DepthwiseConv2dNative')]
  conv2d_BN_layer = [n for n in graph_nodes if (n.op=='BiasAdd')]
  dwconv_BN_layer = [n for n in graph_nodes if (n.op=='FusedBatchNorm')]
  Reshape_layer = [n for n in graph_nodes if (n.name=='MobilenetV2/Predictions/Reshape/shape')]

  weights = []
  for n in wts_layer:
    #print("Name of the node - %s" % (n.name))
    #print("Value - ")
    wts_const = [i for i in graph_nodes if i.name==n.input[1]]
    wts_val = wts_const[0]
    #print(tensor_util.MakeNdarray(wts_val.attr['value'].tensor))
    weights.append(tensor_util.MakeNdarray(wts_val.attr['value'].tensor))

  for n in conv2d_BN_layer:
    #print("Name of the node - %s" % (n.name))
    #print("Value - ")
    wts_const = [i for i in graph_nodes if i.name==n.input[1]]
    wts_val = wts_const[0]
    #print(tensor_util.MakeNdarray(wts_val.attr['value'].tensor))
    weights.append(tensor_util.MakeNdarray(wts_val.attr['value'].tensor))

  for n in dwconv_BN_layer:
    #print("Name of the node - %s" % (n.name))
    #print("Value - ")
    wts_val = [i for i in graph_nodes if (i.name==n.input[1] or i.name==n.input[2] or i.name==n.input[3] or i.name==n.input[4])]
    for i in wts_val:
      #print(tensor_util.MakeNdarray(i.attr['value'].tensor))
      weights.append(tensor_util.MakeNdarray(i.attr['value'].tensor))

  for n in Reshape_layer:
    #print("Name of the node - %s" % (n.name))
    #print("Value - ")
    #print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    weights.append(tensor_util.MakeNdarray(n.attr['value'].tensor))
  
  return weights


#simulate latency of a trimmed model
def count_delay(num_layers, num_filters, table_result):
  #print(table_result[0][num_filters[0]-1][0],'ms')
  time_delay = table_result[0][num_filters[0]-1][0]

  for n in range (1, num_layers):
    #print(table_result[n][num_filters[n]-1][num_filters[n-1]-1],'ms') 
    time_delay = time_delay + table_result[n][num_filters[n]-1][num_filters[n-1]-1]
  
  #print(time_delay)
  
  return time_delay


#decide # of filters remains in a layer according to the simulator data
def choose_num_filters(layer_num, speedup, org_filters, new_filters, table_result):
  layer_base_time = count_delay(org_num_layers, new_filters, table_result)
  org_num = new_filters[layer_num]
  #print(layer_num)
  if layer_num == 0:
    return org_num
    
  if layer_num == 1:
    least_number = 9
  elif layer_num == 2 or layer_num == 4:
    least_number = int(org_filters[layer_num]/2-1)
  else:
    least_number = int(org_filters[layer_num]/8+3)

  for n in range(new_filters[layer_num], least_number, -1):
    #reduce filters in current layer --> reduce in_channels in next layer
    
    if layer_num == 2:
      #print("modify 2 layer due to add op")
      new_filters[layer_num] = n
      new_filters[layer_num+1] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    elif layer_num == 4:
      #print("modify 3 layer due to add op")
      new_filters[layer_num] = n
      new_filters[layer_num+1] = n
      new_filters[layer_num+2] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    elif layer_num == 7:
      #print("modify 4 layer due to add op")
      new_filters[layer_num] = n
      new_filters[layer_num+1] = n
      new_filters[layer_num+2] = n
      new_filters[layer_num+3] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    elif layer_num == 11:
      #print("modify 3 layer due to add op")
      new_filters[layer_num] = n
      new_filters[layer_num+1] = n
      new_filters[layer_num+2] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    elif layer_num == 14:
      #print("modify 3 layer due to add op")
      new_filters[layer_num] = n
      new_filters[layer_num+1] = n
      new_filters[layer_num+2] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    else:
      #print("^^^^^^^^^^^^^^^^^^")
      new_filters[layer_num] = n
      layer_new_time = count_delay(org_num_layers, new_filters, table_result)

    if layer_base_time - layer_new_time >= speedup:
      return n
  
  print("get the least num!")
  if org_num != n:
    return n

  else:
    new_filters[layer_num] = org_num
    return org_num

  
#trimming approach
def weights_prune(layer_num, old_weights, new_filters):
  print("pruning weights in %dth layer..." % (layer_num+1))
  
  #trim 1st layer:
  if layer_num == 0:
    # conv2d
    new_weights_val = np.zeros((len(old_weights[18]),len(old_weights[18][0]),
                              len(old_weights[18][0][0]),new_filters[layer_num]), dtype='float32')
    l2_old_weights = np.zeros((len(old_weights[18]),len(old_weights[18][0]),
                              len(old_weights[18][0][0]),len(old_weights[18][0][0][0])), dtype='float32')
    #max_idx = np.zeros((new_filters[layer_num]), dtype=int)
    
    for i in range(len(old_weights[18])):
      for j in range(len(old_weights[18][i])):
        for k in range(len(old_weights[18][i][j])):
                
          #choose filters with largest l2_norm remain
          l2_old_weights[i][j][k] = old_weights[18][i][j][k] / np.sqrt(np.sum(np.square(old_weights[18][i][j][k])))
          max_idx = np.argsort(-l2_old_weights[i][j][k])
          #print(max_idx)

          for n in range (new_filters[layer_num]):
            new_weights_val[i][j][k][n] = old_weights[18][i][j][k][max_idx[n]]
    old_weights[18] = new_weights_val
    print("pruned conv filters:")
    print(old_weights[18].shape)  

    #conv2d_BN
    new_weights_val = np.zeros(new_filters[layer_num], dtype='float32')

    for n in range(new_filters[layer_num]):
      new_weights_val[n] = old_weights[54][max_idx[n]]
    old_weights[54] = new_weights_val
    print("pruned conv_BN params:")
    print(old_weights[54].shape)

  #trim other layers:
  elif layer_num>0:
    #pjc_conv
    if layer_num == 1:
      prj_c = new_filters[layer_num-1]
    else:
      prj_c = new_filters[layer_num-1]*6
    #print("**CHECK CHANNEL PRJ")
    #print(prj_c)
    #print(len(old_weights[(layer_num-1)*2+19][0][0]))
    #print("CHECK CHANNEL PRJ**")
    new_weights_val = np.zeros((len(old_weights[(layer_num-1)*2+19]),len(old_weights[(layer_num-1)*2+19][0]),
                            prj_c,new_filters[layer_num]), dtype='float32')
    l2_old_weights = np.zeros((len(old_weights[(layer_num-1)*2+19]),len(old_weights[(layer_num-1)*2+19][0]),
                            len(old_weights[(layer_num-1)*2+19][0][0]),len(old_weights[(layer_num-1)*2+19][0][0][0])), dtype='float32')
    #max_idx = np.zeros((new_filters[layer_num]), dtype=int)
    
    for i in range(len(old_weights[(layer_num-1)*2+19])):
      for j in range(len(old_weights[(layer_num-1)*2+19][i])):
        for k in range(len(old_weights[(layer_num-1)*2+19][i][j])):
                
          #choose filters with largest l2_norm remain
          l2_old_weights[i][j][k] = old_weights[(layer_num-1)*2+19][i][j][k] / np.sqrt(np.sum(np.square(old_weights[(layer_num-1)*2+19][i][j][k])))

          max_idx = np.argsort(-l2_old_weights[i][j][k])
          #print(max_idx)

          for n in range (new_filters[layer_num]):
            new_weights_val[i][j][k][n] = old_weights[(layer_num-1)*2+19][i][j][k][max_idx[n]]
    old_weights[(layer_num-1)*2+19] = new_weights_val
    print("pruned prj_conv filters:")
    print(old_weights[(layer_num-1)*2+19].shape)

    #prj_BN
    new_weights_val = np.zeros(new_filters[layer_num], dtype='float32')

    for n in range(new_filters[layer_num]):
      new_weights_val[n] = old_weights[(layer_num-1)*2+55][max_idx[n]]
    old_weights[(layer_num-1)*2+55] = new_weights_val
    print("pruned prj_BN params:")
    print(old_weights[(layer_num-1)*2+55].shape)
  
    #exp_conv
    #in_channel modify:
    new_weights_val = np.zeros((len(old_weights[(layer_num-1)*2+20]),len(old_weights[(layer_num-1)*2+20][0]),
                              new_filters[layer_num], len(old_weights[(layer_num-1)*2+20][0][0][0])), dtype='float32')

    for i in range(len(old_weights[(layer_num-1)*2+20])):
      for j in range(len(old_weights[(layer_num-1)*2+20][i])):
        for n in range (new_filters[layer_num]):
          for k in range(len(old_weights[(layer_num-1)*2+20][i][j][n])):
            new_weights_val[i][j][n][k] = old_weights[(layer_num-1)*2+20][i][j][max_idx[n]][k]

    old_weights[(layer_num-1)*2+20] = new_weights_val
                  
    #filters modify:
    #print("**CHECK CHANNEL EXP")
    #print(new_filters[layer_num])
    #print(len(old_weights[(layer_num-1)*2+20][0][0]))
    #print("CHECK CHANNEL EXP**")
    if layer_num < org_num_layers-1:
      new_weights_val = np.zeros((len(old_weights[(layer_num-1)*2+20]),len(old_weights[(layer_num-1)*2+20][0]),
                              len(old_weights[(layer_num-1)*2+20][0][0]),new_filters[layer_num]*6), dtype='float32')
      l2_old_weights = np.zeros((len(old_weights[(layer_num-1)*2+20]),len(old_weights[(layer_num-1)*2+20][0]),
                              len(old_weights[(layer_num-1)*2+20][0][0]),len(old_weights[(layer_num-1)*2+20][0][0][0])), dtype='float32')
      #max_idx = np.zeros((new_filters[layer_num]*6), dtype=int)
      
      for i in range(len(old_weights[(layer_num-1)*2+20])):
        for j in range(len(old_weights[(layer_num-1)*2+20][i])):
          for k in range(len(old_weights[(layer_num-1)*2+20][i][j])):
                  
            #choose filters with largest l2_norm remain
            l2_old_weights[i][j][k] = old_weights[(layer_num-1)*2+20][i][j][k] / np.sqrt(np.sum(np.square(old_weights[(layer_num-1)*2+20][i][j][k])))

            max_idx = np.argsort(-l2_old_weights[i][j][k])
            #print(max_idx)

            for n in range (new_filters[layer_num]*6):
              new_weights_val[i][j][k][n] = old_weights[(layer_num-1)*2+20][i][j][k][max_idx[n]]

      old_weights[(layer_num-1)*2+20] = new_weights_val
      print("pruned exp_conv filters:")
      print(old_weights[(layer_num-1)*2+20].shape)

      #exp_BN
      new_weights_val = np.zeros(new_filters[layer_num]*6, dtype='float32')

      for n in range(new_filters[layer_num]*6):
        new_weights_val[n] = old_weights[(layer_num-1)*2+56][max_idx[n]]
      old_weights[(layer_num-1)*2+56] = new_weights_val
      print("pruned exp_BN params:")
      print(old_weights[(layer_num-1)*2+56].shape)

    else:
      print("last layer!")
      print(old_weights[(layer_num-1)*2+20].shape)
      print(old_weights[(layer_num-1)*2+56].shape)
      print("last layer fix expand filters")


  #depth-wise conv for layers except the last layer
  if layer_num < org_num_layers-1:
    #dw_conv
    if layer_num == 0:
      dw_c  = new_filters[layer_num]
    else:
      dw_c = new_filters[layer_num]*6
    new_weights_val = np.zeros((len(old_weights[layer_num]),len(old_weights[layer_num][0]),dw_c,1), dtype='float32') 
    rsp_old_weights = np.transpose(old_weights[layer_num], (0,1,3,2))
    l2_rsp_old_weights = np.zeros((len(old_weights[layer_num]),len(old_weights[layer_num][0]),1,len(old_weights[layer_num][0][0])), dtype='float32')
    #max_idx = np.zeros((dw_c), dtype=int)
    
    for i in range(len(old_weights[layer_num])):
      for j in range(len(old_weights[layer_num][i])):

        #choose filters with largest l2_norm remain
        l2_rsp_old_weights[i][j][0] = rsp_old_weights[i][j][0] / np.sqrt(np.sum(np.square(rsp_old_weights[i][j][0])))

        max_idx = np.argsort(-l2_rsp_old_weights[i][j][0])
        #print(max_idx)

        for n in range (dw_c):
          new_weights_val[i][j][n][0] = rsp_old_weights[i][j][0][max_idx[n]]
          #new_weights_val[i][j][n][0] = old_weights[layer_num][i][j][max_idx[n]][0]
    old_weights[layer_num] = new_weights_val
    print("pruned dw_conv filters:")
    print(old_weights[layer_num].shape) 

    #dw_BN
    print("pruned dw_BN params:")
    for m in range (4):
      new_weights_val = np.zeros(dw_c, dtype='float32')

      for n in range(dw_c):
        new_weights_val[n] = old_weights[layer_num*4+89+m][max_idx[n]]

      old_weights[layer_num*4+89+m] = new_weights_val
      print(old_weights[layer_num*4+89+m].shape)

    #nxt_pjc_in_channel
    new_weights_val = np.zeros((len(old_weights[layer_num*2+19]),len(old_weights[layer_num*2+19][0]),
                              dw_c, len(old_weights[layer_num*2+19][0][0][0])), dtype='float32')

    for i in range(len(old_weights[layer_num*2+19])):
      for j in range(len(old_weights[layer_num*2+19][i])):
        for n in range(dw_c):
          for k in range (len(old_weights[layer_num*2+19][i][j][n])):
            new_weights_val[i][j][n][k] = old_weights[layer_num*2+19][i][j][max_idx[n]][k]
          
    old_weights[layer_num*2+19] = new_weights_val

    #some layer need to be trimmd at the same time because of the Residual in MobileNetV2
    if (layer_num != 0 and layer_num != 1 and layer_num != 3 and layer_num != 6 and layer_num != 10 and layer_num != 13 and layer_num != 16):
      print("due to Add op...")
      old_weights = weights_prune(layer_num+1, old_weights, new_filters)
  
  return old_weights


def train(new_filters, retrain=True, short_term=True, weights=None):
  model = MobileNetv2(new_filters, 10)
  try:
    model = multi_gpu_model(model, gpus=8, cpu_relocation=True)
    print("Training using multiple GPUs...")
  except:
    print("Training using single GPU...")

  if short_term == False:
    print("isLong_term!!!!!!!!!")
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.train.RMSPropOptimizer(0.0001, decay=0.95), metrics=[tf.keras.metrics.categorical_accuracy])
    epochs = 500
  else:
    print("isShort_term!!!!!!!!!")
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.train.RMSPropOptimizer(0.001, decay=0.9), metrics=[tf.keras.metrics.categorical_accuracy])  
    epochs = 30

  train_weights = []

  if retrain == True:
    print("isRetrain!!!!!!!!!")
                  
    for i in range(17):
      train_weights.append(weights[18+i*2])
      train_weights.append(weights[54+i*2])
      train_weights.append(weights[0+i])
      for j in range(4):
        train_weights.append(weights[89+i*4+j])
      train_weights.append(weights[19+i*2])
      train_weights.append(weights[55+i*2])

    train_weights.append(weights[52])
    train_weights.append(weights[88])

    train_weights.append(weights[17])
    train_weights.append(weights[53])

    model.set_weights(train_weights)
    
    model.fit(x_train, y_train, epochs=epochs, verbose=2, batch_size=96)
    if short_term == True:
      accuracy = model.evaluate(x_train, y_train, verbose=0, batch_size=96)[1]
    else:
      accuracy = model.evaluate(x_test, y_test, verbose=0, batch_size=96)[1]

    if short_term == False:
      model.save_weights('./final_model_wts')

  else:
    print("isPretrain!!!!!!!!!")
    model.load_weights('./data/mobilenetv2_pretrained_cifar10')
    
    #model.fit(x_train, y_train, epochs=epochs, batch_size=96)

    #model.save('my_model.h5')

    accuracy = model.evaluate(x_test, y_test, verbose=0, batch_size=96)[1]
    
  train_weights = model.get_weights()
  #print(len(train_weights))

  new_weights = [None] * 157
  for i in range(17):
    new_weights[18+i*2] = train_weights[i*9]
    new_weights[54+i*2] = train_weights[i*9+1]
    new_weights[0+i] = train_weights[i*9+2]
    for j in range(4):
      new_weights[89+i*4+j] = train_weights[i*9+3+j]
    new_weights[19+i*2] = train_weights[i*9+7]
    new_weights[55+i*2] = train_weights[i*9+8]

  new_weights[52] = train_weights[153]
  new_weights[88] = train_weights[154]

  new_weights[17] = train_weights[155]
  new_weights[53] = train_weights[156]
  #print(len(new_weights))
  '''for i in range(len(new_weights)):
    if new_weights[i].shape != train_weights[i].shape:
      print("WRONG!!!!!!!!!!!!!!!")'''

  del train_weights[:]

  tf.keras.backend.clear_session()
  
  return [accuracy, new_weights]


if __name__ == '__main__': 
  start = time.perf_counter()

  org_filters = [32,16,24,24,32,32,32,64,64,64,64,96,96,96,160,160,160,320]
  
  #1.5->1.3-->18; 2.0->1.5-->24; 2.5->2.0-->30
  tgt_speedup_ratio = 2
  num_iters = 24

  f = open('./simulator/sim_data/CPU_sim_latency.pickle', 'rb')
  table_result = pickle.load(f)
  f.close()

  print("base_time:")
  base_time = count_delay(org_num_layers, org_filters, table_result)
  print(base_time)

  speedup_per_iter = (base_time - base_time/tgt_speedup_ratio) / num_iters
  #set a decay for peedup_per_iter
  print("speedup_per_iter: %lf * 0.96^n" % (speedup_per_iter))
  #print(speedup_per_iter)

  new_filters = org_filters

  #pre-train
  pretrain_rst = train(retrain=False, short_term=False, new_filters=new_filters)

  old_weights = pretrain_rst[1]

  print("******Pre-trained Accuracy = %f" % (pretrain_rst[0]))

  start_main = time.perf_counter()
  for n in range(num_iters):
    start_n = time.perf_counter()
  
    tmp_filters = []
    tmp_weights = []
    speedup_per_iter = speedup_per_iter*0.96
    
    print("\n\nnum_iters:")
    print(n+1)
    print("old_filters:")
    print(new_filters)
    
    accuracy = []
    
    i = 0
    while i<org_num_layers:
      start_layer = time.perf_counter()

      tmp_w = [None] * len(old_weights)
      for j in range(len(old_weights)):
        tmp_w[j] = old_weights[j].copy()

      print("\nprune %d th layer:" % (i+1))
      tmp_n = new_filters[i]
      #tmp_n1 = new_filters[i+1]      

      choose_num_filters(i, speedup_per_iter, org_filters, new_filters, table_result)

      print("candidate_filters:")
      print(new_filters)
      appended_filters = list(new_filters)
      tmp_filters.append(appended_filters)
      #print(tmp_filters)
   
      if new_filters[i] != tmp_n:
        
        new_weights = weights_prune(i, tmp_w, new_filters)
        print("short-term retrain...")
        retrain_result = train(new_filters=new_filters, weights=new_weights)
        accuracy.append(retrain_result[0])
        print("accu_list:")
        print(accuracy)
        tmp_weights.append(retrain_result[1])

        del new_weights[:]

      else:
        print("donot prune this layer")
        accuracy.append(0)
        print(accuracy)
        tmp_weights.append(old_weights)
      
      new_filters[i] = tmp_n
      
      #some layer need to be trimmd at the same time because of the Residual in MobileNetV2
      if i == 2:
        print("modify 2 layer due to Residual op")
        new_filters[i+1] = tmp_n
        i = i+2

      elif i ==4:
        print("modify 3 layer due to Residual op")
        new_filters[i+1] = tmp_n
        new_filters[i+2] = tmp_n
        i = i+3

      elif i ==7:
        print("modify 4 layer due to Residual op")
        new_filters[i+1] = tmp_n
        new_filters[i+2] = tmp_n
        new_filters[i+3] = tmp_n
        i = i+4

      elif i ==11:
        print("modify 3 layer due to Residual op")
        new_filters[i+1] = tmp_n
        new_filters[i+2] = tmp_n
        i = i+3

      elif i ==14:
        print("modify 3 layer due to Residual op")
        new_filters[i+1] = tmp_n
        new_filters[i+2] = tmp_n
        i = i+3

      else:
        i = i+1

      del tmp_w[:]

      print("******%d layer delay: %f s" % (i, time.perf_counter()-start_layer))
    
    h_idx = accuracy.index(max(accuracy))
    print("candidate_filters_list:")
    print(tmp_filters)
    print("highest accuracy: %f" % (max(accuracy)))
    new_filters = tmp_filters[h_idx]
    print("new_filters:")
    print(new_filters)
    old_weights = tmp_weights[h_idx]

    del tmp_filters[:]
    del tmp_weights[:]
    del accuracy[:]

    print("******%d iteration delay: %f s" % (n+1, time.perf_counter()-start_n))
    
  print("\n\nfinal_filters:")
  print(new_filters)
  optimized_time = count_delay(org_num_layers, new_filters, table_result)
  print("long-term retrain...")
  final_result = train(short_term=False, new_filters=new_filters, weights=old_weights)
  final_accuracy = final_result[0]

  print("\nbase_time = %fms,\noptimized_time = %fms,\naccelerated_time = %fms,\nfinal speedup ratio = %f"
        % (base_time, optimized_time, base_time-optimized_time, base_time / optimized_time))

  print("\nFinal_Accuracy = %f" % (final_accuracy))

  print("******program  delay: %f mins" % ((time.perf_counter()-start)/60))
  print("******main loop  delay: %f mins" % ((time.perf_counter()-start_main)/60))
