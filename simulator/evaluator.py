import pickle

def count_delay(num_layers, num_filters, table_result):
  print(table_result[0][num_filters[0]-1][0],'ms')
  time_delay = table_result[0][num_filters[0]-1][0]

  #accumulate latency of each layer with diff # in/out channels
  for n in range (1, num_layers):
    print(table_result[n][num_filters[n]-1][num_filters[n-1]-1],'ms') 
    time_delay = time_delay + table_result[n][num_filters[n]-1][num_filters[n-1]-1]
  
  print(time_delay)
  
  return time_delay
  

if __name__ == '__main__':
  #MobileNetV2 Model: 18 layers with diff # of filters
  org_num_layers = 18
  org_filters = [32,16,24,24,32,32,32,64,64,64,64,96,96,96,160,160,160,320]
  
 # new_num_layers = 4
  #new_filters = [32, 16, 24, 24, 32,32,32,64,64,64,64,96,96,96,160,160,160,320] 
  new_filters = [32, 12, 20, 20, 24, 24, 24, 12, 12, 12, 12, 22, 22, 22, 24, 24, 24, 44]
  #new_filters = [32, 10, 24, 24, 32, 32, 32, 22, 22, 22, 22, 16, 16, 16, 42, 42, 42, 44]
 
  #open collected data
  f = open('./CPU_224_table_result.pickle', 'rb')
  table_result = pickle.load(f)
  f.close()

  print("\nbase_time_per_layer:")
  base_time = count_delay(org_num_layers, org_filters, table_result)
 
  print("\noptimized_time_per_layer:")
  for new_num_layers in range (1, 19, 1):
    print(new_num_layers)
    #simulate delay of a new trimmed model
    optimized_time = count_delay(new_num_layers, new_filters, table_result)

  accelerated_time = base_time - optimized_time
  accelerated_percent = accelerated_time / base_time

  print("\nbase_time = %fms,\noptimized_time = %fms,\naccelerated_time = %fms,\n%f percent speed up"
        % (base_time, optimized_time, accelerated_time, accelerated_percent*100))
