import numpy as np



class Normalization_Techniques():
    
    def __init__(self, args):
        self.args = args

    def interp(self, min_value, max_value, value):
        return (((value - min_value)/(max_value - min_value + 1e-6)))
    

    ### Normalizes by layer across all batches using the given normalization argument type
    def HSIC_Normalization(self, batchdict=None):

        batch_layer_alphas , layer_values_per_layer_unattacked, layer_values_per_layer_attacked = {} , {}, {}

        # Collect values for each layer across batches
        batch_count=0
        for batch_key, batch_value in batchdict.items():
            for layer_index, layer_value in batch_value['non_attacked'].items():
                if layer_index not in layer_values_per_layer_unattacked:
                    layer_values_per_layer_unattacked[layer_index] = []

                layer_values_per_layer_unattacked[layer_index].append(layer_value.detach().cpu())
            
            for layer_index, layer_value in batch_value['attacked'].items():
                if layer_index not in layer_values_per_layer_attacked:
                    layer_values_per_layer_attacked[layer_index] = []

                layer_values_per_layer_attacked[layer_index].append(layer_value.detach().cpu())
            
            batch_count+=1


        #*# Simplified this, removed the args.Normalizing == True code since you previously said to ignore it
        # Normalize non_attacked values for each layer across batches 
        for layer_index, layer_values in layer_values_per_layer_unattacked.items():
            layer_array = np.array(layer_values)
            if self.args.normalize == 'mean_std':
                normalized_layer_array = (layer_array - np.mean(layer_array)) / np.std(layer_array)
            elif self.args.normalize == "min_max":
                normalized_layer_array = (layer_array - np.min(layer_array)) / ( np.max(layer_array) - np.min(layer_array))
            else:
                normalized_layer_array = layer_array

            # Update the original non_attacked values with normalized values
            idx = 0
            for batch_key, batch_value in batchdict.items():
                batch_value['non_attacked'][layer_index] = normalized_layer_array[idx]
                idx += 1



        # Normalize attacked values for each layer across batches
        for layer_index, layer_values in layer_values_per_layer_attacked.items():
            layer_array = np.array(layer_values)
            if self.args.normalize == 'mean_std':
                normalized_layer_array = (layer_array - np.mean(layer_array)) / np.std(layer_array)
            elif self.args.normalize == "min_max":
                normalized_layer_array = (layer_array - np.min(layer_array)) / ( np.max(layer_array) - np.min(layer_array))
            else:
                normalized_layer_array = layer_array

            # Update original attacked values with normalized values
            idx = 0
            for batch_key, batch_value in batchdict.items():
                batch_value['attacked'][layer_index] = normalized_layer_array[idx]
                idx += 1



        ### Computing Difference of non_attacked and attacked values
        for batch_index , batch_value in batchdict.items():
            batch_layer_alphas['{}'.format(batch_index)]={}
            for layer_index in batch_value['non_attacked'].keys():
                # batch_layer_alphas['{}'.format(batch_index)]['{}'.format(layer_index)] = np.abs(batch_value['non_attacked'][layer_index] - batch_value['attacked'][layer_index])
                batch_layer_alphas['{}'.format(batch_index)]['{}'.format(layer_index)] = batch_value['non_attacked'][layer_index] - batch_value['attacked'][layer_index]
      
        return batch_layer_alphas