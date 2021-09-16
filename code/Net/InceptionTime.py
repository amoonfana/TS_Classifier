import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def calculate_mask_index(kernel_length_now,largest_kernel_length):
    right_zero_mast_length = math.ceil((largest_kernel_length-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_length - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_length):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_length)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_length))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask

def creak_layer_mask(layer_parameter_list):
    largest_kernel_length = 0
    for layer_parameter in layer_parameter_list:
        if layer_parameter[-1]>largest_kernel_length:
            largest_kernel_length = layer_parameter[-1]
        
        
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_length)
        big_weight = np.zeros((i[1],i[0],largest_kernel_length))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_length)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class Parallel_Inception_Layer(nn.Module):
    # this build inception layer that can be calculated parrally.
    # the method it uses it using zero mask to on a big convoluation 
    # for example: kernel sizes 3 5 7 will be like 
    #  0     0    value value value  0     0
    #  0    value value value value value  0  
    # value value value value value value value  
    
    def __init__(self,layer_parameters, use_bias = True, use_batch_Norm =True, use_relu =True):
        super(Parallel_Inception_Layer, self).__init__()
        
        self.use_bias = use_bias
        self.use_batch_Norm = use_batch_Norm
        self.use_relu = use_relu
        

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=max_kernel_size, 
                                      bias =self.use_bias)
        
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        
        if self.use_batch_Norm:
            result_3 = self.bn(result_2)
        else:
            result_3 = result_2
            
        if self.use_relu:
            result = F.relu(result_3)
            return result
        else:
            return result_3

class Inception_module(nn.Module):
    def __init__(self, input_channle_size, nb_filters, bottleneck_size, kernel_sizes, stride = 1, activation = 'linear'):
        super(Inception_module, self).__init__()
        self.input_channle_size = input_channle_size
        self.nb_filters = nb_filters
        self.bottleneck_size = bottleneck_size
        self.kernel_sizes = kernel_sizes-1
        self.stride = stride
        self.activation = activation 
        
        self.n_incepiton_scale = 3
        self.kernel_size_s = [self.kernel_sizes // (2 ** i) for i in range(self.n_incepiton_scale)]
        
        if self.input_channle_size > 1 and self.bottleneck_size!= None:
            self.bottleneck_layer = SampaddingConv1D(self.input_channle_size, self.bottleneck_size,kernel_size = 1, use_bias = False)
            self.layer_parameter_list = [ (self.bottleneck_size,self.nb_filters ,kernel_size) for kernel_size in self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list,use_bias = False, use_batch_Norm = False, use_relu =False)                
        else:
            self.layer_parameter_list = [ (self.input_channle_size,self.nb_filters ,kernel_size) for kernel_size in self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list,use_bias = False, use_batch_Norm = False, use_relu =False)
        
            
        self.maxpooling_layer = SampaddingMaxPool1D(3,self.stride)
        self.conv_6_layer = SampaddingConv1D(self.input_channle_size,self.nb_filters, kernel_size = 1, use_bias = False)
        
        self.output_channel_numebr = self.nb_filters*(self.n_incepiton_scale+1)
        self.bn_layer = nn.BatchNorm1d(num_features=self.output_channel_numebr)
        
        
    def forward(self,X):
        if X.shape[-2] >1:
            input_inception = self.bottleneck_layer(X)
        else: 
            input_inception = X
        concatenateed_conv_list_result  = self.parallel_inception_layer(input_inception)
        conv_6 = self.conv_6_layer(self.maxpooling_layer(X))
        
        
        concatenateed_conv_list_result_2 = torch.cat((concatenateed_conv_list_result,conv_6),1)
        result = F.relu(self.bn_layer(concatenateed_conv_list_result_2))
        return result
        
class ShortcutLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, use_bias = True):
        super(ShortcutLayer, self).__init__()
        self.use_bias = use_bias
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, bias = self.use_bias)
        self.bn = nn.BatchNorm1d(num_features = out_channels)
    def forward(self, X):
        X = self.padding(X)
        X = F.relu(self.bn(self.conv1d(X)))
        return X    
        
class SampaddingConv1D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size, use_bias = True):
        super(SampaddingConv1D, self).__init__()
        self.use_bias = use_bias
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, bias = self.use_bias)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        return X

class SampaddingMaxPool1D(nn.Module):
    def __init__(self,pooling_size, stride):
        super(SampaddingMaxPool1D, self).__init__()
        self.pooling_size = pooling_size
        self.stride = stride
        self.padding = nn.ConstantPad1d((int((pooling_size-1)/2), int(pooling_size/2)), 0)
        self.maxpool1d = nn.MaxPool1d(self.pooling_size, stride=self.stride)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.maxpool1d(X)
        return X
    
class InceptionTime(nn.Module):
    def __init__(self, 
                 input_channle_size, 
                 nb_classes, 
                 verbose=False, 
                 build=True, 
                 nb_filters=32, 
                 use_residual=True, 
                 use_bottleneck=True, 
                 depth=6, 
                 kernel_size=41):
        super(InceptionTime, self).__init__()
        
        self.input_channle_size = input_channle_size
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        if use_bottleneck:
            self.bottleneck_size = 32
        else:
            self.bottleneck_size = None
            
    
        self.res_layer_list = nn.ModuleList()
        self.layer_list = nn.ModuleList()
        self.out_put_channle_number_list = []
        
        for d in range(self.depth):
            if d == 0:
                input_channle_size_for_this_layer = self.input_channle_size
            else:
                input_channle_size_for_this_layer = self.out_put_channle_number_list[-1]
            inceptiontime_layer = Inception_module(input_channle_size_for_this_layer, 
                             self.nb_filters, 
                             self.bottleneck_size, 
                             self.kernel_size,
                             stride = 1, 
                             activation = 'linear')
            self.layer_list.append(inceptiontime_layer)
            self.out_put_channle_number_list.append(inceptiontime_layer.output_channel_numebr)

            if self.use_residual and d % 3 == 2:
                if d ==2:
                    shortcutlayer = ShortcutLayer(self.input_channle_size, self.out_put_channle_number_list[-1], kernel_size = 1, use_bias = False)
                else:   
                    shortcutlayer = ShortcutLayer(self.out_put_channle_number_list[-4], self.out_put_channle_number_list[-1], kernel_size = 1, use_bias = False)
                self.res_layer_list.append(shortcutlayer)
        
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(self.out_put_channle_number_list[-1], self.nb_classes)
    
    def forward(self, X):
        res_layer_index = 0
        input_res = X
        for d in range(self.depth):
            X = self.layer_list[d](X)
            if self.use_residual and d % 3 == 2:
                shot_cut = self.res_layer_list[res_layer_index](input_res)
                res_layer_index = res_layer_index + 1
                X = torch.add(shot_cut,X)
                input_res = X
                
        X = self.averagepool(X)
        X = X.squeeze_(-1)
        X = self.hidden(X)
        return X
