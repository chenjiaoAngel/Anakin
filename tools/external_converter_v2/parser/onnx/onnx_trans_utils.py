import onnx
import numpy as np
#from tensorflow.core.framework import types_pb2, tensor_pb2
from google.protobuf import text_format

ONNX_TO_ANAKIN_DTYPE = {
    onnx.AttributeProto.FLOAT: np.float32,
    onnx.AttributeProto.INT: np.int32,
    # onnx.AttributeProto.HALF: np.float16,
    # onnx.AttributeProto.DOUBLE: np.float64,
    # onnx.AttributeProto.INT32: np.int32,
    # onnx.AttributeProto.INT16: np.int16,
    # onnx.AttributeProto.INT8: np.int8,
    # onnx.AttributeProto.UINT8: np.uint8,
    # onnx.AttributeProto.UINT16: np.uint16,
    # onnx.AttributeProto.INT64: np.int64,
     #types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    # types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    # types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    # nnx.AttributeProto.BOOL: np.bool,
}
ONNX_TO_ANAKIN_DTYPE1 = {
    1: np.float32,
    6: np.int32,
    7: np.int64,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}

ANAKIN_VALID_ATTRIBUTES = {
    'p', 'bias', 'axes', 'pads', 'mean', 'activation_beta', 'spatial_scale', 'broadcast', 'pooled_shape', 'high',
    'activation_alpha', 'is_test', 'hidden_size', 'activations', 'beta', 'input_as_shape', 'drop_states', 'alpha',
    'momentum', 'scale', 'axis', 'dilations', 'transB', 'axis_w', 'blocksize', 'output_sequence', 'mode', 'perm',
    'min', 'seed', 'ends', 'paddings', 'to', 'gamma', 'width_scale', 'normalize_variance', 'group', 'ratio',
'values',
    'dtype', 'output_shape', 'spatial', 'split', 'input_forget', 'keepdims', 'transA', 'auto_pad', 'border', 'low',
    'linear_before_reset', 'height_scale', 'output_padding', 'shape', 'kernel_shape', 'epsilon', 'size', 'starts',
    'direction', 'max', 'clip', 'across_channels', 'value', 'strides', 'extra_shape', 'scales', 'k', 'sample_size',
    'blocksize', 'epsilon', 'momentum'
}

def get_onnx_tensor_data(tensor):
    """Get data from tensor."""
    assert isinstance(tensor, onnx.TensorProto)
    is_raw = False
    if tensor.float_data is not None:
        data = tensor.float_data
    elif tensor.int32_data is not None:
        data = tensor.int32_data
        is_raw = False
    elif tensor.string_data is not None:
        data = tensor.string_data
        is_raw = False
    elif tensor.int64_data is not None:
        data = tensor.int64_data
        is_raw = False
    elif tensor.double_data is not None:
        data = tensor.double_data
        is_raw = False
    elif tensor.uint64_data is not None:
        data = tensor.uint64_data
    else:
        data = tensor.raw_data
        is_raw = True

    if tensor.data_type == 1: #FLOAT
        dtype = 'float32'
    elif tensor.data_type == 6: #INT32
        dtype = 'int32'
    elif tensor.data_type == 7: #INT
        dtype = 'int64'
    elif tensor.data_type == 8: #string
        dtype = 'string'
    elif tensor.data_type == 11:  # string
        dtype = 'double'
    elif tensor.data_type == 12: #uint32
        dtype = 'uint32'
    elif tensor.data_type == 13: #uint32
        dtype = 'uint64'
    return [is_raw, data, dtype]


def map_onnx_dtype(dtype):
    return ONNX_TO_ANAKIN_DTYPE.get(dtype)

def onnx_to_anakin_tensor(tensor):
    """Convert onnx tensor to anakin med tensor."""
    #print 'tensor: ', tensor, type(tensor)
    shape = []
    for dim in tensor.dims:
        shape.append(int(dim))
    [is_raw, data, dtype] = get_onnx_tensor_data(tensor)
    #print 'shape: ', shape
    #print 'float_data: ', tensor.float_data
    # print(type(data),data,tensor.dtype,is_raw)
    if is_raw:
        if len(shape) > 0:
            #print 'type: ', tensor.data_type
            #print 'data: ', len(data)
            #print 'dtype: ', map_onnx_dtype(tensor.data_type)
            anakin_tensor = np.frombuffer(data, map_onnx_dtype(tensor.data_type))
            #print 'anakin: ', anakin_tensor
            anakin_tensor = anakin_tensor.reshape(shape)
        else:
            anakin_tensor = np.zeros(0)
        #print 'anakin_tensor: ', anakin_tensor
        return anakin_tensor, shape, dtype
    else:
        #print 'data'
        return data, shape, dtype

def has_key(attr, key_name):
    for it in attr.keys():
        if it == key_name:
            return True

    return False

def rm_weight_node(onnx_node, weights):
    for node in onnx_node.keys():
        in_node = onnx_node[node]['input']
        for name in in_node:
            if weights.has_key(name):
                in_node.remove(name)

def parse_Conv(onnx_node, weights):
   #print 'parse_Conv2D'
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Convolution'
    wei_name = onnx_node['input'][1]
    weights_node = weights[wei_name]
    if weights.has_key(wei_name):
        weights_node = weights[wei_name]
    else:
        print 'can not find weights', wei_name
    #assert weights_node['type'] == 'Const'
    weights_data = weights_node

    #print 'weights: ', weights_data
    #exit()
    bias_node = None
    if len(onnx_node['input']) > 2:
        bias_name = onnx_node['input'][2]
        bias_node = weights[bias_name]
        if weights.has_key(bias_name):
            bias_node = weights[bias_name]
        else:
            print 'can not find bias', bias_name
        '''
        print 'bias dtype', bias_node['dtype']
        print 'bias shape ', bias_node['shape']
        print 'bias data', bias_node['data']
        exit()
        '''
        onnx_node['input'].remove(bias_name)

    onnx_attr = onnx_node['onnx_attr']
    group = 1
    if 'group' in onnx_attr.keys():
        goup = onnx_attr['group']

    padding_val = []
    if 'pads' in onnx_attr.keys():
        #print 'pads: ', type(onnx_attr['pads'][0])
        padding_val = onnx_attr['pads']
    else:
        padding_val = [0, 0]

    dilations = []
    if 'dilations' in onnx_attr.keys():
        dilations = onnx_attr['dilations']
    else:
        dilations = [1, 1]

    strides = []
    if 'strides' in onnx_attr.keys():
        strides = onnx_attr['strides']
    else:
        strides = [1, 1]

    kernel_shape = onnx_attr['kernel_shape']
    #padding deal
    #if onnx_attr['auto_pad'] == 'SAME_LOWER' or onnx_attr['auto_pad'] =='SAME_UPPER':
    #    padding = [0, 0]
    #else:
    padding = [padding_val[1], padding_val[0]]

    ak_attr = onnx_node['ak_attr']
    ak_attr['weights'] = weights_data
    ak_attr['padding'] = padding
    ak_attr['dilations'] = dilations
    ak_attr['strides'] = strides
    ak_attr['kernel'] = kernel_shape
    ak_attr['group'] = group
    if bias_node is not None:
        ak_attr['bias'] = bias_node

    inputs = onnx_node['input']
    inputs.remove(wei_name)
    '''
    for name in inputs:
        if name == wei_name:
            inputs.remove(name)
        if name == bias_name:
            inputs.remove(bias_name)
    '''
    #print 'name: ', onnx_node['name']
    #print 'ak_attr: ', ak_attr
    #exit()

def parse_Gemm(onnx_node, weights):
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Dense'

    wei_name = onnx_node['input'][1]
    weights_node = weights[wei_name]
    if weights.has_key(wei_name):
        weights_node = weights[wei_name]
    else:
        print 'can not find weights', wei_name
    #assert weights_node['type'] == 'Const'
    weights_data = weights_node

    onnx_attr = onnx_node['onnx_attr']
    alpha = 0
    if 'alpha' in onnx_attr.keys():
        alpha = onnx_attr['alpha']

    beta = 0
    if 'beta' in onnx_attr.keys():
        beta = onnx_attr['beta']
    else:
        beta = 0

    transA = 0
    if 'transA' in onnx_attr.keys():
        transA = onnx_attr['transA']
    else:
        transA = 0

    transB = 0
    if 'transB' in onnx_attr.keys():
        transB = onnx_attr['transB']
    else:
        transB = 0

    ak_attr = onnx_node['ak_attr']
    if alpha == 0 or transA == 1:
        print 'Gemm Error, alpha, transA', alpha, transA
    if beta == 1:
        if len(onnx_node['input']) > 2:
            bias_name = onnx_node['input'][2]
            bias_node = weights[bias_name]
            if weights.has_key(bias_name):
                bias_node = weights[bias_name]
            else:
                print 'can not find bias', bias_name
            ak_attr['bias'] = bias_node
            onnx_node['input'].remove(bias_name)
    print 'name: ', onnx_node['name']
    print 'shape', weights_data['shape']
    if transB == 1:
        #print 'trans'
        ak_attr['trans'] = 1
        weights_data['data'] = np.transpose(weights_node['data'])
        weights_data['shape'] = [weights_data['shape'][1],  weights_data['shape'][0]]

    ak_attr['weights'] = weights_data
    #ak_attr['out_dim'] = weights_data

    onnx_node['input'].remove(wei_name)

def parse_Act(onnx_node, weights):
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Activation'
    if onnx_node['type'] == 'Relu':
        onnx_node['ak_type'] = 'Relu'
        onnx_node['ak_attr']['type'] = 'Relu'
    else:
        raise Exception('un handel activation '+str(onnx_node.op_type))

def parse_Concat(onnx_node, weights):
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Concat'
    onnx_attr = onnx_node['onnx_attr']
    ak_attr = onnx_node['ak_attr']
    if 'axis' in onnx_attr.keys():
        ak_attr['axis'] = onnx_attr['axis']
    else:
        ak_attr['axis'] = 0

def parse_Pooling(onnx_node, weights):
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Pooling'
    if onnx_node['type'] == 'MaxPool':
        ak_attr = onnx_node['ak_attr']
        ak_attr['type'] = 'MAX'

        onnx_attr = onnx_node['onnx_attr']
        padding_val = []
        if 'pads' in onnx_attr.keys():
            padding_val = onnx_attr['pads']
        else:
            padding_val = [0, 0]

        dilations = []
        if 'dilations' in onnx_attr.keys():
            dilations = onnx_attr['dilations']
        else:
            dilations = [1, 1]

        strides = []
        if 'strides' in onnx_attr.keys():
            strides = onnx_attr['strides']
        else:
            strides = [1, 1]

        kernel_shape = onnx_attr['kernel_shape']
        # padding deal
        #if onnx_attr['auto_pad'] == 'SAME_LOWER' or onnx_attr['auto_pad'] == 'SAME_UPPER':
        #    padding = [0, 0]
        # padding = [1, 1, 1, 1] =[left, right, top, bottom]
        #else:
        padding = [padding_val[1], padding_val[0]]

        ak_attr['window'] = kernel_shape
        ak_attr['padding'] = padding
        ak_attr['strides'] = strides

    if onnx_node['type'] == 'AveragePool':
        ak_attr = onnx_node['ak_attr']
        ak_attr['type'] = 'Average'

        onnx_attr = onnx_node['onnx_attr']
        padding_val = []
        if 'pads' in onnx_attr.keys():
            padding_val = onnx_attr['pads']
        else:
            padding_val = [0, 0]

        dilations = []
        if 'dilations' in onnx_attr.keys():
            dilations = onnx_attr['dilations']
        else:
            dilations = [1, 1]

        strides = []
        if 'strides' in onnx_attr.keys():
            strides = onnx_attr['strides']
        else:
            strides = [1, 1]

        kernel_shape = onnx_attr['kernel_shape']
        # padding deal
        if onnx_attr['atuo_pad'] == 'SAME_LOWER' or onnx_attr['atuo_pad'] == 'SAME_UPPER':
            padding = [0, 0]
        else:
            padding = [padding_val[1], padding_val[0]]

        ak_attr['window'] = kernel_shape
        ak_attr['padding'] = padding
        ak_attr['strides'] = strides

    if onnx_node['type'] == 'GlobalMaxPool':
        ak_attr = onnx_node['ak_attr']
        ak_attr['type'] = 'MAX'
        ak_attr['global_pooling'] = True

        onnx_attr = onnx_node['onnx_attr']
        padding_val = [0, 0]
        strides = [0, 0]
        kernel_shape = [1, 1]
        ak_attr['window'] = kernel_shape
        ak_attr['padding'] = padding_val
        ak_attr['strides'] = strides

    if onnx_node['type'] == 'GlobalAveragePool':
        ak_attr = onnx_node['ak_attr']
        ak_attr['type'] = 'Average'
        ak_attr['global_pooling'] = True

        onnx_attr = onnx_node['onnx_attr']
        padding_val = [0, 0]
        strides = [0, 0]
        kernel_shape = [1, 1]
        ak_attr['window'] = kernel_shape
        ak_attr['padding'] = padding_val
        ak_attr['strides'] = strides

def parse_Dropout(onnx_node, weights):
    onnx_node['visted'] = True
    onnx_node['ak_type'] = 'Scale'
    ak_attr = onnx_node['ak_attr'];
    '''
    ratio	(float, default 0.5) the ratio of random dropout
    is_test	(int) if nonzero, run dropout in test mode where the output is simply Y = X.
    '''
    if 'is_test' in onnx_node['onnx_attr'].keys():
        if onnx_node['onnx_attr']['is_test']  == 0:
            ak_attr['drop'] = 1
        else:
            ak_attr['drop'] = 0
    else:
        ak_attr['drop'] = 0
    scale_val = onnx_node['onnx_attr']['ratio']
    shape = [1, 1, 1, 1]
    scale_np = np.full(shape, scale_val); #np.arange([scale_val])
    weight_tensor = {}
    weight_tensor['shape'] = shape
    weight_tensor['data'] = scale_np
    weight_tensor['dtype'] = 'float32'
    ak_attr['weights'] = weight_tensor
    ak_attr['axis'] = 0
    ak_attr['num_axes'] = 0
