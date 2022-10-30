from models import BIT_TYPE_DICT

class Config:

    def __init__(self, int_softmax = True):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.INT_SOFTMAX = True
        self.INT_NORM = True
        self.INT_GELU = False

        self.BIT_TYPE_W = BIT_TYPE_DICT['int6']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_LN = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_S = BIT_TYPE_DICT['uint3']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = 'minmax'
        self.OBSERVER_S = 'minmax'
        self.OBSERVER_A_LN = 'minmax'

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_S = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        # self.QUANTIZER_A_LN = 'log2_for_norm'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_LN = 'channel_wise'





