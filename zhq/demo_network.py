# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:27:15 2020

@author: 99147
"""

from torchstat import stat

#from SEshufflenet import seshufflenet

from seg_cell5 import Discriminator, Generator

#Net = Res_Des_network(3,1)
Net = Generator()

#Net = MobileNetV3()
#Net = mobilenetv2()
#Net = mobilenet()
#Net = squeezenet()
#Net = shufflenetv2()
#Net = shufflenet()
#Net = MnasNet()

stat(Net, (3, 256, 256))