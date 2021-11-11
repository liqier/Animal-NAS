from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat ')

PRIMITIVES = [
    'none',
    'skip_connect',
    'Part_Att_bottleneck',
    'Part_Share_Att_bottleneck',
    'Part_Conv_Att_bottleneck',
    'Channel_Att_bottleneck',
    #'Joint_Att_bottleneck',
    'Frame_Att_bottleneck',
    'Basic_bottleneck',
    'Basic_net'

]

AnimalNAS= genotype =  Genotype(normal=[('Frame_Att_bottleneck', 1), ('Frame_Att_bottleneck', 0),
                                        ('Basic_bottleneck', 0), ('Frame_Att_bottleneck', 2), ('Frame_Att_bottleneck', 2),
                                        ('Basic_bottleneck', 0), ('Basic_bottleneck', 0), ('Basic_bottleneck', 2)],
                                normal_concat=range(2, 6))




arch_1 = AnimalNAS