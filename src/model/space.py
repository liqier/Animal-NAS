from collections import OrderedDict


PRIMITIVES = [
    'none',
    'Part_Att_bottleneck',
    'Part_Share_Att_bottleneck',
    'Part_Conv_Att_bottleneck',
    'Channel_Att_bottleneck',
    #'Joint_Att_bottleneck',
    'Frame_Att_bottleneck',
    'Basic_bottleneck',
    'Basic_net'

]

primitives_5 = OrderedDict([('primitives_normal', 14 * [PRIMITIVES]),
                            ('primitives_reduce', 14 * [PRIMITIVES])])


spaces_dict = {
    's1': primitives_5
}



