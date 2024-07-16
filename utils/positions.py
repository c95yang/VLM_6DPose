
'''
0,      1,       2
top,    bottom,  null 
left,   right,   null 
front,  back,    null
'''

class_to_coding = {
    "topleftfront": [0, 0, 0],
    "topleftback": [0, 0, 1],
    "topleft": [0, 0, 2],
    "toprightfront": [0, 1, 0],
    "toprightback": [0, 1, 1],
    "topright": [0, 1, 2],
    "topfront": [0, 2, 0],
    "topback": [0, 2, 1],
    "top": [0, 2, 2],
    "bottomleftfront": [1, 0, 0],
    "bottomleftback": [1, 0, 1],
    "bottomleft": [1, 0, 2],
    "bottomrightfront": [1, 1, 0],
    "bottomrightback": [1, 1, 1],
    "bottomright": [1, 1, 2],
    "bottomfront": [1, 2, 0],
    "bottomback": [1, 2, 1],            
    "bottom": [1, 2, 2],
    "leftfront": [2, 0, 0],
    "leftback": [2, 0, 1],
    "left": [2, 0, 2],
    "rightfront": [2, 1, 0],
    "rightback": [2, 1, 1],
    "right": [2, 1, 2],
    "front": [2, 2, 0],
    "back": [2, 2, 1],
    "null": [2, 2, 2]
}

classes = list(class_to_coding.keys())

def assign_label(filename):
    if filename == "render_position((0.40784254248348, -0.27499999999999997, -0.08966303887673437))_rotation(<Euler (x=1.7511, y=-0.0000, z=0.9775), order='XYZ'>).png"\
    or filename == "render_position((0.3273409913590437, -0.19166666666666668, -0.325748007307746))_rotation(<Euler (x=2.2804, y=-0.0000, z=1.0411), order='XYZ'>).png":
        return "bottomrightfront"
    elif filename == "render_position((0.244377030147626, -0.09166666666666665, 0.42647050233099176))_rotation(<Euler (x=0.5492, y=-0.0000, z=1.2119), order='XYZ'>).png"\
    or filename == "render_position((0.3414266937374401, -0.225, 0.2877547789412372))_rotation(<Euler (x=0.9576, y=0.0000, z=0.9881), order='XYZ'>).png":
        return "toprightfront"
    elif filename == "render_position((0.1178017771025154, -0.3083333333333333, 0.3755706283337994))_rotation(<Euler (x=0.7210, y=-0.0000, z=0.3649), order='XYZ'>).png":
        return "topright"
    elif filename == "render_position((0.1521582080309768, 0.2416666666666666, -0.4104206402595077))_rotation(<Euler (x=-0.6079, y=-3.1416, z=-0.5619), order='XYZ'>).png":
        return "bottomleft"
    elif filename == "render_position((0.2630477194325678, 0.425, 0.013449806739322445))_rotation(<Euler (x=1.5439, y=-0.0000, z=2.5874), order='XYZ'>).png":
        return "leftfront"
    elif filename == "render_position((0.2700806340632877, 0.02499999999999994, -0.4200374401216783))_rotation(<Euler (x=2.5682, y=0.0000, z=1.6631), order='XYZ'>).png":
        return "bottomfront"
    elif filename == "render_position((-0.3926120551202328, 0.058333333333333376, -0.3040608432476788))_rotation(<Euler (x=2.2245, y=-0.0000, z=-1.7183), order='XYZ'>).png":
        return "bottomback"
    elif filename == "render_position((0.2976086022771371, 0.34166666666666673, 0.21140721070847535))_rotation(<Euler (x=1.1342, y=-0.0000, z=2.4250), order='XYZ'>).png":
        return "topleftfront"
    elif filename == "render_position((0.3777830057983411, 0.15833333333333335, -0.28672383243380056))_rotation(<Euler (x=2.1815, y=0.0000, z=1.9677), order='XYZ'>).png"\
    or filename == "render_position((0.3880900421116613, 0.2916666666666667, -0.11965230783116747))_rotation(<Euler (x=1.8124, y=-0.0000, z=2.2153), order='XYZ'>).png":
        return "bottomleftfront"
    elif filename == "render_position((0.4752284334644108, -0.1416666666666667, 0.06394131358142126))_rotation(<Euler (x=1.4426, y=0.0000, z=1.2811), order='XYZ'>).png":
        return "front"
    elif filename == "render_position((0.08439351810290918, 0.04166666666666666, 0.49106173032634387))_rotation(<Euler (x=0.1894, y=0.0000, z=2.0294), order='XYZ'>).png"\
    or filename == "render_position((-0.08383140887469329, 0.17499999999999996, 0.4608115611462933))_rotation(<Euler (x=0.3986, y=0.0000, z=-2.6949), order='XYZ'>).png":
        return "top"
    elif filename == "render_position((0.011326587900092866, 0.39166666666666666, 0.31059447939196166))_rotation(<Euler (x=0.9005, y=-0.0000, z=3.1127), order='XYZ'>).png"\
    or filename == "render_position((0.08854168781443134, 0.47500000000000003, 0.12858992775086142))_rotation(<Euler (x=1.3107, y=0.0000, z=2.9573), order='XYZ'>).png":
        return "topleft"
    elif filename == "render_position((0.14260144961167637, -0.44166666666666665, 0.1859983390361432))_rotation(<Euler (x=1.1896, y=-0.0000, z=0.3123), order='XYZ'>).png"\
    or filename == "render_position((-0.02163491155805225, -0.17500000000000002, 0.467874909139051))_rotation(<Euler (x=0.3604, y=0.0000, z=-0.1230), order='XYZ'>).png"\
    or filename == "render_position((-0.08068526112163159, -0.39166666666666666, 0.3001451496525579))_rotation(<Euler (x=0.9269, y=0.0000, z=-0.2032), order='XYZ'>).png"\
    or filename == "render_position((-0.11512167919200951, -0.475, 0.10546088839001906))_rotation(<Euler (x=1.3583, y=-0.0000, z=-0.2378), order='XYZ'>).png":
        return "topright"
    elif filename == "render_position((-0.3460309834385277, 0.35833333333333334, 0.043125175046457284))_rotation(<Euler (x=1.4844, y=0.0000, z=-2.3737), order='XYZ'>).png"\
    or filename == "render_position((-0.3972576362517323, 0.225, 0.20386606004846988))_rotation(<Euler (x=1.1508, y=-0.0000, z=-2.0861), order='XYZ'>).png"\
    or filename == "render_position((-0.18067792855565465, 0.4416666666666667, 0.14928510203095707))_rotation(<Euler (x=1.2676, y=-0.0000, z=-2.7533), order='XYZ'>).png"\
    or filename == "render_position((-0.19901912215827608, 0.3083333333333334, 0.33959084877379236))_rotation(<Euler (x=0.8241, y=-0.0000, z=-2.5684), order='XYZ'>).png"\
    or filename == "render_position((-0.33378474972977884, 0.0916666666666667, 0.36081153400362787))_rotation(<Euler (x=0.7647, y=-0.0000, z=-1.8388), order='XYZ'>).png":
        return "topleftback"
    elif filename == "render_position((0.16104753521222676, -0.325, -0.3441492283909214))_rotation(<Euler (x=2.3299, y=-0.0000, z=0.4601), order='XYZ'>).png":
        return "bottomright"
    elif filename == "render_position((-0.3513764885953377, -0.29166666666666663, -0.20362985738335648))_rotation(<Euler (x=1.9902, y=0.0000, z=-0.8780), order='XYZ'>).png"\
    or filename == "render_position((-0.4709570039687757, -0.07499999999999998, -0.15024812948171637))_rotation(<Euler (x=1.8760, y=-0.0000, z=-1.4129), order='XYZ'>).png"\
    or filename == "render_position((-0.15243067061428453, -0.37499999999999994, -0.2934959806472305))_rotation(<Euler (x=2.1981, y=-0.0000, z=-0.3861), order='XYZ'>).png"\
    or filename == "render_position((-0.30386812828026066, -0.15833333333333335, -0.3641355738883618))_rotation(<Euler (x=2.3866, y=0.0000, z=-1.0904), order='XYZ'>).png":
        return "bottomrightback"
    elif filename == "ender_position((0.16140944100999083, 0.25833333333333336, 0.39649827394546266))_rotation(<Euler (x=0.6551, y=-0.0000, z=2.5831), order='XYZ'>).png":
        return "topleftfront"
    elif filename == "render_position((0.21436190790246865, 0.37500000000000006, -0.25184116510295435))_rotation(<Euler (x=-1.0429, y=-3.1416, z=-0.5193), order='XYZ'>).png":
        return "bottomleftfront"
    elif filename == "render_position((0.0023811984131786104, 0.10833333333333331, -0.48811701341277375))_rotation(<Euler (x=-0.2185, y=3.1416, z=-0.0220), order='XYZ'>).png"\
    or filename == "render_position((-0.16901447468928557, -0.024999999999999994, -0.46990329573807504))_rotation(<Euler (x=2.7929, y=-0.0000, z=-1.4239), order='XYZ'>).png"\
    or filename == "render_position((0.10713386522767156, -0.10833333333333336, -0.4762207721322832))_rotation(<Euler (x=2.8319, y=-0.0000, z=0.7798), order='XYZ'>).png":
        return "bottom"
    elif filename == "render_position((0.24346966415403654, -0.4083333333333333, -0.1548754710262718))_rotation(<Euler (x=1.8857, y=0.0000, z=0.5377), order='XYZ'>).png":
        return "bottomrightfront"
    elif filename == "render_position((0.027612022449169438, 0.4583333333333333, -0.19790940293938067))_rotation(<Euler (x=-1.1638, y=3.1416, z=-0.0602), order='XYZ'>).png"\
    or filename == "render_position((-0.07977499739325541, 0.325, -0.37149825005093373))_rotation(<Euler (x=-0.7333, y=3.1416, z=0.2407), order='XYZ'>).png":
        return "bottomleft"
    elif filename == "render_position((-0.05625146776246738, -0.24166666666666667, -0.43408869438951103))_rotation(<Euler (x=2.6223, y=-0.0000, z=-0.2287), order='XYZ'>).png"\
    or filename == "render_position((0.017469960272752345, -0.4583333333333333, -0.1990611866829494))_rotation(<Euler (x=1.9803, y=-0.0000, z=0.0381), order='XYZ'>).png":
        return "bottomright"
    elif filename == "render_position((-0.2025405818232446, 0.4083333333333334, -0.20552664451012287))_rotation(<Euler (x=1.9944, y=-0.0000, z=-2.6811), order='XYZ'>).png"\
    or filename == "render_position((-0.24595994412835182, 0.19166666666666668, -0.3908549536250845))_rotation(<Euler (x=2.4682, y=0.0000, z=-2.2328), order='XYZ'>).png"\
    or filename == "render_position((-0.37735071580991786, 0.27499999999999997, -0.1788335462874418))_rotation(<Euler (x=1.9366, y=-0.0000, z=-2.2006), order='XYZ'>).png":
        return "bottomleftback"
    elif filename == "render_position((0.32527500776280804, 0.12500000000000003, 0.3585682212981306))_rotation(<Euler (x=0.7711, y=0.0000, z=1.9377), order='XYZ'>).png":
        return "topleftfront"
    elif filename == "render_position((0.32754878723992376, -0.35833333333333334, 0.11962029175636288))_rotation(<Euler (x=1.3292, y=-0.0000, z=0.7405), order='XYZ'>).png":
        return "toprightfront"
    elif filename == "render_position((-0.1923591284901971, -0.04166666666666666, 0.45963230366824825))_rotation(<Euler (x=0.4046, y=0.0000, z=-1.3575), order='XYZ'>).png"\
    or filename == "render_position((-0.3374353665502443, -0.3416666666666666, 0.1392884140551223))_rotation(<Euler (x=1.2885, y=0.0000, z=-0.7792), order='XYZ'>).png"\
    or filename == "render_position((-0.24620906558149808, -0.2583333333333333, 0.35020706005786695))_rotation(<Euler (x=0.7948, y=0.0000, z=-0.7614), order='XYZ'>).png":
        return "toprightback"
    elif filename == "render_position((0.43836530623984593, 0.2083333333333333, 0.12013775637770312))_rotation(<Euler (x=1.3281, y=-0.0000, z=2.0144), order='XYZ'>).png"\
    or filename == "render_position((0.16140944100999083, 0.25833333333333336, 0.39649827394546266))_rotation(<Euler (x=0.6551, y=-0.0000, z=2.5831), order='XYZ'>).png":
        return "topleftfront"
    elif filename == "render_position((0.44253324294345403, -0.00833333333333333, 0.23258306998899433))_rotation(<Euler (x=1.0870, y=-0.0000, z=1.5520), order='XYZ'>).png":
        return "topfront"
    elif filename == "render_position((0.45079597192720794, -0.05833333333333332, -0.2082791730260761))_rotation(<Euler (x=2.0005, y=-0.0000, z=1.4421), order='XYZ'>).png":
        return "bottomfront"
    elif filename == "render_position((0.49265515041563346, 0.07499999999999998, -0.04081547217599686))_rotation(<Euler (x=1.6525, y=-0.0000, z=1.7219), order='XYZ'>).png":
        return "front"
    elif filename == "render_position((-0.08859095937619874, 0.49166666666666664, -0.020384572737578016))_rotation(<Euler (x=1.6116, y=0.0000, z=-2.9633), order='XYZ'>).png":
        return "left"
    elif filename == "render_position((0.09090593428863103, -0.49166666666666664, 0.0))_rotation(<Euler (x=1.5708, y=-0.0000, z=0.1828), order='XYZ'>).png":
        return "right"
    elif filename == "render_position((-0.397396333583049, -0.12500000000000003, 0.2764980181750858))_rotation(<Euler (x=0.9848, y=0.0000, z=-1.2660), order='XYZ'>).png":
        return "toprightback"
    elif filename == "render_position((-0.4774645301294175, 0.14166666666666664, -0.044251305334985835))_rotation(<Euler (x=1.6594, y=0.0000, z=-1.8592), order='XYZ'>).png":
        return "leftback"
    elif filename == "render_position((-0.25936500817620495, -0.425, -0.04587801797982511))_rotation(<Euler (x=1.6627, y=0.0000, z=-0.5479), order='XYZ'>).png"\
    or filename == "render_position((-0.45414152765165466, -0.20833333333333334, 0.018780177966238383))_rotation(<Euler (x=1.5332, y=0.0000, z=-1.1407), order='XYZ'>).png":
        return "rightback"
    elif filename == "render_position((-0.48341784724868986, 0.008333333333333333, 0.12742739311858303))_rotation(<Euler (x=1.3131, y=0.0000, z=-1.5880), order='XYZ'>).png":
        return "topback"
    else:
        return filename