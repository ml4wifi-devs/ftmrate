from ctypes import *


# ns3-ai environment structure
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('time', c_double),
        ('distance', c_double),
        ('station_id', c_uint32),
        ('mode', c_uint8),
        ('type', c_uint8)
    ]


# ns3-ai action structure
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('station_id', c_uint32),
        ('mode', c_uint8)
    ]
