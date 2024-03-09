from ctypes import *


# ns3-ai environment structure
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('power', c_double),
        ('time', c_double),
        ('distance', c_double),
        ('station_id', c_uint32),
        ('n_successful', c_uint16),
        ('n_failed', c_uint16),
        ('mode', c_uint8),
        ('type', c_uint8),              # 0: New station created, 1: Sample new MCS
        ('report_source', c_uint8)      # 0: DoReportDataFailed, 1: DoReportDataOk, 2: DoReportAmpduTxStatus
    ]


# ns3-ai action structure
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('station_id', c_uint32),
        ('mode', c_uint8)
    ]
