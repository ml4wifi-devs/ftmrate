import os
import re
from argparse import ArgumentParser

from scapy.all import *


def set_mcs(mcs: int) -> None:
    RATE_MCS_ANT_MSK = 0x0c000
    RATE_MCS_HT_MSK = 0x00100

    monitor_tx_rate = 0x0
    monitor_tx_rate |= RATE_MCS_HT_MSK
    monitor_tx_rate |= RATE_MCS_ANT_MSK
    monitor_tx_rate |= mcs

    mask = '0x{:05x}'.format(monitor_tx_rate)

    path = '/sys/kernel/debug/ieee80211/'
    path += os.listdir(path)[0] + '/'
    path += [f for f in os.listdir(path) if re.match(r'.*:wl.*', f)][0] + '/stations/'
    path += os.listdir(path)[0] + '/rate_scale_table'

    os.system(f'echo {mask} | tee {path}')


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--dst_mac', type=str, default='00:c2:c6:e6:82:0f')
    args.add_argument('--src_mac', type=str, default='00:c2:c6:e6:9a:ec')
    args.add_argument('--bssid', type=str, default='00:c2:c6:e6:82:0f')
    
    args.add_argument('--count', type=int, default=100)
    args.add_argument('--frame_size', type=int, default=1000)
    args.add_argument('--interface', type=str, default='mon0')
    args.add_argument('--mcs', type=int, required=True)
    
    args = args.parse_args()

    data_payload = bytes(bytearray(args.frame_size))

    dot11 = Dot11(
        type='Data',
        subtype=0x28,
        addr1=args.dst_mac,
        addr2=args.src_mac,
        addr3=args.bssid,
        SC=0x3060,
        FCfield=0x01
    )
    frame = RadioTap() / dot11 / Raw(data_payload)

    set_mcs(args.mcs)

    for _ in range(args.count):
        sendp(frame, iface=args.interface, count=1)

    set_mcs(0)
