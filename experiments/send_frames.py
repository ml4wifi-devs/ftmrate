from argparse import ArgumentParser

from scapy.all import *


if __name__ == '__main__':
    args = ArgumentParser()
    
    args.add_argument('--dst_mac', type=str, default='00:16:ea:12:34:56')
    args.add_argument('--src_mac', type=str, default='00:16:ea:12:34:57')
    args.add_argument('--bssid', type=str, default='ff:ff:ff:ff:ff:ff')
    
    args.add_argument('--count', type=int, default=1)
    args.add_argument('--frame_size', type=int, default=1000)
    args.add_argument('--interface', type=str, default='mon0')
    
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

    while True:
        sendp(frame, iface=args.interface, count=args.count)
