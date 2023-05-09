#!/usr/bin/env python3

import argparse
import os
import matplotlib
from scapy.all import *  # this import also some graphical libs, that cause error when there is no X11 forwarding
from os import system
import os, re

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2020 Piotr Gawlowicz"
__version__ = "1.0.0"
__email__ = "gawlowicz.p@gmail.com"


RATE_MCS_ANT_A_MSK = 0x04000
RATE_MCS_ANT_B_MSK = 0x08000
RATE_MCS_ANT_C_MSK = 0x10000
RATE_MCS_HT_MSK = 0x00100
RATE_MCS_HT40_MSK = 0x00800


class WiFiTransmitter(object):
    """docstring for WiFiTransmitter"""
    def __init__(self, interfaceName):
        super(WiFiTransmitter).__init__()
        self.interfaceName = interfaceName
        self.phyName = None
        self.monitor_tx_rate = 0x0
        self.txRateChanged = False
        self.txAntStr = "A"
        self.txAntNum = 1
        self.txAntMask = 0
        self.streamNum = 1
        self.mcs = 0
        self.bwMask = 0x0

        self.filePath = None
        #filePath_Intel8260 = "/sys/kernel/debug/ieee80211/{}/netdev\:wlp1s0/stations/34\:f8\:e7\:c1\:57\:41/rate_scale_table".format('phy4')
        path = "/sys/kernel/debug/ieee80211/"
        path += os.listdir(path)[0]+"/" # Add single 'phy' directory
        #path += glob.glob('*wl*',dir_fd=path)[0]+"/stations/" # Add 'netdev\:wl*' directory, newer python
        #path += "netdev:wlp1s0/stations/" # Add 'netdev\:wl*' directory, workaround
        path += [f for f in os.listdir(path) if re.match(r'.*:wl.*', f)][0]+"/stations/" # Add 'netdev\:wl*' directory
        print(path)
        path += os.listdir(path)[0]+"/rate_scale_table" # Add single MAC directory
        filePath_Intel8260 = path
        #print(filePath_Intel8260)
        
        #cmd="echo 0x4214 | tee " + filePath_Intel8260
        #print(cmd)
        #os.system(cmd)
        
        #cmd2="less " + filePath_Intel8260
        #os.system(cmd2)

        self.filePath = filePath_Intel8260

        # default rate
        # has to be 802.11n
        self.monitor_tx_rate |= RATE_MCS_HT_MSK
        self.monitor_tx_rate |= RATE_MCS_ANT_A_MSK
        self.monitor_tx_rate |= self.mcs

        # frame generation
        self.addr1 = "00:00:00:00:00:00"
        self.addr2 = "00:00:00:00:00:00"
        self.addr3 = "ff:ff:ff:ff:ff:ff"

    def _read_file(self, fn):
        fd = open(fn, 'r')
        dat = fd.read()
        fd.close()
        return dat

    def _write_file(self, fn, msg):
        cmd="echo "+ msg + " | tee " + fn 
        print(cmd)
        os.system(cmd)
        return None

    def set_tx_antennas(self, txChains):
        self.txRateChanged = True

        self.txAntStr = txChains.upper()
        txChains = txChains.lower()

        mask = 0x0
        self.txAntNum = 0
        self.txAntMask = 0

        if "a" in txChains:
            mask |= RATE_MCS_ANT_A_MSK
            self.txAntNum += 1
        if "b" in txChains:
            mask |= RATE_MCS_ANT_B_MSK
            self.txAntNum += 1
        if "c" in txChains:
            mask |= RATE_MCS_ANT_C_MSK
            self.txAntNum += 1

        self.txAntMask |= mask

    def set_mcs(self, mcs):
        self.txRateChanged = True
        self.streamNum = 1
        if(mcs > 8):
            self.streamNum = 2

        if (mcs > 16):
            self.streamNum = 3

        if self.streamNum > self.txAntNum:
            print("Cannot use MCS: {} ({} streams) with {} antennas".format(mcs, self.streamNum, self.txAntNum))
            print("Set MCS to 0")
            mcs = 0

        print("mcs: ", mcs) 
        self.mcs = mcs

    def set_bandwidth(self, bw):
        self.bwMask = RATE_MCS_HT_MSK
        if bw == 20:
            self.bwMask = RATE_MCS_HT_MSK

        if bw == 40:
            self.bwMask |= RATE_MCS_HT40_MSK

    def set_mac_addresses(self, addr1, addr2, addr3):
        self.addr1 = addr1
        self.addr2 = addr2
        self.addr3 = addr3

    def _check_configuration(self):
        return True

    def _configure_tx_rate(self):
        if(self.txRateChanged):
            if (not self._check_configuration()):
                print("Cannot send a frame with given configuration:")
                print("---MCS: {}".format(self.mcs))
                print("---StreamNum: {}".format(self.streamNum))
                print("---AntennaNum: {} ({})".format(self.txAntNum, self.txAntStr))
                print("Send with default config: {}".format(self.txAntNum))
                print("---MCS: 0")
                print("---StreamNum: 1")
                print("---AntennaNum: 1 (A)")
                return

            self.monitor_tx_rate = 0x0
            self.monitor_tx_rate |= self.bwMask
            self.monitor_tx_rate |= self.txAntMask
            self.monitor_tx_rate |= self.mcs

        mask = "0x{:05x}".format(self.monitor_tx_rate)
        print("Set TX mask: ", mask)
        self._write_file(self.filePath, mask)

    def send(self, frameSize=100, interval=1, count=1):
        self._configure_tx_rate()

        rt = RadioTap()
        dot11 = Dot11(addr1=self.addr1,
                      addr2=self.addr2,
                      addr3=self.addr3)

        DOT11_SUBTYPE_QOS_DATA = 0x28
        ds=0x01
        dot11 = Dot11(type="Data", subtype=DOT11_SUBTYPE_QOS_DATA, addr1=self.addr1, addr2=self.addr2, addr3=self.addr3, SC=0x3060, FCfield=ds)

        for i in range(1, count + 1):
            payload = Raw(i.to_bytes(8, byteorder='big') + (0).to_bytes(frameSize - 8,  byteorder='big'))
            frame = rt / dot11 / payload
            frame1= 'IP(dst="192.168.1.1"/ICMP())'
            print(frame1)

            send(frame, iface="wlp1s0", count=1)
            time.sleep(interval)

            send(frame1, iface="wlp1s0", count=1)
            time.sleep(interval)

            print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')
    parser.add_argument('--interface',
                        type=str,
                        default="wlp1s0",
                        help='Select WiFi monitor interface')
    parser.add_argument('--txants',
                        type=str,
                        default="A",
                        help='Which TX chains should be activated, A, B or AB')
    parser.add_argument('--mcs',
                        type=int,
                        default=0,
                        help='MCS index')
    parser.add_argument('--bw',
                        type=int,
                        default=20,
                        help='Channel Bandwidth')
    parser.add_argument('--size',
                        type=int,
                        default=1000,
                        help='Frame size')
    parser.add_argument('--count',
                        type=int,
                        default=1000,
                        help='Number of frames to send')
    parser.add_argument('--interval',
                        type=float,
                        default=0,
                        help='Frame sending interval [s]')

    args = parser.parse_args()
    interface = str(args.interface)
    txants = args.txants
    mcs = args.mcs
    bandwidth = args.bw
    frameSize = args.size
    count = args.count
    interval = args.interval

    transmitter = WiFiTransmitter(interface)
    transmitter.set_tx_antennas(txants)
    transmitter.set_mcs(mcs)
    transmitter.set_bandwidth(bandwidth)

    dstMac = "00:c2:c6:e6:82:0f"
    srcMac = "00:c2:c6:e6:9a:ec"
    bssid = "00:c2:c6:e6:82:0f"
    transmitter.set_mac_addresses(dstMac, srcMac, bssid)

    transmitter.send(frameSize=frameSize, interval=interval, count=count)
