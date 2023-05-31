from scapy.all import *
import os.path

tx_address = '00:c2:c6:e6:9a:ec'

infile = 'sta2'
outfile = infile + '_fixed_no_ftm.pcap'

assert(not os.path.isfile(outfile)) #Check that the output file doesn't exist, we'll be appending to it

packets = rdpcap(infile+".pcap")

for i in range(len(packets)):
    p = packets[i]
    if p.haslayer(Dot11) and p.type == 2:
        if p.addr2 == tx_address: #Data frame sent by our station
            if p.time < 1e6 :
                p.time = packets[i-1].time
                wrpcap(outfile, p, append=True)
        elif p.addr2 != tx_address:
            wrpcap(outfile, p, append=True)
