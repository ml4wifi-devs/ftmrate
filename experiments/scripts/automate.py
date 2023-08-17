import paramiko
from argparse import ArgumentParser
import winsound


# Configuration
AP_HOSTNAME = '192.168.1.1'
STA_HOSTNAME = '192.168.1.7'
PORT = 22
USERNAME = 'root'  # Root us required since paramiko does not support sudo
PASSWORD = 'opus'
HOME_DIR = '/home/opus'

AP_CMD = "tcpdump -i mon0 -e 'ether host 00:c2:c6:e6:9a:ec' -w mcs"
STA_CMD = f"{HOME_DIR}/ftmrate_venv/bin/python3.8 {HOME_DIR}/ftmrate_internal/experiments/success_probability/send_frames_sta.py --mcs "


def measure(distance, measurement, mcs_vals):

    with paramiko.SSHClient() as ap_ssh, paramiko.SSHClient() as sta_ssh:
        # Open SSH connection to AP
        ap_ssh.load_system_host_keys()
        ap_ssh.connect(AP_HOSTNAME, PORT, USERNAME, PASSWORD)

        # Open SSH connection to STA
        sta_ssh.load_system_host_keys()
        sta_ssh.connect(STA_HOSTNAME, PORT, USERNAME, PASSWORD)

        # Stop any leftover running processes (tcpdump on AP, frame generation on STA)
        ap_ssh.exec_command("pkill tcpdump") 
        sta_ssh.exec_command("pkill python3") 

        # Loop over MCS values
        for mcs in mcs_vals:
            print("=== MCS "+str(mcs)+" ===")

            # Open SSH transport channel on AP
            transport = ap_ssh.get_transport()
            channel = transport.open_session()
            channel.get_pty()
            channel.set_combine_stderr(True)

            # Start tcpdump
            print("Starting tcpdump on AP")
            channel.exec_command(AP_CMD+str(mcs)+'_d'+str(distance)+'_'+str(measurement)+'.pcap') 

            # Send frames from STA
            print("Sending frames from STA")
            _,stdout,_ = sta_ssh.exec_command(STA_CMD+str(mcs))
            print(stdout.read())

            # Stop tcpdump on AP
            print("Killing tcpdump...")
            _,stdout,_ = ap_ssh.exec_command("pkill tcpdump") 
            channel.close()     # close channel and let remote side terminate your proc.


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--distance', required=True, type=float, help="The AP-STA distance")
    args.add_argument('-m', '--measurement', required=True, type=int, help="The measurement number")
    args.add_argument('--mcs', default=None, type=int, nargs="*",
                      help="The MCS values to measure, if not specified, the default range is 0 to 15")
    args = args.parse_args()

    if not args.mcs:
        args.mcs = list(range(0, 16))
    
    print(f"[Info] Measurment {args.measurement} at distance {args.distance}")
    print(f"[Info] MCS values: {args.mcs}")
    
    measure(distance=args.distance, measurement=args.measurement, mcs_vals=args.mcs)
    winsound.PlaySound("sound.wav", winsound.SND_ALIAS)
