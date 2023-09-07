import argparse
import time
from datetime import datetime

import paramiko
from playsound import playsound


# Configuration
AP_HOSTNAME = '192.168.1.1'
STA_HOSTNAME = '192.168.1.7'
PORT = 22
USERNAME = 'root'  # Root us required since paramiko does not support sudo
PASSWORD = 'opus'
TIMEOUT = 5
HOME_DIR = '/home/opus'

DATE = datetime.now().strftime('%Y%m%d-%H%M%S')

AP_MONITOR_CMD = f"nohup tcpdump -i mon0 -s 65000 -e  'ether host 00:c2:c6:e6:9e:d9 or ether host 00:c2:c6:e6:9a:ec' -w {HOME_DIR}/ap-{DATE}.pcap"
STA_MONITOR_CMD = f"nohup tcpdump -i mon0 -s 65000 -e 'ether host 00:c2:c6:e6:9e:d9 or ether host 00:c2:c6:e6:9a:ec' -w {HOME_DIR}/sta-{DATE}.pcap"

STA_TRANSMIT_CMD = f"nohup {HOME_DIR}/ftmrate_internal/experiments/scripts/run_send_frames.sh {HOME_DIR}"
STA_FTMRATE_CMD = f"nohup {HOME_DIR}/ftmrate_internal/experiments/scripts/run_ftmrate.sh {HOME_DIR} &> {HOME_DIR}/ftmrate_log-{DATE} &"

SLEEP_CMD = "sleep " + str(TIMEOUT)


def measure(framerate, duration, useFtmrate):

    with paramiko.SSHClient() as ap_ssh, paramiko.SSHClient() as sta_ssh:
        # Open SSH connection to AP
        ap_ssh.load_system_host_keys()
        ap_ssh.connect(AP_HOSTNAME, PORT, USERNAME, PASSWORD)

        # Open SSH connection to STA
        sta_ssh.load_system_host_keys()
        sta_ssh.connect(STA_HOSTNAME, PORT, USERNAME, PASSWORD)

        # Stop any leftover running processes (tcpdump on AP and STA, frame generation on STA)
        ap_ssh.exec_command("pkill tcpdump") 
        sta_ssh.exec_command("pkill tcpdump") 
        sta_ssh.exec_command("pkill python3")

        # Enable FTMRate
        if useFtmrate:
            print("Starting FTMRate on STA")    
            sta_ssh.exec_command(STA_FTMRATE_CMD)
            sta_ssh.exec_command(SLEEP_CMD)
            time.sleep(TIMEOUT)

        # Start tcpdump
        print("Starting tcpdump on AP")
        ap_ssh.exec_command(AP_MONITOR_CMD)
        print("Starting tcpdump on STA")
        sta_ssh.exec_command(STA_MONITOR_CMD)
        ap_ssh.exec_command(SLEEP_CMD)
        sta_ssh.exec_command(SLEEP_CMD)
        time.sleep(TIMEOUT)

        # Send frames from STA
        print("Sending frames from STA")
        sta_ssh.exec_command(f'{STA_TRANSMIT_CMD} {framerate} &> /dev/null &')
        # Play sound to indicate start of experiment
        playsound('sound.wav')
        time.sleep(duration - 1)  # -1 s to account for the duration of sound.wav

        # Cleanup
        print("Killing traffic generator")
        sta_ssh.exec_command("pkill python3") 
        print("Killing tcpdump")
        ap_ssh.exec_command("pkill tcpdump") 
        sta_ssh.exec_command("pkill tcpdump")
        print("Closing connection to AP")
        ap_ssh.close()
        print("Closing connection to STA")
        sta_ssh.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-r', '--framerate', required=True, type=int, help="Station packet generation rate")
    args.add_argument('-d', '--duration', required=True, type=int, help="Measurement duration [s]")
    args.add_argument('--useFtmrate', action=argparse.BooleanOptionalAction, help="Enable FTMRate as the rate manager")
    args = args.parse_args()
    
    print(f"[Info] Generating packets at a rate of {args.framerate} for {args.duration} seconds")

    if args.useFtmrate:
        print("[Info] Data rate manager: FTMRate")
    else:
        print("[Info] Data rate manager: default")
    
    measure(framerate=args.framerate, duration=args.duration, useFtmrate=args.useFtmrate)
    
    # Play sound to indicate end of experiment
    playsound('sound.wav')
