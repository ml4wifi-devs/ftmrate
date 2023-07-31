Prerequisite: enable remote root login on AP and STA by adding `PermitRootLogin yes` to `/etc/ssh/sshd_config` (done). The code below is to be done on your local machine.

Install paramiko:

    pip install paramiko

Dealing with ' Blowfish has been deprecated' warning:

> Commenting out the blowfish-cbc entire in the _cipher_info JSON of paramiko\transport.py has resolved the issue for me. this way there is no need to downgrade cryptography or suppress warnings.

        #"blowfish-cbc": {
        #    "class": algorithms.Blowfish,
        #    "mode": modes.CBC,
        #    "block-size": 8,
        #    "key-size": 16,
        #},

Connect to AP and STA to add SSH keys to known hosts:

   ssh opus@192.168.1.1
   ssh opus@192.168.1.7

Run `automate.py`. The output (pcap files) are generated on the AP in `/root/`.

TODO:

- change input variables (`distance`, `measurement`) to command line parameters (Done)
- change destination folder of pcap files
- update commands to be consistent with `ftmrate_internal/experiments/README.md` (?)