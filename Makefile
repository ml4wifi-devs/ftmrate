PLG_PATH = plgkrusek@ares.cyfronet.pl:/net/ascratch/people/plgkrusek/ftmrate_internal
LABSIM_PATH = labsim:/home/rusek/ml4wifi

.PHONY: to_plg from_plg

to_plg_code:
	rsync -av setup.py *.sh  Makefile ml4wifi $(PLG_PATH)

to_plg: to_plg_code

from_plg:
	rsync -av --inplace --exclude '*.out' --exclude '*.err' '$(PLG_PATH)/log/*' log/plg


