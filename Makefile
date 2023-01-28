PLG_PATH = plgkrusek@prometheus.cyfronet.pl:/net/scratch/people/plgkrusek/ml4wifi
LABSIM_PATH = labsim:/home/rusek/ml4wifi

.PHONY: to_plg from_plg

to_plg_code:
	rsync -av python *.sh *.txt Makefile python epsilon-ts $(PLG_PATH)

to_plg: to_plg_code

from_plg:
	rsync -av --inplace --exclude '*.out' --exclude '*.err' '$(PLG_PATH)/log/*' log/plg


