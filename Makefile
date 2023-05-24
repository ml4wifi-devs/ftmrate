#PLG_PATH = plgkrusek@ares.cyfronet.pl:/net/ascratch/people/plgkrusek/ftmrate_internal
PLG_PATH = plgkrusek@ares.cyfronet.pl:/net/pr2/projects/plgrid/plggml4wifi/ftmrate_internal

LABSIM_PATH = labsim:/home/rusek/ml4wifi

.PHONY: to_plg from_plg fit_pf

to_plg_code:
	rsync -av setup.py *.sh  Makefile ml4wifi $(PLG_PATH)

to_plg: to_plg_code

from_plg:
	rsync -av --inplace --exclude '*.out' --exclude '*.err' '$(PLG_PATH)/log/*' log/plg


fit_pf:
	JAX_PLATFORM_NAME=cpu python3 ml4wifi/params_fit/pf_transition_noise.py --lr=0.05 --n_steps=2000

fit_pf_gpu:
	python3 ml4wifi/params_fit/pf_transition_noise.py --lr=0.05 --n_steps=2000
