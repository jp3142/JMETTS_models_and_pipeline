#!/usr/bin/env bash

# $1 = log_directory (directory containing a collection of tensorflow run folders)

# check python env
desired_env="julien_deepL"

if [ $CONDA_DEFAULT_ENV != $desired_env ]; then
	echo "[EnvError] Bad environment. Please make sure tensorflow is installed in your python env."
	exit 1
fi

# execute tensorboard script to check logs
python3 -m tensorboard.main --logdir="./$1" --port=6006 &

sleep 5 # wait until tensorboard is open
firefox -new-tab 'http://localhost:6006/' # open results at the end of the experiment

# wait until we quit firefox to quit tensorboard
echo -e "\nPress a key to quit tensorboard ..."
while [ true ] ; do
	read -t 3 -n 1
	if [ $? = 0 ] ; then
		# kill all processes active in background (in order to kill tensorboard, normally the only one active) when closing firefox
		for ids in $(lsof -i:6006 | tail -n +0 | cut -d' ' -f2,2); do # pass through all PID of the processes list
			kill $ids
		done
		
		#pkill /usr/lib/firefox/firefox-bin
		
		break
	fi
done
exit 0
