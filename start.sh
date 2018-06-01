#!/bin/bash

pid_file=rl.pid
workers=8

echo "start master..."
python rl_main.py $workers &

sleep 10s

echo "start games"
for i in $(seq 0 $(($workers-1)))
do
    python rl_game.py &
    echo $! >> $pid_file
    sleep 0.5
done
