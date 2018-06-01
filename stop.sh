#!/bin/bash

# start master
pid_file=rl.pid

cat $pid_file | xargs kill -9
echo "" > $pid_file

ps -axu|grep rl_main|grep python|awk '{print $2}'|xargs kill -9 
ps -axu|grep rl_game|grep python|awk '{print $2}'|xargs kill -9 
