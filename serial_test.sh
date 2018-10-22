#!/bin/bash
for ((class=0; class <50 ; class++))
do
    parameter1="$class" 
    parameter2="$1" 
   ./test.py "$parameter1" "$parameter2"
done
