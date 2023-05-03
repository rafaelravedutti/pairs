#!/bin/bash

count=0

# iterate over each remaining line in the file
while IFS= read -r line; do
  if [ $count -gt 0 ]; then
    # get a random number between 0 and 3
    random_num=$((RANDOM % 4))
    # add the random number and a comma to the beginning of the line
    printf "%d,%s\n" "$random_num" "$line"
  else
    printf "%s\n" "$line"
  fi
  count=$((count + 1))
done < "$1"
