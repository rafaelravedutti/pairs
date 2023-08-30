#!/bin/bash

count=0

# iterate over each remaining line in the file
while IFS= read -r line; do
  if [ $count -gt 0 ]; then
    printf "%s,0\n" "$line"
  else
    printf "%s\n" "$line"
  fi
  count=$((count + 1))
done < "$1"
