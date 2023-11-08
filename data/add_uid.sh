#!/bin/bash

count=1

# iterate over each remaining line in the file
while IFS= read -r line; do
  printf "%d,%s\n" "$count" "$line"
  count=$((count + 1))
done < "$1"
