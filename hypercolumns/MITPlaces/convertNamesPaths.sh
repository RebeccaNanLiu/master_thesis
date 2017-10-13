#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
line="$(echo "${line}" | tr -d '\r')"
echo "/Users/nanliu/hypercolumns/MITPlaces/data/${line}"
done
