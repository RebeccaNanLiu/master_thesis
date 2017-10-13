#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
line="$(echo "${line}" | tr -d '\r')"
echo "/Users/nanliu/Documents/data/VOC2012/JPEGImages/${line}.jpg"
done
