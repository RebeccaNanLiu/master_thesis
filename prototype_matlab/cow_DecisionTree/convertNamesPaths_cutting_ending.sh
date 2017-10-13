#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
line="$(echo "${line}" | tr -d '\r')"
id=${line:0:59}
echo $id
done
