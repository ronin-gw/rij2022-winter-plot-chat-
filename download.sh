#!/bin/bash

python3 -m pip install -r requirements.txt

for i in 1688883287 1688886475 1690287845 1691997962; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i} > /dev/null
    fi
done

./main.py *.json
