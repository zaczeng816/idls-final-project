#!/bin/bash
python gen3.py --start-idx 4000 --end-idx 4680 & 
python gen3.py --start-idx 4680 --end-idx 5360 & 
python gen3.py --start-idx 5360 --end-idx 6040 & 
python gen3.py --start-idx 6040 --end-idx 6720 & 
python gen3.py --start-idx 6720 --end-idx 7400 &
wait