#!/bin/zsh
# queued behind the examples run; one model in memory at a time
while pgrep -f exp1_examples.py >/dev/null; do sleep 15; done
BLOCKS=8,16 MARKER=quiet M=2 python3 exp3_chain_n.py > run_chain2_quiet.log 2>&1
BLOCKS=8,12,16,20 MARKER=quiet M=2 python3 exp3_chain_n.py > run_chain4_quiet.log 2>&1
