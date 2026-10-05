#!/bin/bash
# Fetch ARC-AGI-1, ARC-AGI-2 and RE-ARC into raw_data/ (run from the repository root).
set -e
cd "$(dirname "$0")"
[ -d ARC-AGI ]   || git clone --depth 1 https://github.com/fchollet/ARC-AGI.git ARC-AGI
[ -d ARC-AGI-2 ] || git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git ARC-AGI-2
if [ ! -d re_arc/tasks ]; then
  git clone --depth 1 https://github.com/michaelhodel/re-arc.git _re_arc_src
  mkdir -p re_arc
  unzip -q -o _re_arc_src/re_arc.zip -d re_arc_tmp
  # the archive unpacks to re_arc_tmp/re_arc/{tasks,metadata.json} or directly to re_arc_tmp/{tasks,metadata.json}
  if [ -d re_arc_tmp/re_arc ]; then mv re_arc_tmp/re_arc/* re_arc/; else mv re_arc_tmp/* re_arc/; fi
  rm -rf re_arc_tmp _re_arc_src
fi
echo "ARC-AGI tasks:   $(ls ARC-AGI/data/training | wc -l) training / $(ls ARC-AGI/data/evaluation | wc -l) evaluation"
echo "ARC-AGI-2 tasks: $(ls ARC-AGI-2/data/training | wc -l) training / $(ls ARC-AGI-2/data/evaluation | wc -l) evaluation"
echo "RE-ARC tasks:    $(ls re_arc/tasks | wc -l)"
