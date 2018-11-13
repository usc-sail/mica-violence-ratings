#!/bin/bash
set -euxo pipefail

echo "$0 $*"

script=$1
shift

device=$1
shift

fold_dir=$1
shift

out_dir="/home/victor/Workspace/abusive-lang-workshop/analysis/CV/fold_results/$1"
log_dir="/home/victor/Workspace/abusive-lang-workshop/analysis/CV/logs/$1"
shift

run1 () {
	fold=$1
	shift

	if [ ! -f $out_dir/fold$fold.res.npz ]; then
		PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$device ~/miniconda3/bin/python3.6 $script $fold_dir/$fold $out_dir/fold$fold.res --modelname $out_dir/best_model_$fold.hdf5 $* 1>$log_dir/fold$fold.log 2>&1 
	else
		echo "skipping $fold_dir/$fold $out_dir/fold$fold.res" 
	fi
}


if [ ! -d "$out_dir" ]; then
	mkdir "$out_dir" || true
else
	rm $out_dir/* || true
fi

if [ ! -d "$log_dir" ]; then
	mkdir $log_dir || true
fi

for i in {0..4}; do
	run1 $i $*
done