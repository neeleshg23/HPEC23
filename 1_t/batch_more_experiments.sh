LoadTrace_ROOT="/data/pengmiao/ML-DPC-S0/LoadTraces"

GPU_OPTIONS=("0" "1" "4" "5")

# Model names
MODELS=("m" "d" "l" "r")

NUM_GPUS=${#GPU_OPTIONS[@]}

gpu_index=1

app_list=(602.gcc-s0.txt.xz)

#for app1 in `ls $LoadTrace_ROOT`; do
for app1 in "${app_list[@]}"; do
    app="${app1::-7}"

    # if [ "$app" == "pr-3" ]; then
    #     python src/preprocess.py $app1 ${GPU_OPTIONS[$gpu_index]}
    # fi
    
    for ((i=0; i<${#MODELS[@]}; i++)); do
        model="${MODELS[i]}"
        
        GPU_OPTION="${GPU_OPTIONS[$gpu_index]}"
        
        ./run_pipeline.sh $app1 $model $GPU_OPTION

        ((gpu_index = (gpu_index + 1) % NUM_GPUS))
    done
done