meta_path="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage3/metadata_C3.txt"
source="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/source"
dest="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage3/v2"
id_path="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/result/id"

lines=($(cat $meta_path | tr -d "\r"))
fignore=()

declare -a dignore=("id")
declare -i n=1
declare -i freq=0
declare -a cudas=(6 6 6 6 6 6)
function check_file_existance {
    # clear the log file
    > recover_swap_face.log

    local _folders=($(ls $dest))
    for folder in ${_folders[@]};do
        if [[ " ${dignore[*]} " =~ " ${folder} " ]];then
            continue
        fi
        local _sub_folders=($folder/1 $folder/2)
        for sub_folder in ${_sub_folders[@]};do
            local _videos=($(ls $dest/$sub_folder))
            for video in ${_videos[@]};do
                local vname=$(echo "$video" | cut -d "." -f 1)
                echo $sub_folder/$vname >> recover_swap_face.log
            done
        done
    done
}

function swap_faces {
    # retrieve for all existing videos and skip them
    check_file_existance
    fignore=($(cat recover_swap_face.log))

    declare -i n_process=5
    
    for line in ${lines[@]};do
        if [[ " ${fignore[*]} " =~ " ${line} " ]];then
            echo "skipped:" $line
            n=$(($n+1))
            continue
        fi

        v_p=$source/$line".mp4"
        out=$dest/$line".mp4"
        id_p=($(echo $line | tr '/' " "))

        if [[ ${id_p[1]} -eq 1 ]];then
            id_p[1]=2
        else
            id_p[1]=1
        fi
        n=$(($n+1))

        # multiprocessing run the test
        CUDA_VISIBLE_DEVICES=${cudas[$freq]} python test.py --gpu_ids 0 --crop_size 512 \
        --temp_path ./temp_results/$freq \
        --output_path $out \
        --use_mask --no_simswaplogo  \
        --video_path $v_p \
        --pic_a_path $id_path/${id_p[0]}/${id_p[1]}/id.png &

        echo On CUDA:${cudas[$freq]} processing video: $line - $n/${#lines[@]} >> swap_face.log

        freq=$(($freq+1))
        if [[ $freq -ge $n_process ]];then
            echo "waiting for processes to end" >> swap_face.log
            wait
            freq=0
        fi
    done

}


swap_faces
# check_file_existance