source="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage3/final_version1"
meta_path="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage3/metadata_C3.txt"
dest="/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage3/final_version2"

lines=($(cat $meta_path | tr -d "\r"))
fignore=()

declare -a dignore=("id")
declare -i n=1
declare -i freq=0
declare -a cudas=(6 6 6 6 6 6)
function check_file_existance {
    # clear the log file
    > recover_compression.log

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
                echo $sub_folder/$vname >> recover_compression.log
            done
        done
    done
}

function swap_faces {
    # retrieve for all existing videos and skip them
    check_file_existance
    fignore=($(cat recover_compression.log))

    declare -i n_process=3

    for line in ${lines[@]};do
        if [[ " ${fignore[*]} " =~ " ${line} " ]];then
            echo "skipped:" $line
            n=$(($n+1))
            continue
        fi

        v_p=$source/$line".mp4"
        out=$dest/$line".mp4"

        n=$(($n+1))

        ffmpeg -i $v_p  -s 1920x1080 -b:v 800k -b:a 127k $out &

        echo compressing video: $line - $n/${#lines[@]} >> swap_face.log

        freq=$(($freq+1))
        if [[ $freq -ge $n_process ]];then
            echo "waiting for processes to end" >> video_compression.log
            wait
            freq=0
        fi
    done

}


swap_faces
# check_file_existance