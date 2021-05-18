#! /bin/bash

hadoop="/app/hadoop/bin/hadoop"
python="/app/anaconda3/bin//python"

#当前日期
echo "第一个参数为：$1"

# model
model_dir_v1=" "
modelPB_filename_v1=" "
ckpt2pb_pyfile=" "


echo "-----------------------------------------------------------------------"
current_date="$1"
echo "hdfs model put date is :  ${current_date}"
current_yester_date=$(date -d"yesterday ${current_date}" +%Y-%m-%d)

function put_modePB()
{
    hdfs_model_dir=" "
    ${hadoop} fs -mkdir ${hdfs_model_dir}
    ${hadoop} fs -put ${modelPB_filename_v1} ${hdfs_model_dir}
    rm -f ${modelPB_filename_v1}
    return $?
}

function final_bash()
{


    starttime=`date +'%Y-%m-%d %H:%M:%S'`

    echo "model ckpt to pb..."
    ${python} ${ckpt2pb_pyfile} ${model_dir_v1} ${modelPB_filename_v1} ${current_yester_date} 00 00 flj

    put_modePB
    echo "finish model train put modelPb to hdfs"
    endtime=`date +'%Y-%m-%d %H:%M:%S'`
    start_seconds=$(date --date="$starttime" +%s);
    end_seconds=$(date --date="$endtime" +%s);
    echo "本次运行时间： "$((end_seconds-start_seconds))"s"
    return $?
}

final_bash
