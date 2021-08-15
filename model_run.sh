#! /bin/bash

hadoop="/app/hadoop/bin/hadoop"
python="/app/anaconda3/bin//python"

#当前日期
train_date="$1"
echo "train date :  ${train_date}"

model_pyfile_snpfinetuning="main.py"

function final_bash()
{
    echo "model is training..."
    ${python} -u ${model_pyfile_snpfinetuning} ${train_date}
    return $?
}

final_bash
