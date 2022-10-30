#!/bin/bash
#save_path="$1"
class_id="$1"

#读取文件内容并保存
id_arr=($(cat $class_id | sed '/^#.*\|^$/d'))
id_num=${#id_arr[@]}

n=0
for var1 in ${id_arr[*]}
do
    #$(wget -c ${id_arr[n]} -O ${name_arr[n]})
    $(wget -c "http://www.image-net.org/downloads/bbox/bbox/${id_arr[n]}.tar.gz" -O "${id_arr[n]}.tar.gz")
    $(wget -c "http://www.image-net.org/download/synset?wnid=${id_arr[n]}&username=[mhuu0021]&accesskey=[QZH2008001]&release=latest&src=stanford" -O "${id_arr[n]}.tar")
    let n+=1
done

echo "download finsh"
