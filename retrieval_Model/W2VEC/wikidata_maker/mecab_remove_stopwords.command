mecab mhlw.txt -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -Owakati -o mhlw_m.txt -b 30384
awk 'BEGIN{RS=ORS=" "} NR==FNR{r[$1];next} ($1 in r){next} 1' stop_words.txt input_file.txt

