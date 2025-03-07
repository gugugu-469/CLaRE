echo "pid:$$"
gpu=0
cd ../

data_dir=./dataset/ACE05
method_name=gpner18ace05WithRandom
model_dir=/root/nas/Models

model_type=bert
pretrained_model_name=bert-base-uncased
pair_filter=3
lr=5e-5
epoch=40
inner_dim=128
version1s=()
er_model_types=('er-h' 'er-t' 'er-shot' 'er-stoh')

train_batch_size=18
index=0
max_entity_length=10
save_er_model_types=()
entity_thresholds=(0.5)
seeds=(2024)
# TRAIN
for seed in ${seeds[@]}; do
    for er_model_type in ${er_model_types[@]}; do
        echo "er_model_type:${er_model_type}"
        version1=$(command date +%m-%d-%H-%M-%S --date="+8 hour")
        echo "version1:${version1}"
        python -u run_CLaRE.py --model_dir ${model_dir} --er_model_type ${er_model_type} --epochs ${epoch} --inner_dim ${inner_dim} --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate ${lr} --train_batch_size ${train_batch_size} --do_train --with_entity_type --seed ${seed} --pair_filter ${pair_filter}
        version1s=("${version1s[@]}" ${version1})
        save_er_model_types=("${save_er_model_types[@]}" ${er_model_type})
    done
done

len_version1=${#version1s[@]}
echo "len_version1:${len_version1}"

# PREDICT
index_version1=0
for er_model_type in ${save_er_model_types[@]}; do
    for entity_threshold in ${entity_thresholds[@]}; do
        echo "er_model_type:${er_model_type}"
        echo "entity_threshold:${entity_threshold}"
        version1=${version1s[${index_version1}]}
        echo "version1:${version1}"
        python -u run_CLaRE.py --model_dir ${model_dir} --er_model_type ${er_model_type} --inner_dim ${inner_dim} --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --train_batch_size ${train_batch_size} --do_predict --with_entity_type --entity_threshold ${entity_threshold} --max_entity_length ${max_entity_length} --pair_filter ${pair_filter}
    done
    let index_version1+=1  
done
echo "version1s:${version1s[*]}"
echo "save_er_model_types:${save_er_model_types[*]}"