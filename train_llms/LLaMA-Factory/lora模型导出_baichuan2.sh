export CUDA_VISIBLE_DEVICES=7
   
src_path=(
xxxxx
)

export_path=(
xxxxx_export
)

for i in "${!src_path[@]}"; do
    echo "src_path:${src_path[$i]}"
    python src/export_model.py \
        --model_name_or_path  ../../llm_models/Baichuan2-7B-Chat \
        --adapter_name_or_path "${src_path[$i]}" \
        --template baichuan2 \
        --finetuning_type lora \
        --export_dir "${export_path[$i]}" \
        --export_size 3 \
        --export_legacy_format False
done