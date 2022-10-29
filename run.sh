CUDA_VISIBLE_DEVICES=0 python main.py \
               --result_path "./result_split1/" \
               --train_path "../../datasets/0113_split5/train" \
               --valid_path "../../datasets/0113_split5/test" \
               --test_path "../../datasets/0113_split5/test" \
               --optim 'sgd' \
               --lr 1e-2 \
               --batch_size 4 \
               --model_type "MSU_Net" \
               --check_name "only_specific" \
               --ms_mode "only_specific" \
               --eval_frequency 20 \
               --lamda 0


CUDA_VISIBLE_DEVICES=0 python main.py \
               --result_path "./result_split1/" \
               --train_path "../../datasets/0113_split5/train" \
               --valid_path "../../datasets/0113_split5/test" \
               --test_path "../../datasets/0113_split5/test" \
               --optim 'sgd' \
               --lr 1e-2 \
               --batch_size 4 \
               --model_type "MSU_Net" \
               --check_name "only_global" \
               --ms_mode "only_specific" \
               --eval_frequency 20 \
               --lamda 0



CUDA_VISIBLE_DEVICES=0 python main.py \
               --result_path "./result_split1/" \
               --train_path "../../datasets/0113_split5/train" \
               --valid_path "../../datasets/0113_split5/test" \
               --test_path "../../datasets/0113_split5/test" \
               --optim 'sgd' \
               --lr 1e-2 \
               --batch_size 4 \
               --model_type "MSU_Net" \
               --check_name "no_reg" \
               --ms_mode "" \
               --eval_frequency 20 \
               --lamda 0

CUDA_VISIBLE_DEVICES=0 python main.py \
               --result_path "./result_split1/" \
               --train_path "../../datasets/0113_split5/train" \
               --valid_path "../../datasets/0113_split5/test" \
               --test_path "../../datasets/0113_split5/test" \
               --optim 'sgd' \
               --lr 1e-2 \
               --batch_size 4 \
               --model_type "MSU_Net" \
               --check_name "lam0.8" \
               --ms_mode "" \
               --eval_frequency 20 \
               --lamda 0.8