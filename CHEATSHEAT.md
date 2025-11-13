
### Convert Libero -> Lerobot
python convert_libero_to_lerobot.py \
    --input-path /path/to/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5 \
    --output-path /path/to/output/libero_kitchen_dataset \
    --repo-id username/libero_kitchen_tabletop_manipulation

### Teleop
lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader

### Record a Dataset

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.cameras="{ head: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --dataset.repo_id=outputs/bl_success_40 \
    --dataset.num_episodes=40 \
    --dataset.single_task="Place the lego brick inside the red area"

### Train a Policy

lerobot-train \
  --dataset.repo_id=outputs/libero_10_task_9 \
  --policy.type=act \
  --steps=100000 \
  --batch_size=64 \
  --output_dir=outputs/train/act_libero_10_task_9 \
  --job_name=act_libero_10_task_9 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=outputs/train/act_libero_10_task_9 \
  --policy.push_to_hub=false


### Train SmolVLA
lerobot-train \
  --policy.path=outputs/smolvla_base \
  --dataset.repo_id=outputs/libero_10_task_0 \
  --batch_size=32 \
  --epochs=100 \
  --policy.repo_id=outputs/train/test_mixed_smolvla_training \
  --output_dir=outputs/train/libero_10_task_0_smolvla_full_ft \
  --job_name=libero_10_task_0_smolvla_full_ft \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true

### Train Libero

lerobot-train \
  --policy.path=outputs/smolvla_base \
  --policy.repo_id=outputs/train/smolvla_libero_10 \
  --dataset.repo_id=outputs/libero_10 \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/train/smolvla_libero_10 \
  --save_freq=10000 \
  --steps=100000 \
  --batch_size=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=10000 \
  --policy.scheduler_decay_steps=150000 \
  --job_name=smolvla_libero \
  --policy.push_to_hub=false \
  --wandb.enable=false


### Train SmolVLA with Memory Layers
lerobot-train \
  --policy.path=outputs/smolvla_base \
  --policy.repo_id=outputs/train/smolvla_libero_90_mem_pretrained_2 \
  --dataset.repo_id=outputs/libero_90 \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir=./outputs/train/smolvla_libero_90_mem_pretrained_2 \
  --save_freq=10000 \
  --steps=200000 \
  --batch_size=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=10000 \
  --policy.scheduler_decay_steps=150000 \
  --job_name=smolvla_libero_90_mem_pretrained \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --policy.memory_layers=true \
  --policy.memory_layer.memory_only=true \
  --policy.memory_layer.layers="[10]" \
  --policy.memory_layer.vlm_layers="[10]" \
  --policy.memory_layer.log_usage=true \
  --policy.memory_layer.enabled=true \
  --policy.memory_layer.mem_n_keys=256 \
  --policy.memory_layer.mem_heads=4 \
  --policy.memory_layer.mem_knn=16 \
  --policy.memory_layer.mem_k_dim=256 \
  --policy.memory_layer.value_fixed_lr=0.001 \
  --policy.memory_layer.memory_lr=0.001


# Train SmolVLA sequentially

python -m lerobot.scripts.lerobot_sequential_train \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/libero_90_smolvla_memory/checkpoints/last/pretrained_model/ \
  --dataset.repo_id=outputs/libero_10 \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/train/smolvla_libero_10_mem_online_tfidf \
  --steps=200000 \
  --batch_size=64 \
  --num_workers=12 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --log_freq=200 \
  --wandb.enable=true \
  --job_name=smolvla_libero_10_mem_online_tfidf \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --online_steps_per_task=20000 \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --tfidf_enable=true \
  --tfidf_top_t=256 \
  --idf_stats_path=/home/josh/phddev/lerobot/outputs/cluster_train/libero_90_smolvla_memory/checkpoints/last/pretrained_model/memory_usage.json \
  --memory_value_lr=0.001

### Run A Policy
lerobot-record   --robot.type=so100_follower   --robot.port=/dev/ttyACM1   --robot.id=follower   --robot.max_relative_target=18   --robot.cameras="{ head: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}"    --dataset.single_task="Grasp a lego block and put it in the red area."  --teleop.type=so100_leader  --teleop.port=/dev/ttyACM0  --teleop.id=leader --dataset.repo_id=outputs/eval_bl_success_60_smolvla   --dataset.episode_time_s=50 --dataset.reset_time_s=3000  --dataset.num_episodes=100   --policy.path=outputs/train/bl_success_60_smolvla/checkpoints/last/pretrained_model

### Eval

python evaluate_lerobot_act.py   --model_path /home/josh/phddev/lerobot-upstream/outputs/train/act_libero_10_task_9/checkpoints/last/pretrained_model   --task_id 2   --benchmark libero_10   --n_eval 5   --max_steps 600   --device cuda   --seed 10000   --render

### Evaluate on LIBERO
lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot-upstream/outputs/train/libero_10_smolvla_200k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot-upstream/outputs/eval/libero_10_smolvla_200k

### Recalibrate Dataset

python /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/recalibrate_dataset.py \
  /home/josh/phddev/lerobot-upstream/outputs/bl_mixed_100 \
  --old-calibration /home/josh/phddev/lerobot/.cache/calibration/so100/main_follower.json \
  --new-calibration /home/josh/phddev/lerobot-upstream/calibration/robots/so100_follower/my_blue_follower_arm.json
Wrote recalibrated dataset to: /home/josh/phddev/lerobot-upstream/outputs/bl_mixed_100_calibrated



## Libero

**Libero 10**
                                                                                          task_index
put the white mug on the left plate and put the yellow and white mug on the right plate            0
put the white mug on the plate and put the chocolate pudding to the right of the plate             1
put the yellow and white mug in the microwave and close it                                         2
turn on the stove and put the moka pot on it                                                       3
put both the alphabet soup and the cream cheese box in the basket                                  4
put both the alphabet soup and the tomato sauce in the basket                                      5
put both moka pots on the stove                                                                    6
put both the cream cheese box and the butter in the basket                                         7
put the black bowl in the bottom drawer of the cabinet and close it                                8
pick up the book and place it in the back compartment of the caddy                                 9

**Other**
put the bowl on the plate                                                                         10
put the wine bottle on the rack                                                                   11
open the top drawer and put the bowl inside                                                       12
put the cream cheese in the bowl                                                                  13
put the wine bottle on top of the cabinet                                                         14
push the plate to the front of the stove                                                          15
turn on the stove                                                                                 16
put the bowl on the stove                                                                         17
put the bowl on top of the cabinet                                                                18
open the middle drawer of the cabinet                                                             19
pick up the orange juice and place it in the basket                                               20
pick up the ketchup and place it in the basket                                                    21
pick up the cream cheese and place it in the basket                                               22
pick up the bbq sauce and place it in the basket                                                  23
pick up the alphabet soup and place it in the basket                                              24
pick up the milk and place it in the basket                                                       25
pick up the salad dressing and place it in the basket                                             26
pick up the butter and place it in the basket                                                     27
pick up the tomato sauce and place it in the basket                                               28
pick up the chocolate pudding and place it in the basket                                          29
pick up the black bowl next to the cookie box and place it on the plate                           30
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate          31
pick up the black bowl on the ramekin and place it on the plate                                   32
pick up the black bowl on the stove and place it on the plate                                     33
pick up the black bowl between the plate and the ramekin and place it on the plate                34
pick up the black bowl on the cookie box and place it on the plate                                35
pick up the black bowl next to the plate and place it on the plate                                36
pick up the black bowl next to the ramekin and place it on the plate                              37
pick up the black bowl from table center and place it on the plate                                38
pick up the black bowl on the wooden cabinet and place it on the plate                            39

### Env Task Ids

00 put both the alphabet soup and the tomato sauce in the basket
01 put both the cream cheese box and the butter in the basket
02 turn on the stove and put the moka pot on it
03 put the black bowl in the bottom drawer of the cabinet and close it
04 put the white mug on the left plate and put the yellow and white mug on the right plate
05 pick up the book and place it in the back compartment of the caddy
06 put the white mug on the plate and put the chocolate pudding to the right of the plate
07 put both the alphabet soup and the cream cheese box in the basket
08 put both moka pots on the stove
09 put the yellow and white mug in the microwave and close it

### Mapping Dataset -> Env
0 → 4
1 → 6
2 → 9
3 → 2
4 → 7
5 → 0
6 → 8
7 → 1
8 → 3
9 → 5