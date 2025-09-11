
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
  --steps=200000 \
  --batch_size=64 \
  --output_dir=outputs/train/act_libero_10_task_9 \
  --job_name=act_libero_10_task_9 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=outputs/train/act_libero_10_task_9 \
  --policy.push_to_hub=false


### Train SmolVLA
lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=outputs/bl_mixed_100   --batch_size=64   --steps=40000  --policy.repo_id=outputs/train/test_mixed_smolvla_training --output_dir=outputs/train/test_mixed_smolvla_training   --job_name=test_mixed_smolvla_training   --policy.device=cuda   --wandb.enable=true

### Run A Policy
lerobot-record   --robot.type=so100_follower   --robot.port=/dev/ttyACM1   --robot.id=follower   --robot.max_relative_target=18   --robot.cameras="{ head: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}"    --dataset.single_task="Grasp a lego block and put it in the red area."  --teleop.type=so100_leader  --teleop.port=/dev/ttyACM0  --teleop.id=leader --dataset.repo_id=outputs/eval_bl_success_60_smolvla   --dataset.episode_time_s=50 --dataset.reset_time_s=3000  --dataset.num_episodes=100   --policy.path=outputs/train/bl_success_60_smolvla/checkpoints/last/pretrained_model

### Eval

python evaluate_lerobot_act.py   --model_path /home/josh/phddev/lerobot-upstream/outputs/train/act_libero_10_task_9/checkpoints/last/pretrained_model   --task_id 2   --benchmark libero_10   --n_eval 5   --max_steps 600   --device cuda   --seed 10000   --render

### Recalibrate Dataset

python /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/recalibrate_dataset.py \
  /home/josh/phddev/lerobot-upstream/outputs/bl_mixed_100 \
  --old-calibration /home/josh/phddev/lerobot/.cache/calibration/so100/main_follower.json \
  --new-calibration /home/josh/phddev/lerobot-upstream/calibration/robots/so100_follower/my_blue_follower_arm.json
Wrote recalibrated dataset to: /home/josh/phddev/lerobot-upstream/outputs/bl_mixed_100_calibrated