import os
import sys
import torch
import json
import argparse
import transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dist import *
import torch.distributed as dist
from evaluation import VLNEvaluator

import os
import random
import numpy as np
import torch
import tqdm
import copy
import json
import random
import habitat
import time
import gzip
from PIL import Image
from omegaconf import OmegaConf
from typing import List, Dict
from PIL import Image

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, append_text_underneath_image


from utils.dist import *
from habitat_extensions.maps import image_resize


import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

min_pixels: int = 28 * 28
max_pixels: int = 1605632

DATASET = "rxr"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = "./generated_data"

DEFAULT_EPISODE_LENGTH = 60
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85

class DAggerCollector:
    def __init__(self, args, rank, world_size):
        self.device = torch.device("cuda")
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)
        print(OmegaConf.to_yaml(self.config))

        with gzip.open(self.args.dagger_gt_annotations_path, 'rt', encoding='utf-8') as f:
            self.gt_annotations = json.load(f)
        
        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.dagger_config = OmegaConf.create({
            "p": self.args.dagger_p,
            "update_size": self.args.dagger_update_size,
            "commit_freq": self.args.dagger_commit_freq,
        })
        print(self.dagger_config)
        

    def config_env(self, scene=None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)
    
    
    def generate(
        self,
        env: habitat.Env,
        evaluator = None,
        save_video: bool = True,
        force_expert: bool = False,
    ) -> Dict:
        
        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it

        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        instructions = episode.instruction.instruction_text
        ref_path = episode.reference_path

        observation = env.reset()
        annotation = []
        rgb_data_list = []
        step_id = 0
        actions = [-1]
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, 'videos'), exist_ok=True)

        mem_ids=[]
        vis_frames = []
        left_expert_actions_num = 0
        from_expert = True if force_expert else False
        force_episode_end = False
        model_success = True
        action_seq, action_mask = [], []
        rgb_list,  time_ids = [], []
        evaluator.model.model.past_key_values_vggt = None
        metrics = None
        accumulated_error = 0 


        ref_actions_len = DEFAULT_EPISODE_LENGTH
        ref_actions_len = len(self.gt_annotations[str(episode_id)]['actions'])
        
        
        while not env.episode_over:
            time_ids.append(step_id)
            rgb = observation["rgb"]

            rgb_path = os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb", f"{step_id:03d}.jpg")
            
            rgb_data_list.append((rgb, rgb_path))
            

            if evaluator is not None:
                image = Image.fromarray(rgb).convert('RGB')
                rgb_list.append(image)
  
                if len(action_seq) == 0 and left_expert_actions_num == 0:
                    from_expert = True if force_expert else random.random() < beta
                
                if len(action_seq) == 0:
                    if left_expert_actions_num > 0:
                        action = agent.get_next_action(ref_path[next_waypoint_id])
                        action_seq = [action]
                        left_expert_actions_num -= 1
                    else:
                        if from_expert:
                            action = agent.get_next_action(ref_path[next_waypoint_id])
                            action_seq = [action]
                            left_expert_actions_num = self.args.num_future_steps - 1 
                        else:    
                            history_len = len(rgb_list) - 1  
                            if history_len <= evaluator.num_history:
                                history_images = rgb_list[:history_len]
                                images = history_images + [rgb_list[-1]]
                            else:
                                indices = np.linspace(0, history_len, evaluator.num_history + 1, dtype=int)
                                images = [rgb_list[i] for i in indices]
                            
                            action = evaluator.model.call_model(images, instructions,step_id)[0]
                            if action in evaluator.actions2idx:
                                action = evaluator.actions2idx[action][0]
                            else:
                                action = 0
                            action_seq = [action]

            else:
                action = agent.get_next_action(ref_path[next_waypoint_id])  
                action_seq = [action]
                pass

            action_source = "expert" if from_expert else "model"


            if len(action_seq) == 0:
                action_seq = [0]

            action = action_seq.pop(0)
            if action != agent.get_next_action(ref_path[next_waypoint_id]):
                accumulated_error += 1
            
            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break

            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)

            error_not_toleranted = ((from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0) or (accumulated_error/max(1,int(ref_actions_len/(len(ref_path)-1))) > 0.8) or accumulated_error > 12)
            if wp_id_available and error_not_toleranted:
                model_success = False
                force_expert = True
                accumulated_error = 0
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_source = "expert"
                action_seq = []
            

            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])
            

            observation = env.step(action)
            metrics = env.get_metrics()


            if save_video:
                metrics = env.get_metrics()
                if metrics['top_down_map'] is not None:
                    resized_rgb = np.array(image_resize(img=observation['rgb'],
                                                        size=(int(observation['rgb'].shape[0] * 1.6), int(observation['rgb'].shape[1] * 1.6)),
                                                        channels_last=True))
                    frame = observations_to_image({'rgb': resized_rgb}, metrics)
                    frame = append_text_underneath_image(frame, episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")

                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:            
                break
            actions.append(action)
            step_id += 1     
        
        assert len(rgb_data_list) == len(actions), f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {(actions)}"

        annotation.append({
            "id": episode_id,
            "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
            "instructions": instructions if isinstance(instructions, list) else [instructions],
            "actions": actions,
        })


        episode_save = metrics["distance_to_goal"] < MIDGOAL_RADIUS and (((not model_success) and (metrics["pl"] < RELATIVE_PATH_LENGTH_THRESHOLD)) or (metrics["pl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD))
        if episode_save:
            os.makedirs(os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb"), exist_ok=True)

            for rgb, rgb_path in rgb_data_list:
                Image.fromarray(rgb).convert("RGB").save(rgb_path)


        if save_video:
            if episode_save:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'save_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()
            else:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'notsave_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()

    
        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_episode_end,
            }
        )        
        
        episode_dict = dict(
            anno=annotation,
            metrics=metrics,
        )

        return episode_dict

    def update_dataset(self, evaluator, dataset=None):
        '''Update dataset with the collected data.'''
        
        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        
        env = self.config_env()
        scene_episode_dict = {}
        episode_uuids = []
        start = time.time()
        for episode in env.episodes:
            episode_uuid = (episode.scene_id, episode.episode_id, episode.trajectory_id)
            episode_uuids.append(episode_uuid)
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        sampled_episodes_uuids = episode_uuids
        sampled_episodes_by_scene = {}
        for scene_id in sorted(scene_episode_dict.keys()):
            sampled_episodes_traj_ids = [(episode_uuid[1], episode_uuid[2]) for episode_uuid in sampled_episodes_uuids if episode_uuid[0] == scene_id]
            sampled_episodes_by_scene[scene_id] = [ep for ep in scene_episode_dict[scene_id] if (ep.episode_id, ep.trajectory_id) in sampled_episodes_traj_ids]

        num_collect_episodes = 0
        start_id = 0
        annotations = []
        with tqdm.tqdm(total=min(self.dagger_config.update_size, len(sampled_episodes_uuids)) // self.world_size, dynamic_ncols=True) as pbar, \
            torch.no_grad():         
            for scene_id in sorted(scene_episode_dict.keys()):
                episodes = sampled_episodes_by_scene[scene_id]
                if len(episodes) == 0:
                    continue
                print(f"scene_id: {scene_id}, len of episodes: {len(episodes)}")
                for episode in episodes[self.rank::self.world_size]:  
                    assert scene_id == episode.scene_id, f"scene mismatch: {scene_id} vs {episode.scene_id}"          
                    scan = episode.scene_id.split('/')[-2]
                    env.current_episode = episode
                    env.current_episode.goals[0].radius = MIDGOAL_RADIUS 
                    
                    episode_dagger = self.generate(
                        env=env,
                        evaluator=evaluator,
                        save_video=self.args.dagger_save_video,
                        force_expert=self.args.force_expert
                    )

                    with open(os.path.join(self.output_path, f"result.json"), "a") as f:
                        result = {"scene": scan, 
                                  "episode_id": episode.episode_id, 
                                  "trajectory_id": episode.trajectory_id, 
                                  "save": episode_dagger["metrics"]["save"],
                                  "model_success": episode_dagger["metrics"]["model_success"], 
                                  "success": episode_dagger["metrics"]["success"], 
                                  "relative_pl": episode_dagger["metrics"]["pl"],
                                  "step_id": episode_dagger["metrics"]["step_id"],
                                  "ref_actions": episode_dagger["metrics"]["ref_actions_len"],
                                  "accumulated_error": episode_dagger["metrics"]["accumulated_error"],
                                  "force_episode_end": episode_dagger["metrics"]["force_episode_end"],
                                  }
                        f.write(json.dumps(result) + "\n")
                    
                    if not episode_dagger["metrics"]["save"]:
                        pbar.update()
                        continue 

                    for k,v in episode_dagger.items():
                        if isinstance(v, torch.Tensor):
                            episode_dagger[k] = v.numpy()

                    print(f"model_success = {episode_dagger['metrics']['model_success']}, scene {scan} id {episode.episode_id} trajectory {episode.trajectory_id}")

                    annotations.extend(episode_dagger['anno'])
                    pbar.update()
                    num_collect_episodes += 1


                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")

                        if os.path.exists(tgt_anno_path):
                            merged_anno = json.load(open(tgt_anno_path))
                        else:
                            merged_anno = []
                        with open(tgt_anno_path, "w") as json_file:
                            merged_anno.extend(annotations)
                            anno_videos = set()
                            for item in merged_anno:
                                anno_videos.add(item["video"])
                            temp_anno = []
                            for item in merged_anno:
                                if item["video"] in anno_videos:
                                    temp_anno.append(item)
                                    anno_videos.remove(item["video"])
                            merged_anno = temp_anno
                            json_data = json.dumps(merged_anno, indent=4)
                            json_file.write(json_data) 

                    if num_collect_episodes >= self.dagger_config.update_size:
                        break
                if num_collect_episodes >= self.dagger_config.update_size:
                    break

            tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")

            if os.path.exists(tgt_anno_path):
                merged_anno = json.load(open(tgt_anno_path))
            else:
                merged_anno = []
            with open(tgt_anno_path, "w") as json_file:
                merged_anno.extend(annotations)
                anno_videos = set()
                for item in merged_anno:
                    anno_videos.add(item["video"])
                temp_anno = []
                for item in merged_anno:
                    if item["video"] in anno_videos:
                        temp_anno.append(item)
                        anno_videos.remove(item["video"])
                merged_anno = temp_anno
                json_data = json.dumps(merged_anno, indent=4)
                json_file.write(json_data) 

            print(f"save scene_id {scene_id} with total episodes {num_collect_episodes} time cost {time.time() - start}")

        dist.barrier()
        if get_rank() == 0:
            tgt_anno_path = os.path.join(self.output_path, f"annotations.json")
            merged_anno = []
            sub_tgt_anno_list = [
                os.path.join(self.output_path, f)
                for f in os.listdir(self.output_path)
                if f.startswith('annotations_') and f.endswith('.json')
            ]
            for sub_tgt_anno_path in sub_tgt_anno_list:
                if os.path.exists(sub_tgt_anno_path):
                    merged_anno.extend(json.load(open(sub_tgt_anno_path)))
            merged_anno = sorted(merged_anno, key=lambda x: x['id'])
            with open(tgt_anno_path, "w") as json_file:
                anno_videos = set()
                for item in merged_anno:
                    anno_videos.add(item["video"])
                temp_anno = []
                for item in merged_anno:
                    if item["video"] in anno_videos:
                        temp_anno.append(item)
                        anno_videos.remove(item["video"])
                merged_anno = temp_anno
                json_data = json.dumps(merged_anno, indent=4)
                json_file.write(json_data)





class JanusVLN_Inference:
    def __init__(self, pretrained, device="cuda"):
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode='evaluation'
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device


    def call_model(
        self,
        observations, 
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        for i in enumerate([task]):
    
            visual = observations
            if isinstance(visual, Image.Image): 
                message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
                image_content = []
                image_count = 0
                for v in visual:
                    if add_frame_index:
                        image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                    image_content.append({"type": "image", "image": v})
                    image_count += 1
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")   
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]

                if i == len(vision_info) - 1:
                    cur_images_vggt.append(image)

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
    
            images_vggt.append(torch.stack(cur_images_vggt))
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device
        
        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers




if __name__ == "__main__":

    global local_rank
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_dagger.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=1)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument("--dagger_p",type=float, default=0.9)
    parser.add_argument("--dagger_update_size", type=int, default=1)
    parser.add_argument("--dagger_commit_freq",type=int, default=1)
    parser.add_argument("--dagger_dataset", type=str, default=DATASET)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_output_path",type=str, default="data/dagger")
    parser.add_argument("--dagger_data_path", type=str, default="data/datasets/vln_datasets/{split}.json.gz")
    parser.add_argument("--dagger_gt_annotations_path", type=str, default="data/datasets/vln_datasets/annotations.json")
    parser.add_argument("--dagger_save_video", action="store_true", default=False, help="whether to save video during dagger collection")

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    model = JanusVLN_Inference(args.model_path, device=f"cuda:{local_rank}")

        
    rank = get_rank()
    world_size = get_world_size()


    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    
    collector = DAggerCollector(args=args, rank=rank, world_size=world_size)
    collector.update_dataset(evaluator=evaluator)