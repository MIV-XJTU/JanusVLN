import os
import gzip
import json
import glob
import numpy as np
import concurrent.futures
from functools import partial
from tqdm import tqdm
import random
import argparse 

def process_episode_scalevln(ep, img_root, act_map):

    episode_results = []
    episode_id = str(ep['id'])
    instruction = ep['instructions'][0]
    name = ep['video'].split('/')[1]
    img_dir = os.path.join(img_root, name, 'rgb')
    missing_images_count = 0  
    

    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    except FileNotFoundError:
        return [] 

    num_images = len(img_files)
    if num_images == 0:
        return []
        
    for i in range(num_images):

        if i <= 8:
            idxs = list(range(i + 1))
        else:

            idxs = np.linspace(0, i, 9, dtype=int).tolist()
            idxs = sorted(list(set(idxs)))
            
        sampled_imgs = [img_files[j] for j in idxs]

        original_len = len(sampled_imgs)
        
        sampled_imgs = [img_path for img_path in sampled_imgs if os.path.exists(img_path)]
        
        num_missing = original_len - len(sampled_imgs)
        if num_missing > 0:
            missing_images_count += num_missing

        if not sampled_imgs:
            continue 
        
        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        

        if i + 1 >= len(ep['actions']):
             action = 'STOP'
        else:
            action = act_map[ep['actions'][i + 1]]
        
        
        conversations = [
            {"from": "human", "value": f"You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{his_img_tags}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
            {"from": "gpt", "value": action}
        ]
        
        sample_dict = {
            "id": f"{episode_id}/{os.path.basename(img_files[i])}",
            "conversations": conversations,
            "images": sampled_imgs
        }

        episode_results.append(sample_dict) 

    if missing_images_count != 0:
        pass

    return episode_results


def process_episode_vlnce(ep, img_root):
    episode_results = []
    episode_id = str(ep['episode_id'])
    instruction = ep['instruction']['instruction_text'].strip()
    img_dir = os.path.join(img_root, episode_id)
    
    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    except FileNotFoundError:
        return []

    if not img_files:
        return []
        

    reference_path = ep.get('reference_path', [])
    if not reference_path:
      
        return []
        
    num_steps = len(reference_path) -1

    for i in range(num_steps + 1):
      
        if i <= 8:
            idxs = list(range(i + 1))
        else:
     
            idxs = np.linspace(0, i, 9, dtype=int).tolist()
            idxs = sorted(list(set(idxs)))
            
  
        current_image_file = f"{reference_path[i]}.png"
        sampled_img_files = [os.path.join(img_dir, f"{reference_path[j]}.png") for j in idxs]

  
        sampled_imgs = [path for path in sampled_img_files if os.path.exists(path)]
        
        if not sampled_imgs:
            continue

        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        

        if i >= num_steps:
            action = 'STOP'
        else:

            action = ep['instruction']['instruction_tokens'][i+1][0]
            
        conversations = [
            {"from": "human", "value": f"You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{his_img_tags}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
            {"from": "gpt", "value": action}
        ]
        
        sample_dict = {
            "id": f"{episode_id}/{current_image_file}",
            "conversations": conversations,
            "images": sampled_imgs
        }

        episode_results.append(sample_dict)

    return episode_results


def main():
    parser = argparse.ArgumentParser(description="Process VLN datasets for training.")
    parser.add_argument(
        '--use_extra_data', 
        action='store_true',  
        help="Include extra datasets (ScaleVLN, DAgger R2R, DAgger RxR) in the processing."
    )
    args = parser.parse_args()

    all_results = []
    

    img_root_scalevln = "/mnt/nas-data-5/zengshuang.zs/ScaleVLN/images"
    json_path_scalevln = "/mnt/nas-data-5/zengshuang.zs/ScaleVLN/annotations.json"
    act_map_scalevln = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    img_root_dagger_r2r = "/mnt/nas-data-5/zengshuang.zs/data/dagger_data/R2R/images"
    json_path_dagger_r2r = "/mnt/nas-data-5/zengshuang.zs/data/dagger_data/R2R/annotations.json"
    act_map_dagger_r2r = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    img_root_dagger_rxr = "/mnt/nas-data-5/zengshuang.zs/data/dagger_data/RxR/images"
    json_path_dagger_rxr = "/mnt/nas-data-5/zengshuang.zs/data/dagger_data/RxR/annotations.json"
    act_map_dagger_rxr = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


    img_root_r2r = "/mnt/nas-data-5/zengshuang.zs/data/R2R-CE-640x480/train"
    json_path_r2r = "/mnt/nas-data-5/zengshuang.zs/VLN-CE/data/datasets/r2r/train/train.json.gz"
    
    img_root_rxr = "/mnt/nas-data-5/zengshuang.zs/data/RxR-CE-640x480/train"
    json_path_rxr = "/mnt/nas-data-5/zengshuang.zs/VLN-CE/data/datasets/rxr/train/train_guide.json.gz"

    
    datasets_to_process = []
    

    if args.use_extra_data:
        print("--- Mode: Including EXTRA datasets (ScaleVLN, DAgger) ---")
        datasets_to_process.extend([
            {"name": "ScaleVLN", "json_path": json_path_scalevln, "img_root": img_root_scalevln, "processor": process_episode_scalevln, "act_map": act_map_scalevln, "is_gz": False},
            {"name": "DAgger R2R", "json_path": json_path_dagger_r2r, "img_root": img_root_dagger_r2r, "processor": process_episode_scalevln, "act_map": act_map_dagger_r2r, "is_gz": False},
            {"name": "DAgger RxR", "json_path": json_path_dagger_rxr, "img_root": img_root_dagger_rxr, "processor": process_episode_scalevln, "act_map": act_map_dagger_rxr, "is_gz": False},
        ])
    else:
        print("--- Mode: Processing BASE datasets only (R2R, RxR) ---")
        print("To include extra data, run with the --use_extra_data flag.")

    datasets_to_process.extend([
        {"name": "R2R", "json_path": json_path_r2r, "img_root": img_root_r2r, "processor": process_episode_vlnce, "is_gz": True},
        {"name": "RxR", "json_path": json_path_rxr, "img_root": img_root_rxr, "processor": process_episode_vlnce, "is_gz": True},
    ])


    with concurrent.futures.ProcessPoolExecutor() as executor:
        for d in datasets_to_process:
            print(f"\nLoading data for {d['name']}...")
            if d['is_gz']:
                with gzip.open(d['json_path'], 'rt', encoding='utf-8') as f:
                    data = json.load(f)['episodes']
            else:
                with open(d['json_path'], 'r', encoding='utf-8') as f:
                    data = json.load(f)

            print(f"Processing {d['name']} dataset with {len(data)} episodes...")


            if d['processor'] == process_episode_scalevln:
                p_process = partial(d['processor'], img_root=d['img_root'], act_map=d['act_map'])
            else: 
                p_process = partial(d['processor'], img_root=d['img_root'])

            results_iterator = tqdm(executor.map(p_process, data), total=len(data))
            
            for episode_res in results_iterator:
                all_results.extend(episode_res)
            
            print(f"Finished {d['name']}. Total samples so far: {len(all_results)}")


    if args.use_extra_data:
        output_path = "train_r2r_rxr_extra.json"
    else:
        output_path = "train_r2r_rxr_base.json"

    print(f"\nAll processing finished. Shuffling and saving {len(all_results)} samples to {output_path}...")
    random.shuffle(all_results) 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
