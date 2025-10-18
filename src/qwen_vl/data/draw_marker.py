import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def scale_bbox(bbox, width, height):
    return (np.array(bbox[0]) / 1000) * np.array([width, height, width, height])

def scale_point(point, width, height):
    return (np.array(point[0]) / 1000) * np.array([width, height])

def draw_points(draw, image, data_entry, points):
    for color, key in points:
        point = scale_point(data_entry[key], image.width, image.height)
        point = tuple(map(int, point))
        draw.ellipse([point[0]-20, point[1]-20, point[0]+20, point[1]+20], fill=color)

def draw_thick_bbox(draw, image, bbox, color, stroke=20):
    bbox = scale_bbox(bbox, image.width, image.height)
    extend = stroke * 7 / 8
    bbox_out = [bbox[0] - extend, bbox[1] - extend, bbox[2] + extend, bbox[3] + extend]
    draw.rectangle(tuple(map(int, bbox_out)), outline=color, width=stroke)

def draw_spatial_relation_oo(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)
    draw_thick_bbox(draw, image, data_entry["blue_bbox"], "blue", stroke=20)

def draw_depth_prediction_oc(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point"), ("blue", "blue_point")])

def draw_depth_prediction_oo(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point"), ("blue", "blue_point"), ("green", "green_point")])

def draw_distance_prediction_oc(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

def draw_distance_prediction_oo(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point"), ("blue", "blue_point")])

def draw_distance_inference_oc(image, data_entry):
    pass 

def draw_distance_inference_oo(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point"), ("blue", "blue_point"), ("green", "green_point")])

def draw_spatial_volume_infer(image, data_entry):
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)

def draw_spatial_imagination_oc(image, data_entry):
    draw = ImageDraw.Draw(image)

    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)
    draw_thick_bbox(draw, image, data_entry["blue_bbox"], "blue", stroke=20)
    draw_thick_bbox(draw, image, data_entry["green_bbox"], "green", stroke=20)

def draw_position_matching(images, data_entry):
    image_to_draw = images[0]
    draw = ImageDraw.Draw(image_to_draw)

    draw_thick_bbox(draw, image_to_draw, data_entry["red_bbox"], "red", stroke=20)

def draw_spatial_imagination_oo(image, data_entry):
    draw = ImageDraw.Draw(image)

    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)
    draw_thick_bbox(draw, image, data_entry["blue_bbox"], "blue", stroke=20)
    draw_thick_bbox(draw, image, data_entry["green_bbox"], "green", stroke=20)
    draw_thick_bbox(draw, image, data_entry["yellow_bbox"], "yellow", stroke=20)

def draw_view_change_infer(images, data_entry):
    pass

def draw_depth_prediction_oc_mv(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("blue", "blue_point")])

def draw_depth_prediction_oo_mv(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("blue", "blue_point")])

    image = images[image_idx[2]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("green", "green_point")])

def draw_distance_prediction_oc_mv(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

def draw_distance_prediction_oo_mv(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("blue", "blue_point")])

def draw_obj_spatial_relation_oc_mv(images, data_entry):
    image_idx = data_entry["bbox_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)

    red_bbox = data_entry.get("red_bbox", None)
    if red_bbox:
        draw_thick_bbox(draw, image, red_bbox, "red", stroke=20)

    blue_bbox = data_entry.get("blue_bbox", None)
    if blue_bbox:
        draw_thick_bbox(draw, image, blue_bbox, "blue", stroke=20)
    
    green_bbox = data_entry.get("green_bbox", None)
    if green_bbox:
        draw_thick_bbox(draw, image, green_bbox, "green", stroke=20)

def draw_obj_spatial_relation_oo_mv(images, data_entry):
    image_idx = data_entry["bbox_img_idx"][0]
    red_bbox = data_entry.get("red_bbox", None)
    blue_bbox = data_entry.get("blue_bbox", None)
    green_bbox = data_entry.get("green_bbox", None)
    if red_bbox is None:
        image = images[image_idx[0]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, blue_bbox, "blue", stroke=20)

        image = images[image_idx[1]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, green_bbox, "green", stroke=20)
    elif blue_bbox is None:
        image = images[image_idx[0]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, red_bbox, "red", stroke=20)

        image = images[image_idx[1]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, green_bbox, "green", stroke=20)
    elif green_bbox is None:
        image = images[image_idx[0]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, red_bbox, "red", stroke=20)

        image = images[image_idx[1]]
        draw = ImageDraw.Draw(image)
        draw_thick_bbox(draw, image, blue_bbox, "blue", stroke=20)
    else:
        raise ValueError("Unexpected data entry")
    
def draw_distance_infer_center_oc_mv(images, data_entry):
    pass

def draw_distance_infer_center_oo_mv(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("green", "green_point")])

    image = images[image_idx[2]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("blue", "blue_point")])

def draw_spatial_imagination_oc_mv(images, data_entry):
    image_idx = data_entry["bbox_img_idx"][0]

    image = images[image_idx[2]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["green_bbox"], "green", stroke=20)

    
    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["blue_bbox"], "blue", stroke=20)


def draw_spatial_imagination_oo_mv(images, data_entry):
    image_idx = data_entry["bbox_img_idx"][0]

    image = images[image_idx[2]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["green_bbox"], "green", stroke=20)

    
    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["blue_bbox"], "blue", stroke=20)

    image = images[image_idx[3]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["yellow_bbox"], "yellow", stroke=20)

def draw_spatial_imagination_map_mv(images, data_entry):
    for i, bbox in enumerate(data_entry["bbox_list"][0]):
        img_idx = data_entry["bbox_img_idx"][0][i]
        image = images[img_idx]
        image = np.array(image)

        bbox = (np.array(bbox) / 1000) * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        bbox = bbox.astype(int)

        cv2.rectangle(image, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), color=(0, 0, 255), thickness=20)

        label = f"object{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 10
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

        text_position = (bbox[2] - text_size[0], bbox[3] + text_size[1])
        if text_position[1] > image.shape[0]:
            text_position = (bbox[2] - text_size[0], bbox[3] - text_size[1])

        cv2.putText(image, label, text_position, font, font_scale, color=(0, 255, 0), thickness=thickness)

        images[img_idx] = Image.fromarray(image)

def draw_distance_prediction_oo_video(images, data_entry):
    image_idx = data_entry["point_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("red", "red_point")])

    image = images[image_idx[1]]
    draw = ImageDraw.Draw(image)
    draw_points(draw, image, data_entry, [("blue", "blue_point")])

def draw_distance_infer_center_oo_video(images, data_entry):
    for i, point in enumerate(data_entry["point_list"][0]):
        img_idx = data_entry["point_img_idx"][0][i]
        image = images[img_idx]
        image = np.array(image)

        point = (np.array(point) / 1000) * np.array([image.shape[1], image.shape[0]])
        point = tuple(map(int, point))

        cv2.circle(image, point, radius=20, color=(0, 0, 255), thickness=-1)

        if i == 0:
            label = f"objectA"
        else:
            label = f"object{i-1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 10
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

        text_position = (point[0] + 20, point[1] + 20)

        cv2.putText(image, label, text_position, font, font_scale, color=(0, 255, 0), thickness=thickness)

        images[img_idx] = Image.fromarray(image)

def draw_spatial_imagination_oo_video(images, data_entry):
    image_indices = data_entry["bbox_img_idx"][0]
    
    bboxes_info = [
        {"bbox_key": "green_bbox", "color": (0, 255, 0), "label": "object0"},
        {"bbox_key": "blue_bbox",  "color": (255, 0, 0), "label": "object1"},
        {"bbox_key": "red_bbox",   "color": (0, 0, 255), "label": "object2"},
        {"bbox_key": "yellow_bbox", "color": (225, 225, 0),   "label": "object3"}
    ]
    
    for i, bbox_info in enumerate(bboxes_info):
        img_idx = image_indices[i]
        image = images[img_idx]
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
        
        bbox = data_entry[bbox_info["bbox_key"]]
        bbox_coords = (np.array(bbox[0]) / 1000) * np.array([image_np.shape[1], image_np.shape[0], image_np.shape[1], image_np.shape[0]])
        bbox_coords = bbox_coords.astype(int)
        bbox_tuple = tuple(bbox_coords)
        
        cv2.rectangle(image_np, 
                      (bbox_tuple[0]-20, bbox_tuple[1]-20), 
                      (bbox_tuple[2]+20, bbox_tuple[3]+20), 
                      color=bbox_info["color"], 
                      thickness=20)
        
        label = bbox_info["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness_text = 10
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness_text)
        
        text_x = bbox_tuple[2] - text_size[0]
        text_y = bbox_tuple[3] + text_size[1]
        
        if text_y > image_np.shape[0]:
            text_y = bbox_tuple[3] - 5
        
        text_position = (text_x, text_y)
        
        cv2.rectangle(image_np, 
                      (text_x, text_y - text_size[1]), 
                      (text_x + text_size[0], text_y + 5), 
                      bbox_info["color"], 
                      cv2.FILLED)
        
        cv2.putText(image_np, 
                    label, 
                    text_position, 
                    font, 
                    font_scale, 
                    (255, 255, 255),
                    thickness_text, 
                    cv2.LINE_AA)
        
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        images[img_idx] = image_pil

def draw_spatial_imagination_oc_video(images, data_entry):
    image_indices = data_entry["bbox_img_idx"][0]
    
    bboxes_info = [
        {"bbox_key": "green_bbox", "color": (0, 255, 0), "label": "object0"},
        {"bbox_key": "blue_bbox",  "color": (255, 0, 0), "label": "object1"},
        {"bbox_key": "red_bbox",   "color": (0, 0, 255), "label": "object2"},
    ]
    
    for i, bbox_info in enumerate(bboxes_info):
        img_idx = image_indices[i]
        image = images[img_idx]
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
        
        bbox = data_entry[bbox_info["bbox_key"]]
        bbox_coords = (np.array(bbox[0]) / 1000) * np.array([image_np.shape[1], image_np.shape[0], image_np.shape[1], image_np.shape[0]])
        bbox_coords = bbox_coords.astype(int)
        bbox_tuple = tuple(bbox_coords)
        
        cv2.rectangle(image_np, 
                      (bbox_tuple[0]-20, bbox_tuple[1]-20), 
                      (bbox_tuple[2]+20, bbox_tuple[3]+20), 
                      color=bbox_info["color"], 
                      thickness=20)
        
        label = bbox_info["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness_text = 10
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness_text)
        
        text_x = bbox_tuple[2] - text_size[0]
        text_y = bbox_tuple[3] + text_size[1]
        
        if text_y > image_np.shape[0]:
            text_y = bbox_tuple[3] - 5
        
        text_position = (text_x, text_y)
        
        cv2.rectangle(image_np, 
                      (text_x, text_y - text_size[1]), 
                      (text_x + text_size[0], text_y + 5), 
                      bbox_info["color"], 
                      cv2.FILLED)
        
        cv2.putText(image_np, 
                    label, 
                    text_position, 
                    font, 
                    font_scale, 
                    (255, 255, 255),
                    thickness_text, 
                    cv2.LINE_AA)
        
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        images[img_idx] = image_pil

def draw_obj_frame_locate(images, data_entry):
    image_idx = data_entry["bbox_img_idx"][0]

    image = images[image_idx[0]]
    draw = ImageDraw.Draw(image)
    draw_thick_bbox(draw, image, data_entry["red_bbox"], "red", stroke=20)

def draw_appearance_order(images, data_entry):
    pass

def draw_room_size(images, data_entry):
    pass

def draw_camera_motion_infer(images, data_entry):
    pass

def draw_nav(images, data_entry):
    pass

def draw_obj_count(images, data_entry):
    pass

def draw_spatial_imagination_oc_video_hard(images, data_entry):
    pass

def draw_spatial_imagination_oo_video_hard(images, data_entry):
    pass

DRAW_FUNCTIONS = {
    # Single View
    "obj_spatial_relation_oo": draw_spatial_relation_oo,
    "depth_prediction_oc": draw_depth_prediction_oc,
    "depth_prediction_oo": draw_depth_prediction_oo,
    "distance_prediction_oc": draw_distance_prediction_oc,
    "distance_prediction_oo": draw_distance_prediction_oo,
    "distance_infer_center_oc": draw_distance_inference_oc,
    "distance_infer_center_oo": draw_distance_inference_oo,
    "spatial_volume_infer": draw_spatial_volume_infer,
    "spatial_imagination_oc": draw_spatial_imagination_oc,
    "spatial_imagination_oo": draw_spatial_imagination_oo,

    # Multi View
    "position_matching": draw_position_matching,
    "view_change_infer": draw_view_change_infer,
    "depth_prediction_oc_mv": draw_depth_prediction_oc_mv,
    "depth_prediction_oo_mv": draw_depth_prediction_oo_mv,
    "distance_prediction_oc_mv": draw_distance_prediction_oc_mv,
    "distance_prediction_oo_mv": draw_distance_prediction_oo_mv,
    "obj_spatial_relation_oc_mv": draw_obj_spatial_relation_oc_mv, 
    "obj_spatial_relation_oo_mv": draw_obj_spatial_relation_oo_mv,
    "distance_infer_center_oc_mv": draw_distance_infer_center_oc_mv,
    "distance_infer_center_oo_mv": draw_distance_infer_center_oo_mv,
    "spatial_imagination_oc_mv": draw_spatial_imagination_oc_mv,
    "spatial_imagination_oo_mv": draw_spatial_imagination_oo_mv,
    "spatial_imagination_map_mv": draw_spatial_imagination_map_mv,
    "camera_motion_infer": draw_camera_motion_infer,

    # Video
    "distance_prediction_oo_video": draw_distance_prediction_oo_video,
    "distance_infer_center_oo_video": draw_distance_infer_center_oo_video,
    "spatial_imagination_oo_video": draw_spatial_imagination_oo_video,
    "spatial_imagination_oc_video": draw_spatial_imagination_oc_video,
    "spatial_imagination_oc_video_hard": draw_spatial_imagination_oc_video_hard,
    "spatial_imagination_oo_video_hard": draw_spatial_imagination_oo_video_hard,
    "obj_frame_locate": draw_obj_frame_locate,
    "appearance_order": draw_appearance_order,
    "room_size": draw_room_size,
    "obj_count": draw_obj_count,
    "nav": draw_nav,
}