from dotenv import load_dotenv
import os
import json
import re
import copy
import yaml
from jinja2 import Environment, StrictUndefined

from utils.src.utils import ppt_to_images, get_json_from_response

from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from utils.pptx_utils import *
from utils.wei_utils import *

load_dotenv()

IMAGE_SCALE_RATIO_MIN = 50
IMAGE_SCALE_RATIO_MAX = 40
TABLE_SCALE_RATIO_MIN = 100
TABLE_SCALE_RATIO_MAX = 80

def compute_tp(raw_content_json):
    total_length = 0
    for section in raw_content_json['sections']:
        total_length += len(section['content'])

    for i in range(len(raw_content_json['sections'])):
        raw_content_json['sections'][i]['tp'] = len(raw_content_json['sections'][i]['content']) / total_length
        raw_content_json['sections'][i]['text_len'] = len(raw_content_json['sections'][i]['content'])

def compute_gp(table_info, image_info):
    total_area = 0
    for k, v in table_info.items():
        total_area += v['figure_size']

    for k, v in image_info.items():
        total_area += v['figure_size']

    for k, v in table_info.items():
        v['gp'] = v['figure_size'] / total_area

    for k, v in image_info.items():
        v['gp'] = v['figure_size'] / total_area

def filter_image_table(args, filter_config):
    images = json.load(open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_images.json', 'r'))
    tables = json.load(open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_tables.json', 'r'))
    doc_json = json.load(open(f'contents/{args.model_name_t}_{args.poster_name}_raw_content.json', 'r'))
    agent_filter = 'image_table_filter_agent'
    with open(f"utils/prompt_templates/{agent_filter}.yaml", "r", encoding="utf-8") as f:
        config_filter = yaml.safe_load(f)

    image_information = {}
    for k, v in images.items():
        image_information[k] = copy.deepcopy(v)
        image_information[k]['min_width'] = v['width'] // IMAGE_SCALE_RATIO_MIN
        image_information[k]['min_height'] = v['height'] // IMAGE_SCALE_RATIO_MIN
        image_information[k]['max_width'] = v['width'] // IMAGE_SCALE_RATIO_MAX
        image_information[k]['max_height'] = v['height'] // IMAGE_SCALE_RATIO_MAX

    table_information = {}
    for k, v in tables.items():
        table_information[k] = copy.deepcopy(v)
        table_information[k]['min_width'] = v['width'] // TABLE_SCALE_RATIO_MIN
        table_information[k]['min_height'] = v['height'] // TABLE_SCALE_RATIO_MIN
        table_information[k]['max_width'] = v['width'] // TABLE_SCALE_RATIO_MAX
        table_information[k]['max_height'] = v['height'] // TABLE_SCALE_RATIO_MAX

    filter_actor_sys_msg = config_filter['system_prompt']

    if args.model_name_t.startswith('vllm_qwen'):
        filter_model = ModelFactory.create(
            model_platform=filter_config['model_platform'],
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
            url=filter_config['url'],
        )
    else:
        filter_model = ModelFactory.create(
            model_platform=filter_config['model_platform'],
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
        )

    filter_actor_agent = ChatAgent(
        system_message=filter_actor_sys_msg,
        model=filter_model,
        message_window_size=10,
    )

    filter_jinja_args = {
        'json_content': doc_json,
        'table_information': json.dumps(table_information, indent=4),
        'image_information': json.dumps(image_information, indent=4),
    }
    jinja_env = Environment(undefined=StrictUndefined)
    filter_prompt = jinja_env.from_string(config_filter["template"])
    filter_actor_agent.reset()
    response = filter_actor_agent.step(filter_prompt.render(**filter_jinja_args))
    input_token, output_token = account_token(response)
    response_json = get_json_from_response(response.msgs[0].content)
    table_information = response_json['table_information']
    image_information = response_json['image_information']
    json.dump(image_information, open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_images_filtered.json', 'w'), indent=4)
    json.dump(table_information, open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_tables_filtered.json', 'w'), indent=4)

    return input_token, output_token

def gen_outline_layout_v2(args, actor_config):
    total_input_token, total_output_token = 0, 0
    agent_name = 'poster_planner_new_v2'
    doc_json = json.load(open(f'contents/{args.model_name_t}_{args.poster_name}_raw_content.json', 'r'))
    filtered_table_information = json.load(open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_tables_filtered.json', 'r'))
    filtered_image_information = json.load(open(f'{args.model_name_t}_images_and_tables/{args.poster_name}_images_filtered.json', 'r'))

    # Normalise: the LLM sometimes returns a list instead of a dict keyed by index.
    # Recover the original numeric ID from the image/table file path so that later
    # lookups like `filtered_image_information[str(figure['image'])]` still work.
    def _list_to_dict(items: list, path_key: str) -> dict:
        result = {}
        for i, item in enumerate(items):
            path = item.get(path_key, '')
            m = re.search(r'-(\d+)\.\w+$', path)
            key = m.group(1) if m else str(i + 1)
            result[key] = item
        return result

    if isinstance(filtered_table_information, list):
        filtered_table_information = _list_to_dict(filtered_table_information, 'table_path')
    if isinstance(filtered_image_information, list):
        filtered_image_information = _list_to_dict(filtered_image_information, 'image_path')

    filtered_table_information_captions = {}
    filtered_image_information_captions = {}

    for k, v in filtered_table_information.items():
        filtered_table_information_captions[k] = {
            v['caption']
        }

    for k, v in filtered_image_information.items():
        filtered_image_information_captions[k] = {
            v['caption']
        }

    with open(f"utils/prompt_templates/{agent_name}.yaml", "r", encoding="utf-8") as f:
        planner_config = yaml.safe_load(f)

    compute_tp(doc_json)

    jinja_env = Environment(undefined=StrictUndefined)
    outline_template = jinja_env.from_string(planner_config["template"])
    planner_jinja_args = {
        'json_content': doc_json,
        'table_information': filtered_table_information_captions,
        'image_information': filtered_image_information_captions,
    }

    if args.model_name_t.startswith('vllm_qwen'):
        planner_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
        planner_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )


    planner_agent = ChatAgent(
        system_message=planner_config['system_prompt'],
        model=planner_model,
        message_window_size=10,
    )

    print(f'Generating outline...')
    planner_prompt = outline_template.render(**planner_jinja_args)
    planner_agent.reset()
    response = planner_agent.step(planner_prompt)
    input_token, output_token = account_token(response)
    total_input_token += input_token
    total_output_token += output_token

    figure_arrangement = get_json_from_response(response.msgs[0].content)

    print(f'Figure arrangement: {json.dumps(figure_arrangement, indent=4)}')

    arranged_images = {}
    arranged_tables = {}
    assigned_images = set()
    assigned_tables = set()
    
    for section_name, figure in figure_arrangement.items():
        if 'image' in figure:
            image_id = str(figure['image'])
            if image_id in assigned_images:
                continue
            if image_id in filtered_image_information:
                arranged_images[image_id] = filtered_image_information[image_id]
                assigned_images.add(image_id)
        if 'table' in figure:
            table_id = str(figure['table'])
            if table_id in assigned_tables:
                continue
            if table_id in filtered_table_information:
                arranged_tables[table_id] = filtered_table_information[table_id]
                assigned_tables.add(table_id)
    
    compute_gp(arranged_tables, arranged_images)

    # Obtain panel input
    paper_panels = []
    for i in range(len(doc_json['sections'])):
        section = doc_json['sections'][i]
        panel = {}
        panel['panel_id'] = i
        panel['section_name'] = section['title']
        panel['tp'] = section['tp']
        panel['text_len'] = section['text_len']
        panel['gp'] = 0
        panel['figure_size'] = 0
        panel['figure_aspect'] = 1
        if section['title'] in figure_arrangement:
            curr_arrangement = figure_arrangement[section['title']]
            if 'table' in curr_arrangement:
                table_id = str(curr_arrangement['table'])
                if table_id in arranged_tables:
                    panel['gp'] = arranged_tables[table_id]['gp']
                    panel['figure_size'] = arranged_tables[table_id]['figure_size']
                    panel['figure_aspect'] = arranged_tables[table_id]['figure_aspect']
            elif 'image' in curr_arrangement:
                image_id = str(curr_arrangement['image'])
                if image_id in arranged_images:
                    panel['gp'] = arranged_images[image_id]['gp']
                    panel['figure_size'] = arranged_images[image_id]['figure_size']
                    panel['figure_aspect'] = arranged_images[image_id]['figure_aspect']

        paper_panels.append(panel)

    return total_input_token, total_output_token, paper_panels, figure_arrangement
