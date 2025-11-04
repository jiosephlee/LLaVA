import argparse
import torch
import os
import sys
import json
import numpy as np
from tqdm import tqdm

# --- Path Setup (from train.py) ---
# Add LLaVA project root to Python path to allow llava imports
llava_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if llava_project_root not in sys.path:
    sys.path.insert(0, llava_project_root)
    
# Add therapeutic-tuning project root to path to import utils
# This assumes eval_tdc.py is in the same directory as train.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# LLaVA Imports
from llava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.model.language_model.llava_intern import LlavaInternForCausalLM

# TDC Imports (from therapeutic-tuning)
import utils.tdc_utils as tdc_utils
import utils.tdc_data_utils as tdc_data_utils
import utils.tdc_evaluation as tdc_eval
import utils.tdc_prompts as tdc_prompts
import utils.utils as utils


def eval_model(args):
    # Setup logging
    log = utils.setup_logging()
    
    # Model Loading
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    log.info(f"Loading model: {model_name} from {model_path}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    
    # Get the molecule processor from the vision tower
    try:
        mol_processor = model.get_vision_tower().mol_processor
    except AttributeError:
        log.error("Could not find 'mol_processor' on the model's vision tower.")
        log.error("Please ensure you are loading a model checkpoint trained for SMILES data.")
        sys.exit(1)

    # Data Loading (from train.py)
    log.info(f"Loading TDC data for task group: {args.task_group_name}")
    tasks = tdc_data_utils.get_task_list(args.task_group_name, log)
    data_dir = os.path.join(project_root, args.data_dir)
    _, val_dfs, test_dfs = tdc_data_utils.load_multitask_data(
        tasks, log, data_dir=data_dir
    )

    if args.split == 'test':
        eval_dfs = test_dfs
        split_name = "Test"
    else:
        eval_dfs = val_dfs
        split_name = "Validation"

    all_scores = {}
    all_targets = {}
    all_preds = {}
    
    # Set conversation template
    conv = conv_templates[args.conv_mode].copy()
    
    # Start Evaluation Loop (from train.py's get_predictions)
    for task in tasks:
        metric = tdc_utils.TASK_METRICS[task]
        df = eval_dfs[task]
        log.info(f"--- Evaluating task: {task} ({metric}) on {split_name} set ---")
        
        targets, preds = [], []
        positive_token, negative_token = "B", "A" # For MCQ

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference on {task}"):
            # 1. Get data
            smiles_text = row['Drug']
            gt_answer = row['Y']

            # 2. Format conversations (from train.py)
            conversations = tdc_prompts.row_to_conversations_mcq(
                row=row,
                dataset=task,
                split=args.split.lower(),
                prompt_style="txgemma_v3",
                is_intern=True,
            )

            # 3. Preprocess: Add image token (logic from preprocess_multimodal)
            human_turn_idx = -1
            for i, sent in enumerate(conversations):
                if sent['from'].lower() == 'human':
                    human_turn_idx = i
                    break
            
            if human_turn_idx != -1:
                val = conversations[human_turn_idx]['value']
                if DEFAULT_IMAGE_TOKEN not in val:
                    val = DEFAULT_IMAGE_TOKEN + '\n' + val.strip()
                
                # Apply <im_start> and <im_end> if needed
                if model.config.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    val = val.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                
                conversations[human_turn_idx]['value'] = val
            else:
                log.warning(f"Could not find human turn in prompt for row: {row}")

            # 4. Apply Chat Template (from train.py)
            conv.messages = []
            for sentence in conversations:
                role = conv.roles[0] if sentence["from"].lower() == "human" else conv.roles[1]
                conv.append_message(role, sentence["value"])
            prompt_text = conv.get_prompt()

            # 5. Process SMILES input (from train.py)
            smiles = mol_processor(
                smiles_text, 
                padding='max_length', 
                max_length=256, 
                return_tensors="pt"
            )
            # Move smiles tensors to the model's device
            smiles = {k: v.to(model.device) for k, v in smiles.items()}

            # 6. Tokenize text input (from train.py)
            input_ids = tokenizer_image_token(
                prompt_text, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)

            # 7. Run Inference (from train.py)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    smiles=smiles,  # Pass the processed smiles
                    max_new_tokens=1, # We only want 'A' or 'B'
                    use_cache=True,
                    temperature=0.0 # Greedy decoding
                )
            
            # 8. Post-process output (from train.py)
            decoded_output = tokenizer.decode(output_ids[0, -1]).strip()

            if positive_token in decoded_output:
                score = 1.0
            elif negative_token in decoded_output:
                score = 0.0
            else:
                log.warning(f"Unexpected output: '{decoded_output}'. Defaulting to 0.5")
                score = 0.5

            targets.append(gt_answer)
            preds.append(score)

        # 9. Calculate and store task-level metrics
        targets, preds = np.array(targets), np.array(preds)
        task_score = tdc_eval.calculate_metric_score(targets, preds, metric, log)
        
        all_scores[task] = {
            'overall': task_score,
            'covered': None,
            'uncovered': None,
        }
        all_targets[task] = targets.tolist()
        all_preds[task] = preds.tolist()
        log.info(f"Task: {task} | {split_name} Score ({metric}): {task_score:.4f}")

    # 10. Aggregate and Save Results (from train.py)
    log.info("--- Aggregating and Saving TDC Results ---")
    avg_score, std_score = tdc_eval.aggregate_scores(all_scores, 'overall')
    
    eval_results = {
        f'per_task_{args.split}': all_scores, 
        f'avg_{args.split}': avg_score, 
        f'std_{args.split}': std_score,
        'raw_targets': all_targets,
        'raw_preds': all_preds
    }
    log.info(f"Average {split_name} Score: {avg_score:.4f} (+/- {std_score:.4f})")

    results_dir = os.path.join(args.output_dir, "tdc_results")
    hyperparameters = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "task_group_name": args.task_group_name,
        "split": args.split,
        "conv_mode": args.conv_mode,
    }

    if hasattr(tdc_eval, 'save_all_results_llava'):
        tdc_eval.save_all_results_llava(
            results_dir, model_name, args.task_group_name, tasks, 
            eval_results, hyperparameters, log
        )
    else:
        tdc_eval.save_all_results(
            results_dir, model_name, args.task_group_name, tasks, 
            eval_results, hyperparameters, log
        )
    
    log.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the LLaVA model checkpoint.")
    parser.add_argument("--model-base", type=str, default=None, 
                        help="Optional base model path if loading LoRA weights.")
    parser.add_argument("--task-group-name", type=str, required=True, 
                        help="Name of the TDC task group (e.g., 'admet_group').")
    parser.add_argument("--data-dir", type=str, default="data/TDC", 
                        help="Relative path to the TDC data directory.")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to save the evaluation results.")
    parser.add_argument("--conv-mode", type=str, default="intern", 
                        help="Conversation template to use. Should match training.")
    parser.add_argument("--split", type=str, default="test", 
                        choices=["validation", "test"], 
                        help="Data split to evaluate on.")
    
    args = parser.parse_args()
    eval_model(args)