import logging
import os
import sys
import codecs
import json

from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    Trainer,
    default_data_collator,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
        Arguments pretaining to what data we'are going to input our model for training and testing.

    """
    task_name: Optional[str] = field(
        default="Dialogue-Detection for speaker related information",
        metadata={
            "help": f"The Dialogue-Detection Mission"
        }
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": f"The max_seq_length for training, default is 128"
        }
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": f"Whether to pad all samples to `max_seq_length`"
        }
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the train data."
        }
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the validation data."
        }
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the test data."
        }
    )

    overwrite_cache: bool = field(
        default=None,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        }
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": f"For debugging purposes or quicker training, "
                    f"truncate the number of training examples to this value if set."
        }
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": f"For debugging purposes or quicker training, "
                    f"truncate the number of evaluation examples to this value if set."
        }
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": f"For debugging purposes or quicker training, "
                    f"truncate the number of prediction examples to this value if set."
        }
    )

    input_columns_name: Optional[str] = field(
        default="sentence",
        metadata={
            "help": f"The input key for columns name."
        }
    )

    output_columns_name: Optional[str] = field(
        default="label",
        metadata={
            "help": f"The output key for columns name."
        }
    )

    def __post_init__(self):
        train_extension = self.train_file.split(".")[-1]
        validation_extension = self.validation_file.split(".")[-1]
        test_extension = self.test_file.split(".")[-1]

        assert train_extension in ["json", "csv"]
        assert validation_extension in ["json", "csv"]
        assert test_extension in ["json", "csv"]


@dataclass
class ModelArguments:
    """
        Arguments
    """

    model_name_or_path: str = field(
        default=None,  # bert-base-chinese
        metadata={
            "help": f"Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": f"Pretrained config name or path if not the same as model_name"
        }
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models download from huggingface.co"
        }
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        }
    )

    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        }
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": f"Will use the token generated when running `huggingface-cli login` "
                    f"(necessary to use this script with private models)."
        },
    )

    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        }
    )


def detect_last_checkpoint(model_args, data_args, training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                f"Use --overwrite_output_dir to overcome"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
                f"To avoid this behavior, change "
                f"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def build_raw_datasets(model_args, data_args, training_args):
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                    test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    return raw_datasets


def get_args():
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments, TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def build_datasets_from_raw_datasets(model_args, data_args, training_args,
                                     raw_datasets, tokenizer,
                                     padding,
                                     max_length,
                                     input_columns_name="sentence",
                                     output_columns_name="label",
                                     ):
    def map_fn(examples):
        data = examples[input_columns_name]
        results = tokenizer(
            *(data,), padding=padding, max_length=max_length, truncation=True
        )
        data_num = len(data)
        if output_columns_name in examples:
            results['label'] = examples[output_columns_name]
        else:
            results['label'] = [False] * data_num
        return results

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            map_fn,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

    train_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    predict_dataset = None
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    return train_dataset, eval_dataset, predict_dataset


def compute_metrics(p: EvalPrediction):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = p.label_ids

    res = {
        "accuracy": accuracy_score(predictions, labels),
        "f1": f1_score(predictions, labels),
        "precision": precision_score(predictions, labels),
        "recall": recall_score(predictions, labels),
    }
    return res


def main_process_with_transformers(model_args, data_args, training_args,
                                   trainer: Trainer,
                                   train_dataset, eval_dataset, predict_dataset,
                                   last_checkpoint=None):
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_sample = (
            data_args.max_train_sample if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_sample"] = min(max_train_sample, len(train_dataset))

        trainer.save_model()
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info(f"*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples or len(eval_dataset)

        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics(split="eval", metrics=metrics)
        trainer.save_metrics(split="eval", metrics=metrics)

    if training_args.do_predict:
        logger.info(f"*** Predict ***")

        sentence2label = {
            predict_dataset[index]['conversation_id']: predict_dataset[index]['label'] for index, _ in
            enumerate(predict_dataset)
        }
        predict_dataset = predict_dataset.remove_columns(data_args.output_columns_name)
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions_logits = np.copy(predictions)
        predictions = np.argmax(predictions, axis=1)
        predict_num = len(predict_dataset)

        output_predict_simple_results_filename = os.path.join(training_args.output_dir, f"predict_simple_results.txt")
        output_predict_json_file = os.path.join(training_args.output_dir, f"output_predict.json")

        predictions_logits_dict = {}
        output_predict_simple_result_list = []
        conversation2utt = dict()
        output_predict_list = []

        acc_num = 0
        for index in range(predict_num):
            prediction = predictions[index]
            utt_id = predict_dataset[index]['utt_id']
            conversation_id = predict_dataset[index]['conversation_id']
            conversation2utt[conversation_id] = utt_id
            label = sentence2label[conversation_id]
            sentence = predict_dataset[index]['sentence']

            if prediction == label:
                acc_num += 1

            predictions_logits_dict[conversation_id] = predictions_logits[index, ...]
            output_predict_simple_result_list.append((
                conversation_id, sentence, prediction
            ))
            output_predict_list.append({
                'utt_id': utt_id,
                'conversation_id': conversation_id,
                "sentence": sentence,
                "is_dialogue": prediction.item()
            })

        logger.info(f"***** Predict results conversation judge predict_dataset = {len(predict_dataset)}, "
                    f"predictions = {len(predictions)} acc = {acc_num / predict_num} *****")

        with open(output_predict_simple_results_filename, "w") as fw:
            for conversation_id, sentence, prediction in output_predict_simple_result_list:
                fw.write(f"{conversation_id} {prediction} {sentence}\n")

        with codecs.open(output_predict_json_file, 'w', encoding="utf-8") as fw:
            json.dump(output_predict_list, fw, indent=2, ensure_ascii=False)


def main():
    model_args, data_args, training_args = get_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training / Evaluation parameters {training_args}")

    last_checkpoint = detect_last_checkpoint(model_args, data_args, training_args)

    set_seed(training_args.seed)

    raw_datasets = build_raw_datasets(
        model_args, data_args, training_args
    )

    label_list = [0, 1]  # raw_datasets['train'].unique("label")
    label_list.sort()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(label_list),
        fintuneing_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    train_dataset, eval_dataset, predict_dataset = build_datasets_from_raw_datasets(
        model_args, data_args, training_args,
        raw_datasets, tokenizer,
        padding=padding,
        max_length=max_seq_length,
        input_columns_name=data_args.input_columns_name,
        output_columns_name=data_args.output_columns_name,
    )

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    main_process_with_transformers(
        model_args, data_args, training_args,
        trainer,
        train_dataset, eval_dataset, predict_dataset,
        last_checkpoint=last_checkpoint
    )


if __name__ == '__main__':
    main()
