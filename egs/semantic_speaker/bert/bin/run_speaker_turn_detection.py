import copy
import logging
import os
import sys
import codecs
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
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
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner",
        metadata={
            "help": "The name of the task (ner, pos...)."
        }
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    label_num: int = field(
        default=2, metadata={"help": "The number of labels in the file"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


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

    # for key in data_files.keys():
    #     logger.info(f"load a local file for {key}: {data_files[key]}")

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
                                     output_columns_name="label"
                                     ):
    def tokenize_and_align_labels(examples):
        initial_inputs_sentences = copy.deepcopy(examples[input_columns_name])
        tokenized_inputs = tokenizer(
            examples[input_columns_name],
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        labels = []
        for i, change_point_list in enumerate(examples[output_columns_name]):
            label = [-100] * max_length
            possible_list = set()
            for j, ch in enumerate(initial_inputs_sentences[i]):
                if j >= max_length:
                    break
                if ch in ['，', '。', '？', '！']:
                    possible_list.add(j)
                    label[j] = 0
            # logger.info(
            #     f"\npossible_list     = {sorted(possible_list)}"
            #     f"\nchange_point_list = {sorted(change_point_list)}"
            # )
            for cp in change_point_list:
                if cp >= max_length:
                    continue
                # assert cp - 1 in possible_list
                label[cp - 1] = 1
            labels.append(label)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    train_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    predict_dataset = None
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    return train_dataset, eval_dataset, predict_dataset


def main_process_with_transformers(model_args, data_args, training_args,
                                   trainer: Trainer,
                                   train_dataset, eval_dataset, predict_dataset,
                                   last_checkpoint=None):
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics(split="eval", metrics=metrics)
        trainer.save_metrics(split="eval", metrics=metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        def merge_split_sign_with_sentence(sentence, sign, label):
            assert len(sign) == len(label)
            sign_length = len(sign)
            result_sentence = ""
            speaker_change_list = []
            for i, ch in enumerate(sentence):
                result_sentence += ch
                if i >= sign_length:
                    continue
                if label[i] not in [0, 1]:
                    continue
                if sign[i] == 1:
                    result_sentence += "|"
                    speaker_change_list.append(i + 1)
            return result_sentence, speaker_change_list

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)
        predict_num = len(predict_dataset)

        outputs_predict_list = []
        for index in range(predict_num):
            utt_id = predict_dataset[index]['utt_id']
            conversation_id = predict_dataset[index]['conversation_id']
            sentence = predict_dataset[index]['sentence']
            predict = predictions[index]
            label = labels[index]

            # predict_list = predict.squeeze().tolist()
            predict_sentence, predict_speaker_change_list = merge_split_sign_with_sentence(sentence, predict, label)
            label_sentence, _ = merge_split_sign_with_sentence(sentence, label, label)

            logger.info(
                f"\npredict = {predict_sentence}"
                f"\nlabel   = {label_sentence}"
            )
            outputs_predict_list.append({
                "utt_id": utt_id,
                "conversation_id": conversation_id,
                "sentence": sentence,
                "predict": predict_speaker_change_list,
                "merged_sentence": predict_sentence
            })

        # Remove ignored index (special tokens)
        label_list = [0, 1]
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics(split="predict", metrics=metrics)
        trainer.save_metrics(split="predict", metrics=metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        output_predict_json_file = os.path.join(training_args.output_dir, "predict_results.json")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(str(prediction)) + "\n")

            with codecs.open(output_predict_json_file, "w", encoding="utf-8") as fw:
                json.dump(outputs_predict_list, fw, indent=2, ensure_ascii=False)


def main():
    model_args, data_args, training_args = get_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = detect_last_checkpoint(model_args, data_args, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = build_raw_datasets(
        model_args, data_args, training_args
    )
    assert data_args.text_column_name is not None
    assert data_args.label_column_name is not None
    text_column_name = data_args.text_column_name
    label_column_name = data_args.label_column_name
    num_labels = data_args.label_num
    label_list = [i for i in range(num_labels)]

    # download the pretrained models and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    logger.info(f"Model = {model}")

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    padding = "max_length" if data_args.pad_to_max_length else False

    train_dataset, eval_dataset, predict_dataset = build_datasets_from_raw_datasets(
        model_args, data_args, training_args,
        raw_datasets, tokenizer,
        padding,
        data_args.max_seq_length,
        input_columns_name=text_column_name,
        output_columns_name=label_column_name,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    # metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        concat_true_predictions = []
        concat_true_labels = []
        for prediction, label in zip(true_predictions, true_labels):
            logger.info(
                f"\nprediction = {prediction}"
                f"\nlabel      = {label}"
            )
            concat_true_predictions.extend(prediction)
            concat_true_labels.extend(label)
        results = {
            "overall_precision": precision_score(concat_true_labels, concat_true_predictions, average=None)[1],
            "overall_recall": recall_score(concat_true_labels, concat_true_predictions, average=None)[1],
            "overall_f1": f1_score(concat_true_labels, concat_true_predictions, average=None)[1],
            "overall_accuracy": accuracy_score(concat_true_labels, concat_true_predictions)
        }
        # results = {
        #     "overall_precision": precision_score(true_labels, true_predictions, average=None)[1],
        #     "overall_recall": recall_score(true_labels, true_predictions, average=None)[1],
        #     "overall_f1": f1_score(true_labels, true_predictions, average=None)[1],
        #     "overall_accuracy": accuracy_score(true_labels, true_predictions)
        # }
        # results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    main_process_with_transformers(
        model_args, data_args, training_args,
        trainer,
        train_dataset, eval_dataset, predict_dataset,
        last_checkpoint=last_checkpoint
    )


if __name__ == "__main__":
    main()
