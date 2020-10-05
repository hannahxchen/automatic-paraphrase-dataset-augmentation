import os
import csv
import sys
import copy
import json
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_examples(self, filename, load_type="train"):
        """Gets a collection of `InputExample`s for the dataset."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
        
    @classmethod
    def _read_txt(cls, input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line == '\n':
                    continue
                
                lines.append(line.split("\t"))
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file) as f:
            return json.load(f)
   
    
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_examples(self, filename, load_type="train"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(filename), load_type)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        # prevent duplicate sentences
        s = []
            
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                # if i % 2 == 0:
                #     text_a = line[0]
                #     text_b = line[1]
                # else:
                #     text_a = line[1]
                #     text_b = line[0]

                text_a = line[0]
                text_b = line[1]
                    
                label = line[2]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                
            except IndexError:
                continue
                
        return examples

    
class SickProcessor(DataProcessor):
    """Processor for the SICK data set."""

    def get_examples(self, filename, load_type='train'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(filename), load_type)

    def get_labels(self):
        """See base class."""
        return ["entails", "neutral", "contradicts"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev, and, test sets."""
        examples = []
            
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:            
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                
            except IndexError:
                continue
                
        return examples
    
    
class DnliProcessor(DataProcessor):
    """Processor for the DNLI data set."""

    def get_examples(self, filename, load_type="train", datatype='json'):
        """See base class."""
        if datatype == 'json':
            return self._create_examples_from_json(
                self._read_json(filename), load_type)
        elif datatype == 'tsv':
            return self._create_examples_from_tsv(
                self._read_tsv(filename), load_type)
        else:
            raise Exception('datatype error')

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples_from_json(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for d in data:
            guid = d['id']
            text_a = d['sentence1']
            text_b = d['sentence2']
            label = d['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                
        return examples

    def _create_examples_from_tsv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:            
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                
            except IndexError:
                continue
                
        return examples
    

    
processors = {
    "qqp": QqpProcessor,
    "sick": SickProcessor,
    "dnli": DnliProcessor
}
    
    
def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 label_list=None,
                                 output_mode=None,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    return features 