import os
import csv
import json
import copy
import random
import argparse
from itertools import combinations, product

import networkx as nx


train_file = 'data/train.tsv'
dev_file = 'data/dev.tsv'

class InputExample(object):
	def __init__(self, _id, text_a, text_b, label):
		self.id = _id
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


def read_examples(filename):
	examples = []
		
	with open(filename) as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for i, row in enumerate(csv_reader):
			if i == 0:
				continue

			_id = row[0]
			q1 = row[3]
			q2 = row[4]
			label = row[5]

			examples.append(InputExample(_id, q1, q2, label))
			
	return examples


def save_data(output_dir, filename, pairs):
	fieldnames = ['quesiton1', 'question2', 'is_duplicate']
	filename = os.path.join(output_dir, filename)

	with open(filename, 'w') as f:
		csvwriter = csv.writer(f, delimiter='\t')
		csvwriter.writerow(fieldnames)
		csvwriter.writerows([[q1, q2, label] for q1, q2, label in pairs])

	print('File saved to {}'.format(filename))


def find_mislabeled_pairs(graph):
	# Find mislabled edges (non-paraphrase edges within the paraphrase cluster)
	
	mislabeled_edges = []
	for n, attr in graph.nodes(data=True):
		if 'group' not in attr:
			continue
			
		edges = graph.edges(n, data=True)
		
		type_of_edges = set([e[2]['is_duplicate'] for e in edges])
		if len(type_of_edges) == 1:
			continue
			
		neg_edges = [e for e in edges if e[2]['is_duplicate'] == 0]
		
		for u, v, a in neg_edges:
			if 'group' not in graph.nodes[v]:
				continue
				
			if graph.nodes[v]['group'] == attr['group']:
				mislabeled_edges.append((u, v))
	
	
	if len(mislabeled_edges) == 0:
		return None
	
	mislabeled_edges = map(tuple,[sorted(i) for i in mislabeled_edges])
	mislabeled_edges = list(set(mislabeled_edges))
	return mislabeled_edges


def generate_original_flipped(examples):
	g = nx.Graph() #find paraphrase cluster

	for e in examples:
		if e.label == '1':
			g.add_edge(e.text_a, e.text_b)
			
	group_attr = {}

	for i, nodes in enumerate(nx.connected_components(g)):
		group_attr.update({n: {'group': i} for n in nodes})

	G = nx.Graph()

	for e in examples:
		G.add_edge(e.text_a, e.text_b, is_duplicate=int(e.label))
		
	nx.set_node_attributes(G, group_attr)

	subgraphs = []

	for nodes in nx.connected_components(G):
		graph = G.subgraph(nodes)
		type_of_edges = set([e[2]['is_duplicate'] for e in graph.edges(data=True)])
		
		# We only consider the subgraphs with both paraphrases and non-paraphrases edges
		if len(type_of_edges) == 2:
			subgraphs.append(graph)

	mislabeled_pairs = []

	for graph in subgraphs:
		temp = find_mislabeled_pairs(graph)
		if temp:
			mislabeled_pairs.extend(temp)

	new_examples = examples.copy()

	for e in new_examples:
		if (e.text_a, e.text_b) in mislabeled_pairs:
			e.label = '1'
		elif (e.text_b, e.text_a) in mislabeled_pairs:
			e.label = '1'

	return [(e.text_a, e.text_b, e.label) for e in new_examples]


def infer_transitive(examples):
	g = nx.Graph()

	for e in examples:
		if e.label == '1':
			g.add_edge(e.text_a, e.text_b, is_duplicate=e.label)
			
	group_attr = {}
	groups = {}
	paraphrase_pairs = []

	for i, nodes in enumerate(nx.connected_components(g)):
		paraphrase_pairs.extend(list(combinations(nodes, 2)))
		group_attr.update({n: {'group': i} for n in nodes})
		groups[i] = nodes
		   
	nx.set_node_attributes(g, group_attr)

	return g, groups, paraphrase_pairs


def infer_non_paraphrases(graph, groups, examples):
	non_paraphrase_groups = set()
	non_paraphrase_pairs = []

	for e in examples:
		if e.label == '1':
			continue

		if e.text_a not in graph or e.text_b not in graph:
			non_paraphrase_pairs.append((e.text_a, e.text_b))
			continue

		group_1 = graph.nodes[e.text_a]['group']
		group_2 = graph.nodes[e.text_b]['group']
		
		if group_1 == group_2:
			continue

		non_paraphrase_groups.add(tuple(sorted([group_1, group_2])))

	for g1, g2 in non_paraphrase_groups:
		non_paraphrase_pairs.extend(list(product(groups[g1], groups[g2])))

	return non_paraphrase_pairs


def generate_augmented(examples):

	paraphrase_graph, groups, paraphrase_pairs = infer_transitive(examples)
	non_paraphrase_pairs = infer_non_paraphrases(paraphrase_graph, groups, examples)

	for e in examples:
		if e.label == '0':
			paraphrase_graph.add_edge(e.text_a, e.text_b, is_duplicate=e.label)

	mislabeled_pairs = []

	for nodes in nx.connected_components(paraphrase_graph):
		graph = paraphrase_graph.subgraph(nodes)
		type_of_edges = set([e[2]['is_duplicate'] for e in graph.edges(data=True)])
		
		# We only consider the subgraphs with both paraphrases and non-paraphrases
		if len(type_of_edges) != 2:
			continue

		temp = find_mislabeled_pairs(graph)
		if temp:
			mislabeled_pairs.extend(temp)

	for q1, q2 in mislabeled_pairs:
		if (q1, q2) in paraphrase_pairs:
			paraphrase_pairs.remove((q1, q2))
		elif (q2, q1) in paraphrase_pairs:
			paraphrase_pairs.remove((q2, q1))

	non_paraphrase_pairs.extend(mislabeled_pairs)

	paraphrase_pairs = [(q1, q2, '1') for q1, q2 in paraphrase_pairs]
	non_paraphrase_pairs = [(q1, q2, '0') for q1, q2 in non_paraphrase_pairs]

	return paraphrase_pairs + non_paraphrase_pairs


def generate_augmented_flipped(examples):

	paraphrase_graph, groups, paraphrase_pairs = infer_transitive(examples)
	non_paraphrase_pairs = infer_non_paraphrases(paraphrase_graph, groups, examples)

	paraphrase_pairs = [(q1, q2, '1') for q1, q2 in paraphrase_pairs]
	non_paraphrase_pairs = [(q1, q2, '0') for q1, q2 in non_paraphrase_pairs]

	return paraphrase_pairs + non_paraphrase_pairs


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--output_dir",
                        "-o",
                        type=os.path.abspath,
                        default='./data',
                        help="Output directory")
	parser.add_argument("--generate_data",
                        "-d",
                        nargs='+', default=[],
                        required=True,
                        help="Options: [original_flipped | augmented | augmented_flipped]")

	args = parser.parse_args()

	train_examples = read_examples(train_file)
	dev_examples = read_examples(dev_file)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	if 'original_flipped' in args.generate_data:
		orig_flipped_pairs = generate_original_flipped(train_examples)
		save_data(args.output_dir, 'train_orig_flipped.tsv', orig_flipped_pairs)
		orig_flipped_pairs = generate_original_flipped(dev_examples)
		save_data(args.output_dir, 'dev_orig_flipped.tsv', orig_flipped_pairs)

	if 'augmented' in args.generate_data:
		augmented_pairs = generate_augmented(train_examples)
		save_data(args.output_dir, 'train_augmented.tsv', augmented_pairs)
		augmented_pairs = generate_augmented(dev_examples)
		save_data(args.output_dir, 'dev_augmented.tsv', augmented_pairs)

	if 'augmented_flipped':
		augmented_flipped_pairs = generate_augmented_flipped(train_examples)
		save_data(args.output_dir, 'train_augmented_flipped.tsv', augmented_flipped_pairs)
		augmented_flipped_pairs = generate_augmented_flipped(dev_examples)
		save_data(args.output_dir, 'dev_augmented_flipped.tsv', augmented_flipped_pairs)
