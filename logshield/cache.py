"""Drain structure cache with conditional original-text span mappings.

A structure match locates fields. The per-cluster tree selects sensitivity from
current Content; a template never fixes its variables' categories.
"""
import re
from collections import defaultdict
from os.path import commonprefix

import numpy as np
from drain3.drain import Drain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

TOKENS = re.compile(r''' "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | \[[^\]]*\] | \w+ | [^\w\s] ''', re.X)


def _tokens(text):
    return [(match.start(), match.end(), match.group()) for match in TOKENS.finditer(text)]


class FieldDrain(Drain):
    def get_content_as_tokens(self, content):
        if not isinstance(content, str):
            return [value for _, _, value in content]
        return [match.group() for match in TOKENS.finditer(content)]

    def fast_match(self, cluster_ids, tokens, sim_th, include_params):
        if sim_th != 1.0 or not include_params:
            return super().fast_match(cluster_ids, tokens, sim_th, include_params)
        plans = getattr(self, '_perfect_matches', None)
        if plans is None:
            plans = self._perfect_matches = {}
        key = tuple(cluster_ids)
        if key not in plans:
            candidates = []
            for identifier in key:
                cluster = self.id_to_cluster.get(identifier)
                if cluster is not None:
                    fixed = tuple((i, value) for i, value in enumerate(cluster.log_template_tokens)
                                  if value != self.param_str)
                    candidates.append((len(cluster.log_template_tokens) - len(fixed), identifier, fixed))
            # Preserve Drain's parameter-count tie break and original tie order.
            plans[key] = sorted(candidates, key=lambda candidate: -candidate[0])
        for _, identifier, fixed in plans[key]:
            if all(tokens[i] == value for i, value in fixed):
                return self.id_to_cluster[identifier]
        return None

    def add_log_message(self, content):
        result = super().add_log_message(content)
        if result[1] != 'none':
            self._perfect_matches = {}
        return result


def _view(text, cluster, tokens=None):
    tokens = _tokens(text) if tokens is None else tokens
    if len(tokens) != len(cluster.log_template_tokens):
        return None
    fixed, parameters, cursor = [], [], 0
    for (start, end, value), pattern in zip(tokens, cluster.log_template_tokens):
        # Keep empty gaps too, so whitespace never changes field numbering.
        parameters.append((cursor, start, text[cursor:start]))
        if pattern == '<*>':
            parameters.append((start, end, value))
        elif pattern == value:
            fixed.append((start, end, value))
        else:
            return None
        cursor = end
    parameters.append((cursor, len(text), text[cursor:]))
    return {'fixed_fragments': fixed, 'parameters': parameters}


def _features(text):
    return re.sub(r'\d+', '0', text)


def _case(text, spans):
    """Keep context while abstracting the values of the current predicted spans."""
    if spans is None:
        return None
    boundaries = sorted({(span['start'], span['end']) for span in spans})
    intervals = []
    for start, end in boundaries:
        # Outer events must preserve the context around their nested fields.
        if any(start <= first and last <= end and (start, end) != (first, last)
               for first, last in boundaries):
            continue
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    parts, cursor = [], 0
    for start, end in intervals:
        parts.append(_features(text[cursor:start]))
        cursor = end
    return (*parts, _features(text[cursor:]))


def _grams(text):
    return {text[i:i + n] for n in (2, 3, 4) for i in range(len(text) - n + 1)}


def _fit_tree(texts, labels, max_depth, min_samples_leaf):
    vectorizer = CountVectorizer(analyzer=_grams, binary=True)
    if any(len(text) >= 2 for text in texts):
        matrix = vectorizer.fit_transform(texts)
        grams = list(vectorizer.get_feature_names_out())
    else:
        matrix, grams = np.zeros((len(texts), 1)), []
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  random_state=42).fit(matrix, labels)
    model = {'tree': tree, 'grams': {int(i): grams[int(i)] for i in set(tree.tree_.feature) if i >= 0}}
    _leaf_values(model)
    return model


def _leaf_values(model):
    if 'nodes' not in model:
        structure = model['tree'].tree_
        model['nodes'] = (tuple(map(int, structure.children_left)),
                          tuple(map(int, structure.children_right)),
                          tuple(model['grams'].get(int(feature)) for feature in structure.feature),
                          tuple(map(float, structure.threshold)))
    if 'leaf_values' not in model:
        tree, values = model['tree'], {}
        structure = tree.tree_
        for node in np.flatnonzero(structure.children_left == structure.children_right):
            node = int(node)
            counts = structure.value[node][0]
            n, p = int(structure.n_node_samples[node]), float(max(counts) / sum(counts))
            values[node] = int(tree.classes_[int(np.argmax(counts))]), (n * p + 1) / (n + 2)
        model['leaf_values'] = values
    return model['leaf_values']


def _leaf(model, text):
    if 'nodes' not in model:
        _leaf_values(model)
    left, right, grams, thresholds = model['nodes']
    node = 0
    while left[node] != right[node]:
        value = int(grams[node] in text)
        node = left[node] if value <= thresholds[node] else right[node]
    return node


def _selection(model, text):
    parent = _leaf(model, _features(text))
    selected = model['refinements'].get(parent)
    node = parent
    if selected is not None:
        node = _leaf(selected, text)
    label, score = _leaf_values(selected or model)[node]
    return (parent, node if selected is not None else None), label, score


def _occurrences(value, literal):
    offsets, offset = [], value.find(literal)
    while offset >= 0:
        offsets.append(offset)
        offset = value.find(literal, offset + 1)
    return offsets


def _position(expression, view):
    kind, index, *detail = expression
    start, end, value = view['fixed_fragments' if kind == 'fixed' else 'parameters'][index]
    if kind == 'enclosed':
        if len(value) < 2 or value[0] + value[-1] not in ('""', "''", '[]'):
            return None
        return (start + 1, end - 1)[detail[0]]
    if kind == 'local':
        direction, literal, *ordinal = detail
        if direction == 'prefix':
            return start + len(literal) if value.startswith(literal) else None
        if direction == 'suffix':
            return end - len(literal) if value.endswith(literal) else None
        if ordinal:
            offsets = _occurrences(value, literal)
            index = ordinal[0] - 1 if ordinal[0] > 0 else ordinal[0]
            if not -len(offsets) <= index < len(offsets):
                return None
            offset = offsets[index]
        else:
            offset = value.find(literal)
            if offset < 0 or value.find(literal, offset + 1) >= 0:
                return None
        return start + offset + (len(literal) if direction == 'after' else 0)
    return start + detail[0] if kind == 'fixed' else (start, end)[detail[0]]


def _coordinate(position, view, anchors=None, category=None, side=None):
    for i, (start, end, _) in enumerate(view['fixed_fragments']):
        if start <= position <= end:
            return ('fixed', i, position - start)
    for i, (start, end, value) in enumerate(view['parameters']):
        if position == start:
            return ('parameter', i, 0)
        if position == end:
            return ('parameter', i, 1)
        if len(value) >= 2 and value[0] + value[-1] in ('""', "''", '[]'):
            if position == start + 1:
                return ('enclosed', i, 0)
            if position == end - 1:
                return ('enclosed', i, 1)
        if start < position < end:
            for expression in (anchors or {}).get((category, side, i), ()):
                if _position(expression, view) == position:
                    return expression
    return None


def _learn_anchors(references):
    boundaries, candidates = defaultdict(list), {}
    for row, view in references:
        text = row['text']
        for span in row['spans']:
            value = text[span['start']:span['end']]
            for side in ('start', 'end'):
                position = span[side]
                if _coordinate(position, view) is not None:
                    continue
                for slot, (start, end, _) in enumerate(view['parameters']):
                    if not start < position < end:
                        continue
                    key = span['category'], side, slot
                    boundaries[key].append((text[start:position][::-1], text[position:end],
                                            value, text, position, view))
    for key, entries in boundaries.items():
        group, slot = candidates.setdefault(key, {}), key[2]

        def add(direction, literal, entry, *ordinal):
            expression = ('local', slot, direction, literal, *ordinal)
            if ordinal or _position(expression, entry[5]) == entry[4]:
                rows, values = group.setdefault(expression, (set(), set()))
                rows.add(entry[3])
                values.add(entry[2])

        for entry in entries:
            if key[1] == 'start':
                add('prefix', entry[0][::-1], entry)
            else:
                add('suffix', entry[1], entry)
        context_index, direction = (0, 'after') if key[1] == 'start' else (1, 'before')
        ordered = sorted(entries, key=lambda entry: entry[context_index])
        first, offset_cache = 0, {}
        while first < len(ordered):
            last = first + 1
            while last < len(ordered) and ordered[last][3] == ordered[first][3]:
                last += 1
            neighbors = [ordered[i][context_index] for i in (first - 1, last)
                         if 0 <= i < len(ordered)]
            for entry in ordered[first:last]:
                context = entry[context_index]
                # Nearest distinct Content bounds shared outside context.
                limit = max((len(commonprefix((context, other))) for other in neighbors), default=0)
                for width in range(1, limit + 1):
                    literal = context[:width]
                    literal = literal[::-1] if context_index == 0 else literal
                    start, _, value = entry[5]['parameters'][slot]
                    lookup = value, literal
                    if lookup not in offset_cache:
                        offset_cache[lookup] = _occurrences(value, literal)
                    offsets = offset_cache[lookup]
                    target = entry[4] - start - (len(literal) if direction == 'after' else 0)
                    if target not in offsets:
                        continue
                    index = offsets.index(target)
                    for ordinal in (index + 1, index - len(offsets)):
                        add(direction, literal, entry, ordinal)
            first = last
    # Repeated Content does not provide additional evidence for a locator.
    return {key: sorted((expr for expr, (rows, values) in group.items()
                         if len(rows) >= 2),
                        key=lambda expr: (-len(group[expr][1]), -len(group[expr][0]),
                                          len(expr[3]) + (len(str(expr[4])) if len(expr) > 4 else 0),
                                          expr[2], expr[3], expr[4] if len(expr) > 4 else 0))
            for key, group in candidates.items()}


def _mapping(text, spans, view, categories, anchors=None):
    result = set()
    for span in spans:
        start, end, category = span.get('start'), span.get('end'), span.get('category')
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or category not in categories):
            raise ValueError('Training spans must use original-text coordinates/categories')
        first, last = (_coordinate(span[side], view, anchors, category, side) for side in ('start', 'end'))
        if first is None or last is None:
            return None
        result.add((first, last, category))
    return tuple(sorted(result))


def _instantiate(mapping, view, text):
    if mapping is None:
        return None
    result = []
    for first, last, category in mapping:
        start, end = _position(first, view), _position(last, view)
        if start is None or end is None or not 0 <= start < end <= len(text):
            return None
        result.append({'start': start, 'end': end, 'category': category})
    return sorted(result, key=lambda value: (value['start'], value['end'], value['category']))


class ParameterCache:
    """Beta-smoothed leaf scores rank outputs; calibration remains independent."""
    def __init__(self, categories=None, *, max_depth=None, min_samples_leaf=1):
        self.categories = list(categories) if categories is not None else None
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.models, self.drain = {}, FieldDrain()
        self.pending = {}
        self.decision_version = 0

    def fit(self, records):
        records = [{'text': row['text'], 'spans': row['spans'], 'reference': True}
                   for row in records]
        if not records:
            raise ValueError('At least one training reference is required')
        if self.categories is None:
            self.categories = sorted({span['category'] for row in records for span in row['spans']})
        self.models, self.drain = {}, FieldDrain()
        self.pending = {}
        self.decision_version = 0
        for row in records:
            if not isinstance(row['text'], str):
                raise TypeError('Training Content must be a string')
            self.drain.add_log_message(row['text'])
        self.references = self._group(records)
        self.seen = {(row['text'], self._label(row['spans'])) for row in records}
        usable = sum(self._fit_cluster(self.drain.id_to_cluster[identifier]) for identifier in self.references)
        self.training = {'rows': len(records), 'templates': len(self.models),
                         'covered_rows': sum(map(len, self.references.values())), 'mappable_rows': usable,
                         'unsupported_rows': len(records) - usable}
        return self

    @staticmethod
    def _label(spans):
        return tuple(sorted({(s['start'], s['end'], s['category']) for s in spans}))

    def _group(self, records):
        groups = defaultdict(list)
        for row in records:
            cluster = self.drain.match(row['text'], full_search_strategy='fallback')
            if cluster is not None:
                groups[cluster.cluster_id].append(row)
        return groups

    def _fit_cluster(self, cluster):
        identifier = cluster.cluster_id
        references = [(row, view) for row in self.references[identifier]
                      if (view := _view(row['text'], cluster)) is not None]
        self.models.pop(identifier, None)
        if not references:
            return 0
        anchors, texts, labels, outputs, usable = _learn_anchors(references), [], [], [], 0
        for row, view in references:
            mapping = _mapping(row['text'], row['spans'], view, self.categories, anchors)
            usable += mapping is not None
            if mapping not in outputs:
                outputs.append(mapping)
            texts.append(row['text'])
            labels.append(outputs.index(mapping))
        normalized = [_features(text) for text in texts]
        model = _fit_tree(normalized, labels, self.max_depth, self.min_samples_leaf)
        leaves = defaultdict(list)
        for i, text in enumerate(normalized):
            leaves[_leaf(model, text)].append(i)
        # Preserve raw-value distinctions only where normalized training labels conflict.
        model['refinements'] = {
            leaf: _fit_tree([texts[i] for i in indices], [labels[i] for i in indices],
                            self.max_depth, self.min_samples_leaf)
            for leaf, indices in leaves.items() if len({labels[i] for i in indices}) > 1}
        confirmed = defaultdict(set)
        for i, (row, _) in enumerate(references):
            if row['reference'] and not row.get('confirmed'):
                continue
            path, _, _ = _selection(model, row['text'])
            confirmed[(_case(row['text'], row['spans']), path)].add(outputs[labels[i]])
        model.update(outputs=outputs, references=len(texts),
                     confirmed={key: next(iter(values)) for key, values in confirmed.items()
                                if len(values) == 1 and None not in values})
        self.models[identifier] = model
        return usable

    def _observed(self, text, spans):
        if not isinstance(text, str) or spans is None:
            raise ValueError('An observed Content and resolved spans are required')
        spans = [{'start': start, 'end': end, 'category': category}
                 for start, end, category in self._label(spans)]
        if any(type(s['start']) is not int or type(s['end']) is not int
               or not 0 <= s['start'] < s['end'] <= len(text) or s['category'] not in self.categories for s in spans):
            raise ValueError('Observed spans must use original-text coordinates/categories')
        return spans

    def observe_many(self, records):
        old, added, touched, changed_structure = self.references, [], set(), False
        for row in records:
            text, spans = row['text'], self._observed(row['text'], row['spans'])
            key = text, self._label(spans)
            if key in self.seen:
                continue
            cluster, changed = self.drain.add_log_message(text)
            changed_structure |= changed != 'none'
            touched.add(cluster.cluster_id)
            added.append({'text': text, 'spans': spans, 'reference': False})
            self.seen.add(key)
        if not added:
            return None
        self.references = self._group([row for group in old.values() for row in group] + added)
        affected = touched | {i for i in old.keys() | self.references.keys()
                              if old[i] != self.references[i]}
        for identifier in affected:
            self._fit_cluster(self.drain.id_to_cluster[identifier])
        self.decision_version = getattr(self, 'decision_version', 0) + 1
        return True if changed_structure else frozenset(affected)

    def observe(self, text, spans):
        """Return refitted IDs (True if structure changed), False to confirm, or None."""
        spans = self._observed(text, spans)
        key = text, self._label(spans)
        candidate = self.predict(text)
        if (candidate['spans'] is not None
                and self._label(candidate['spans']) == key[1]):
            identifier = candidate['template']
            model = self.models[identifier]
            path, label, _ = _selection(model, text)
            confirmation = _case(text, candidate['spans']), path
            if model['confirmed'].get(confirmation) == model['outputs'][label]:
                return None
            model['confirmed'][confirmation] = model['outputs'][label]
            if key not in self.seen:
                self.references[identifier].append({'text': text, 'spans': spans, 'reference': False})
                self.seen.add(key)
            else:
                for row in self.references[identifier]:
                    if row['text'] == text and self._label(row['spans']) == key[1]:
                        row['confirmed'] = True
            self.decision_version = getattr(self, 'decision_version', 0) + 1
            return False
        if key in self.seen:
            return None
        cluster, changed = self.drain.add_log_message(text)
        old = self.references
        row = {'text': text, 'spans': spans, 'reference': False}
        if changed != 'none':
            self.references = self._group([r for group in old.values() for r in group] + [row])
            affected = {i for i in old.keys() | self.references.keys() if old[i] != self.references[i]}
        else:
            target = self.drain.match(text, full_search_strategy='fallback').cluster_id
            self.references[target].append(row)
            affected = {target}
        refitted = affected | {cluster.cluster_id}
        for identifier in refitted:
            self._fit_cluster(self.drain.id_to_cluster[identifier])
        self.seen.add(key)
        self.decision_version = getattr(self, 'decision_version', 0) + 1
        return frozenset(refitted) if changed == 'none' else True

    def predict(self, text, threshold=None):
        if not isinstance(text, str):
            raise TypeError('Content must be a string')
        tokens = _tokens(text)
        cluster = self.drain.match(tokens, full_search_strategy='fallback')
        model = self.models.get(cluster.cluster_id) if cluster is not None else None
        if model is None or (view := _view(text, cluster, tokens)) is None:
            return {'spans': None, 'score': 0.0, 'template': None}
        path, label, score = _selection(model, text)
        spans = _instantiate(model['outputs'][label] if label is not None else None, view, text)
        score = score if spans is not None else 0.0
        if (spans is not None and (threshold is None or score < threshold)
                and model.get('confirmed', {}).get((_case(text, spans), path)) == model['outputs'][label]):
            score = 1.0
        return {'spans': spans, 'score': score,
                'template': cluster.cluster_id}

    def examples(self, text):
        cluster = self.drain.match(text, full_search_strategy='fallback')
        candidates = [row for row in self.references.get(cluster.cluster_id, []) if row['reference']] if cluster else []
        same_template = bool(candidates)
        if not candidates:
            candidates = [row for rows in self.references.values() for row in rows if row['reference']]
        candidates = list({(row['text'], self._label(row['spans'])): row for row in candidates}.values())
        if not candidates:
            return []
        texts = [_features(row['text']) for row in candidates]
        if any(len(value) >= 2 for value in texts):
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
            matrix = vectorizer.fit_transform(texts)
            scores = (matrix @ vectorizer.transform([_features(text)]).T).toarray().ravel()
            ranked = [candidates[i] for i in np.argsort(-scores, kind='stable')]
        else:
            ranked = candidates
        if not same_template:
            return [{key: row[key] for key in ('text', 'spans', 'reference')} for row in ranked[:2]]
        first = ranked[0]
        categories = lambda row: {span['category'] for span in row['spans']}
        opposite = next((row for row in ranked[1:] if bool(row['spans']) != bool(first['spans'])), None)
        different = next((row for row in ranked[1:] if categories(row) != categories(first)), None)
        selected = [first] + ([opposite or different] if opposite is not None or different is not None else [])
        return [{key: row[key] for key in ('text', 'spans', 'reference')} for row in selected]
