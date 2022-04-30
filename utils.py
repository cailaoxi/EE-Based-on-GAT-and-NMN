import os
import constant
from xml.dom.minidom import parse
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import re
import random
import json
import numpy as np
import copy


class StanfordCoreNLPv2(StanfordCoreNLP):
    def __init__(self, path):
        super(StanfordCoreNLPv2, self).__init__(path)

    def sent_tokenize(self, sentence):
        r_dict = self._request('ssplit,tokenize', sentence)
        tokens = [[token['originalText'] for token in s['tokens']] for s in r_dict['sentences']]
        spans = [[(token['characterOffsetBegin'], token['characterOffsetEnd']) for token in s['tokens']] for s in
                 r_dict['sentences']]
        return tokens, spans


class Extractor():
    def __init__(self):
        self.dirs = ['bc', 'bn', 'cts', 'nw', 'un', 'wl']
        self.split_tags = {'bc': ["</SPEAKER>", '</TURN>', '<HEADLINE>', '</HEADLINE>'],
                           'bn': ["<TURN>", "</TURN>"],
                           "cts": ["</SPEAKER>", "</TURN>"],
                           'nw': ['<TEXT>', '</TEXT>', '<HEADLINE>', '</HEADLINE>'],
                           'un': ['</SUBJECT>', '<HEADLINE>', '</HEADLINE>', '<SUBJECT>', '</POST>', '<QUOTE'],
                           'wl': ['</POSTDATE>', '</POST>', '<HEADLINE>', '</HEADLINE>', '<TEXT>', '</TEXT>']}
        self.Events = []
        self.None_events = []
        self.Entities = []

    def find_index(self, offsets, offset):  # offsets [) offset []
        idx_start = -1
        idx_end = -1
        for j, _offset in enumerate(offsets):
            if idx_start == -1 and _offset[0] <= offset[0] and _offset[1] > offset[0]:
                idx_start = j
            if idx_end == -1 and _offset[0] <= offset[1] and _offset[1] > offset[1]:
                idx_end = j
                break
        assert idx_start != -1 and idx_end != -1
        return idx_start, idx_end

    def sentence_distillation(self, sents, offsets, direct_offsets, dir):
        mark_split_tag = self.split_tags[dir]

        new_sents = []
        new_offsets = []
        new_direct_offsets = []

        if dir == 'cst':
            sents = sents[1:]
            offsets = offsets[1:]
            direct_offsets = direct_offsets[1:]

        for i, sent in enumerate(sents):
            offset_per_sentence = offsets[i]
            direct_offset_per_sentence = direct_offsets[i]
            select = True

            start_posi = 0
            for j, token in enumerate(sent):
                if bool(sum([token.startswith(e) for e in mark_split_tag])):
                    subsent = sent[start_posi:j]
                    suboffset = offset_per_sentence[start_posi:j]
                    subdirectoffset = direct_offset_per_sentence[start_posi:j]
                    if select and len(subsent) > 0:
                        assert (0, 0) not in suboffset
                        new_sents.append(subsent)
                        new_offsets.append(suboffset)
                        new_direct_offsets.append(subdirectoffset)
                    start_posi = j + 1
                    select = True
                elif token.startswith('<'):
                    select = False

            subsent = sent[start_posi:]
            suboffset = offset_per_sentence[start_posi:]
            subdirectoffset = direct_offset_per_sentence[start_posi:]
            if select and len(subsent) > 0:
                assert (0, 0) not in suboffset
                new_sents.append(subsent)
                new_offsets.append(suboffset)
                new_direct_offsets.append(subdirectoffset)
        return new_sents, new_offsets, new_direct_offsets

    def correct_offsets(self, sents, offsets):
        new_offsets = []
        minus = 0
        for i, offsets_per_sentence in enumerate(offsets):
            sentence = sents[i]
            new_offsets_per_sentence = []
            for j, offset in enumerate(offsets_per_sentence):
                if sentence[j].startswith('<'):
                    new_offsets_per_sentence.append((0, 0))
                    minus += len(sentence[j])

                else:
                    new_offsets_per_sentence.append((offset[0] - minus, offset[1] - minus))
            new_offsets.append(new_offsets_per_sentence)
        return sents, new_offsets

    def Files_Extract(self):
        self.event_files = {}
        self.source_files = {}
        self.amp_files = []
        for dir in self.dirs:
            path = constant.ACE_FILES + '/' + dir + '/timex2norm'
            files = os.listdir(path)
            self.event_files[dir] = [file for file in files if file.endswith('.apf.xml')]
            self.source_files[dir] = [file for file in files if file.endswith('.sgm')]
            for file in self.source_files[dir]:
                with open(path + '/' + file, 'r') as f:
                    text = f.read()
                if '&amp;' in text:
                    self.amp_files.append(file[:-3])

        srclen = 0
        evtlen = 0
        for dir in self.dirs:
            srclen += len(self.source_files[dir])
            evtlen += len(self.event_files[dir])
        assert evtlen == srclen
        assert evtlen == 599

    def Entity_Extract(self):
        for dir in self.dirs:
            path = constant.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.event_files[dir]
            for file in files:
                DOMtree = parse(path + "/" + file)
                collection = DOMtree.documentElement
                mention_tags = ['entity_mention', 'value_mention', 'timex2_mention']
                for mention_tag in mention_tags:
                    mention = collection.getElementsByTagName(mention_tag)
                    for sample in mention:
                        start = int(
                            sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].getAttribute(
                                "START"))
                        end = int(
                            sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].getAttribute(
                                "END"))
                        name = str(
                            sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].childNodes[
                                0].data)
                        entity_info = (name, start, end, file, dir)
                        self.Entities.append(entity_info)
        self.Entities = list(set(self.Entities))
        self.Entities = [{'name': e[0], 'start': e[1], 'end': e[2], 'file': e[3], 'dir': e[4], 'role': 'None'} for e in
                         self.Entities]

    def Event_Extract(self):
        nlp = StanfordCoreNLPv2(constant.corenlp_path)
        offsets2idx = {}
        for dir in self.dirs:
            path = constant.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.event_files[dir]
            for file in files:
                DOMtree = parse(path + "/" + file)
                collection = DOMtree.documentElement
                events = collection.getElementsByTagName("event")
                Entities = [e for e in self.Entities if e['dir'] == dir and e['file'] == file]
                for event in events:
                    event_type = str(event.getAttribute("SUBTYPE"))
                    event_mentions = event.getElementsByTagName("event_mention")
                    for event_mention in event_mentions:
                        event_info = event_mention.getElementsByTagName("ldc_scope")[0].getElementsByTagName("charseq")[
                            0]
                        sent = str(event_info.childNodes[0].data)
                        start = int(event_info.getAttribute("START"))
                        end = int(event_info.getAttribute("END"))

                        trigger_info = event_mention.getElementsByTagName("anchor")[0].getElementsByTagName("charseq")[
                            0]
                        trigger = str(trigger_info.childNodes[0].data)
                        trigger_start = int(trigger_info.getAttribute("START"))
                        trigger_end = int(trigger_info.getAttribute("END"))

                        entities = [copy.deepcopy(e) for e in Entities if e['start'] >= start and e['end'] <= end]

                        map_entity = {(e['start'], e['end']): i for i, e in enumerate(entities)}

                        arguments = event_mention.getElementsByTagName("event_mention_argument")
                        for argument in arguments:
                            role = str(argument.getAttribute("ROLE"))
                            argument_info = argument.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[
                                0]
                            argument_name = str(argument_info.childNodes[0].data)
                            argument_start = int(argument_info.getAttribute("START"))
                            argument_end = int(argument_info.getAttribute("END"))
                            assert (argument_start, argument_end) in map_entity
                            entity_id = map_entity[(argument_start, argument_end)]
                            assert argument_name == entities[entity_id]['name']
                            entities[entity_id]['role'] = role
                        tokens, offsets = nlp.word_tokenize(sent, True)
                        dependency_parsing = nlp.dependency_parse(sent)
                        pos_tags = [e[1] for e in nlp.pos_tag(sent)]
                        ner_tags = [e[1] for e in nlp.ner(sent)]

                        plus = 0
                        for j, token in enumerate(tokens):
                            st = offsets[j][0] + plus
                            if file[:-7] in self.amp_files:
                                plus += 4 * token.count('&')
                            ed = offsets[j][1] + plus
                            offsets[j] = (st, ed)

                        tokens_offsets = [(e[0] + start, e[1] - 1 + start) for e in offsets]
                        find_offsets = [(e[0] + start, e[1] + start) for e in offsets]
                        trigger_s, trigger_e = self.find_index(find_offsets, (trigger_start, trigger_end))
                        trigger_offsets = tokens_offsets[trigger_s:trigger_e + 1]
                        trigger_tokens = tokens[trigger_s:trigger_e + 1]
                        _entities = []
                        for e in entities:
                            idx_start, idx_end = self.find_index(find_offsets, (e['start'], e['end']))
                            entity_tokens = tokens[idx_start:idx_end + 1]
                            entity_offsets = tokens_offsets[idx_start:idx_end + 1]
                            entity_start = entity_offsets[0][0]
                            entity_end = entity_offsets[-1][1]
                            entity_info = {'tokens': entity_tokens,
                                           'offsets': entity_offsets,
                                           'start': entity_start,
                                           'end': entity_end,
                                           'idx_start': idx_start,
                                           'idx_end': idx_end,
                                           'role': e['role']
                                           }
                            _entities.append(entity_info)
                        event_summary = {"tokens": tokens,
                                         'offsets': tokens_offsets,
                                         "event_type": [event_type],
                                         "start": start,
                                         "end": end,
                                         "trigger_tokens": [trigger_tokens],
                                         "trigger_start": [trigger_s],
                                         "trigger_end": [trigger_e],
                                         'trigger_offsets': [trigger_offsets],
                                         "entities": [_entities],
                                         "dependency_parsing": dependency_parsing,
                                         'pos_tags': pos_tags,
                                         'ner_tags': ner_tags,
                                         'file': file[:-8],
                                         'dir': dir}
                        offsets_join = str(event_summary['start']) + '_' + str(event_summary['end']) + "_" + \
                                       event_summary['file'] + "_" + event_summary['dir']

                        if offsets_join in offsets2idx:
                            event_idx = offsets2idx[offsets_join]
                            self.Events[event_idx]['event_type'].extend(event_summary['event_type'])
                            self.Events[event_idx]['trigger_tokens'].extend(event_summary['trigger_tokens'])
                            self.Events[event_idx]['trigger_start'].extend(event_summary['trigger_start'])
                            self.Events[event_idx]['trigger_end'].extend(event_summary['trigger_end'])
                            self.Events[event_idx]['trigger_offsets'].extend(event_summary['trigger_offsets'])
                            self.Events[event_idx]['entities'].extend(event_summary['entities'])
                        else:
                            offsets2idx[offsets_join] = len(self.Events)
                            self.Events.append(event_summary)
        nlp.close()

    def None_event_Extract(self):
        nlp = StanfordCoreNLPv2(constant.corenlp_path)
        for dir in self.dirs:
            path = constant.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.source_files[dir]
            for file in files:
                event_in_this_file = [(e['start'], e['end']) for e in self.Events if
                                      e['file'] == file[:-4] and e['dir'] == dir]
                Entities = [(e['start'], e['end']) for e in self.Entities if
                            e['dir'] == dir and e['file'][:-7] == file[:-3]]
                with open(path + '/' + file, 'r') as f:
                    text = f.read()
                sents, direct_offsets = nlp.sent_tokenize(text)
                sents, offsets = self.correct_offsets(sents, direct_offsets)
                sents, offsets, direct_offsets = self.sentence_distillation(sents, offsets, direct_offsets, dir)

                new_sents = []
                new_offsets = []
                new_direct_offsets = []
                for j, sent in enumerate(sents):
                    offset = offsets[j]
                    select = True
                    for event in event_in_this_file:
                        if (offset[0][0] >= event[0] and offset[0][0] <= event[1]) or \
                                (offset[-1][1] - 1 >= event[0] and offset[-1][1] - 1 <= event[1]):
                            select = False
                            break
                    if select:
                        new_sents.append(sent)
                        new_offsets.append(offset)
                        new_direct_offsets.append(direct_offsets[j])

                sents = new_sents
                offsets = new_offsets
                direct_offsets = new_direct_offsets

                for i, sent in enumerate(sents):
                    offset = offsets[i]
                    direct_offset = direct_offsets[i]
                    tokens = sent
                    start = offset[0][0]
                    end = offset[-1][1] - 1
                    tokens_offset = [(e[0], e[1] - 1) for e in offset]
                    tokens_direct_offset = [(e[0], e[1] - 1) for e in direct_offset]
                    event_type = 'None'
                    trigger_tokens = []
                    trigger_offsets = []
                    trigger_start = -1
                    trigger_end = -1
                    entities = []

                    origin_sent = text[tokens_direct_offset[0][0]:tokens_direct_offset[-1][-1] + 1]
                    space_link_tokens = ' '.join(tokens)

                    if len(nlp.word_tokenize(origin_sent)) == len(tokens):
                        parse_sent = origin_sent
                    elif len(nlp.word_tokenize(space_link_tokens)) == len(tokens):
                        parse_sent = space_link_tokens
                    else:
                        # print('\n')
                        # print(origin_sent)
                        # print(' '.join(tokens))
                        # print(tokens)
                        # print(tokens_offset)
                        # print(nlp.word_tokenize(' '.join(tokens)))
                        # print(nlp.word_tokenize(origin_sent))
                        continue  # about 10 sents. won't affect results.

                    dependency_parsing = nlp.dependency_parse(parse_sent)
                    pos_tags = [e[1] for e in nlp.pos_tag(parse_sent)]
                    ner_tags = [e[1] for e in nlp.ner(parse_sent)]

                    _entities = [e for e in Entities if e[0] >= start and e[1] <= end]
                    for e in _entities:
                        idx_start, idx_end = self.find_index(offset, e)
                        entity_info = {'token': sent[idx_start:idx_end + 1],
                                       'role': 'None',
                                       'offsets': [(e[0], e[1] - 1) for e in offset[idx_start:idx_end + 1]],
                                       'start': offset[idx_start][0],
                                       'end': offset[idx_end][1] - 1,
                                       'idx_start': idx_start,
                                       'idx_end': idx_end}
                        entities.append(entity_info)
                    none_event_summary = {
                        'tokens': tokens,
                        'start': start,
                        'end': end,
                        'offsets': tokens_offset,
                        'event_type': event_type,
                        'trigger_tokens': trigger_tokens,
                        'trigger_start': trigger_start,
                        'trigger_end': trigger_end,
                        'trigger_offsets': trigger_offsets,
                        'entities': entities,
                        'pos_tags': pos_tags,
                        'ner_tags': ner_tags,
                        "dependency_parsing": dependency_parsing,
                        'file': file[:-4],
                        'dir': dir
                    }
                    self.None_events.append(none_event_summary)
        nlp.close()

    def process(self):
        Events = []
        for event in self.Events:
            for i in range(len(event['trigger_start'])):
                _event = {
                    'tokens': event['tokens'],
                    'start': event['start'],
                    'end': event['end'],
                    'offsets': event['offsets'],
                    'event_type': event['event_type'][i],
                    'trigger_tokens': event['trigger_tokens'][i],
                    'trigger_start': event['trigger_start'][i],
                    'trigger_end': event['trigger_end'][i],
                    'trigger_offsets': event['trigger_offsets'][i],
                    'entities': event['entities'][i],
                    "dependency_parsing": event['dependency_parsing'],
                    'pos_tags': event['pos_tags'],
                    'ner_tags': event['ner_tags'],
                    'file': event['file'],
                    'dir': event['dir']
                }
                Events.append(_event)

            _entities = []
            for entity in event['entities'][0]:
                add_entity = copy.deepcopy(entity)
                add_entity['role'] = 'None'
                _entities.append(add_entity)
            for i in range(len(event['tokens'])):
                if i in event['trigger_start']:
                    continue
                _event = {
                    'tokens': event['tokens'],
                    'start': event['start'],
                    'end': event['end'],
                    'offsets': event['offsets'],
                    'event_type': 'None',
                    'trigger_tokens': [event['tokens'][i]],
                    'trigger_start': i,
                    'trigger_end': i,
                    'trigger_offsets': [event['offsets'][i]],
                    'entities': _entities,
                    "dependency_parsing": event['dependency_parsing'],
                    'pos_tags': event['pos_tags'],
                    'ner_tags': event['ner_tags'],
                    'file': event['file'],
                    'dir': event['dir']
                }
                Events.append(_event)
        self.Events = Events

        None_events = []
        for none_event in self.None_events:
            for i in range(len(none_event['tokens'])):
                _none_event = {
                    'tokens': none_event['tokens'],
                    'start': none_event['start'],
                    'end': none_event['end'],
                    'offsets': none_event['offsets'],
                    'event_type': 'None',
                    'trigger_tokens': [none_event['tokens'][i]],
                    'trigger_start': i,
                    'trigger_end': i,
                    'trigger_offsets': [none_event['offsets'][i]],
                    'entities': none_event['entities'],
                    "dependency_parsing": none_event['dependency_parsing'],
                    'pos_tags': none_event['pos_tags'],
                    'ner_tags': none_event['ner_tags'],
                    'file': none_event['file'],
                    'dir': none_event['dir']
                }
                None_events.append(_none_event)
        self.None_events = None_events

    def Extract(self):
        if os.path.exists(constant.ACE_DUMP + '/train.json'):
            print('--Already Exists Files--')
            return

        self.Files_Extract()
        print('--File Extraction Finish--')
        self.Entity_Extract()
        print('--Entity Extraction Finish--')
        self.Event_Extract()
        print('--Event Mention Extraction Finish--')
        self.None_event_Extract()
        print('--Negetive Mention Extraction Finish--')
        self.process()
        print('--Preprocess Data Finish--')

        nw = self.source_files['nw']
        random.shuffle(nw)
        random.shuffle(nw)
        other_files = [file for dir in self.dirs for file in self.source_files[dir] if dir != 'nw'] + nw[40:]
        random.shuffle(other_files)
        random.shuffle(other_files)

        test_files = nw[:40]
        dev_files = other_files[:30]
        train_files = other_files[30:]

        test_set = [instance for instance in self.Events if instance['file'] + '.sgm' in test_files] + [instance for
                                                                                                        instance in
                                                                                                        self.None_events
                                                                                                        if instance[
                                                                                                            'file'] + ".sgm" in test_files]
        dev_set = [instance for instance in self.Events if instance['file'] + '.sgm' in dev_files] + [instance for
                                                                                                      instance in
                                                                                                      self.None_events
                                                                                                      if instance[
                                                                                                          'file'] + ".sgm" in dev_files]
        train_set = [instance for instance in self.Events if instance['file'] + '.sgm' in train_files] + [instance for
                                                                                                          instance in
                                                                                                          self.None_events
                                                                                                          if instance[
                                                                                                              'file'] + ".sgm" in train_files]

        with open(constant.ACE_DUMP + '/train.json', 'w') as f:
            json.dump(train_set, f)
        with open(constant.ACE_DUMP + '/dev.json', 'w') as f:
            json.dump(dev_set, f)
        with open(constant.ACE_DUMP + '/test.json', 'w') as f:
            json.dump(test_set, f)

class StanfordCoreNLPv2(StanfordCoreNLP):
    def __init__(self,path):
        super(StanfordCoreNLPv2,self).__init__(path)
    def sent_tokenize(self,sentence):
        r_dict = self._request('ssplit,tokenize', sentence)
        tokens = [[token['originalText'] for token in s['tokens']] for s in r_dict['sentences']]
        spans = [[(token['characterOffsetBegin'], token['characterOffsetEnd']) for token in s['tokens']] for s in r_dict['sentences'] ]
        return tokens, spans

class Loader():
    def __init__(self,cut_len):
        self.train_path = constant.ACE_DUMP+'/train.json'
        self.dev_path = constant.ACE_DUMP+'/dev.json'
        self.test_path = constant.ACE_DUMP+'/test.json'
        self.glove_path = constant.GloVe_file
        self.cut_len = cut_len

    def load_embedding(self):
        word2idx = {}
        wordemb = []
        with open(self.glove_path,'r',encoding='utf-8') as f:
            for line in f:
                splt = line.split()
                assert len(splt)==constant.embedding_dim+1
                vector = list(map(float, splt[-constant.embedding_dim:]))
                word = splt[0]
                word2idx[word] = len(word2idx)+2
                wordemb.append(vector)
        return word2idx,np.asarray(wordemb,np.float32)

    def get_maxlen(self):
        if self.cut_len!=None:
            self.maxlen = self.cut_len
            return self.maxlen
        paths = [self.train_path,self.dev_path,self.test_path]
        maxlens = []
        for path in paths:
            with open(path,'r') as f:
                data = json.load(f)
            _maxlen = max([len(d['tokens']) for d in data])
            maxlens.append(_maxlen)
        self.maxlen = max(maxlens)
        return self.maxlen
    
    def get_max_argument_len(self):
        paths = [self.train_path,self.dev_path,self.test_path]
        maxlens = []
        for path in paths:
            with open(path,'r') as f:
                data = json.load(f)
            for instance in data:
                if len(instance['entities'])==0:
                    continue
                _maxlen = max([entity['idx_end']+1-entity['idx_start'] for entity in instance['entities']])
                maxlens.append(_maxlen)
        self.max_argument_len = max(maxlens)
        return self.max_argument_len

    def get_positions(self,start_idx,sent_len,maxlen):
        return list(range(maxlen-start_idx, maxlen)) + [maxlen]  + \
               list(range(maxlen+1, maxlen+sent_len - start_idx))+[0]*(maxlen-sent_len)

    def get_word(self,tokens,word2idx,pad_lenth):
        idx = []
        for word in tokens:
            if word.lower() in word2idx:
                idx.append(word2idx[word.lower()])
            else:
                idx.append(1)
        idx += [0]*(pad_lenth-len(idx))
        return idx

    def get_trigger_mask(self,posi,sent_len,maxlen,direction):
        assert direction in ['left','right']
        mask = [0.]*maxlen
        if direction=='left':
            mask[:posi] = [1.]*posi
        else:
            mask[posi:sent_len] = [1.]*(sent_len-posi)
        return mask
    
    def get_argument_mask(self,posi1,posi2,sent_len,maxlen,direction):
        assert direction in ['left','right','mid']
        mask = [0.]*maxlen
        posi_min = min(posi1,posi2)
        posi_max = max(posi1,posi2)
        if direction=='left':
            mask[:posi_min] = [1.]*posi_min
        elif direction=='mid':
            mask[posi_min:posi_max] = [1.]*(posi_max-posi_min)
        else:
            mask[posi_max:sent_len] = [1.]*(sent_len-posi_max)
        return mask

    def load_one_trigger(self,path,maxlen,word2idx):
        trigger_posis,sents,trigger_maskls,trigger_maskrs,event_types,trigger_lexical= [], [], [], [], [], []
        with open(path,'r') as f:
            data = json.load(f)
        indices_s,pos,ner = [],[],[]
        trigger_idxs = []
        for instance in data:
            tokens = instance['tokens'][:maxlen]
            event_type = instance['event_type']
            trigger_posi = instance['trigger_start']
            if trigger_posi > maxlen - 1:
                continue
            ner_tags = [constant.NER_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['ner_tags']][
                       :maxlen] + [0] * (maxlen - len(instance['ner_tags']))
            pos_tags = [constant.POS_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['pos_tags']][
                       :maxlen] + [0] * (maxlen - len(instance['pos_tags']))
            ner.append(ner_tags)
            pos.append(pos_tags)

            words = self.get_word(tokens, word2idx, maxlen)
            dependency_parsing = instance['dependency_parsing']

            start_word = 0
            current_max = 0
            indices = []
            for edge in dependency_parsing:
                if edge[0] == "ROOT":
                    start_word = max(start_word, current_max)
                else:
                    if edge[1] - 1 + start_word > maxlen - 1 or edge[2] - 1 + start_word > maxlen - 1:
                        continue
                    indices.append([edge[1] - 1 + start_word, edge[2] - 1 + start_word])
                current_max = max([current_max, edge[1] + start_word, edge[2] + start_word])
            indices_s.append(indices)

            trigger_posis.append(self.get_positions(trigger_posi, len(tokens), maxlen))
            trigger_idxs.append(trigger_posi)
            sents.append(words)
            trigger_maskls.append(self.get_trigger_mask(trigger_posi, len(tokens), maxlen, 'left'))
            trigger_maskrs.append(self.get_trigger_mask(trigger_posi, len(tokens), maxlen, 'right'))
            event_types.append(constant.EVENT_TYPE_TO_ID[event_type])

            _trigger_lexical = []
            if trigger_posi == 0:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi - 1])

            _trigger_lexical.append(words[trigger_posi])

            if trigger_posi == len(tokens) - 1:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi + 1])

            trigger_lexical.append(_trigger_lexical)

        return (np.array(trigger_posis, np.int32), np.array(sents, np.int32), np.array(trigger_maskls, np.int32), \
                np.array(trigger_maskrs, np.int32), np.array(event_types, np.int32),np.array(trigger_lexical, np.int32), \
                np.array(pos, np.int32), np.array(ner, np.int32), np.array(trigger_idxs, np.int32)), indices_s

    def load_trigger(self):
        print('--Loading Trigger--')
        word2idx,self.wordemb = self.load_embedding()
        maxlen = self.get_maxlen()
        paths = [self.train_path, self.dev_path, self.test_path]
        results = []
        for path in paths:
            result = self.load_one_trigger(path,maxlen,word2idx)
            results.append(result)
        return results

    def load_one_argument(self,path,maxlen,word2idx,max_argument_len,flag,dataset ='train'):
        sents, event_types, roles, maskl, maskm, maskr, trigger_lexical, argument_lexical,\
        trigger_maskl, trigger_maskr,trigger_posis,argument_posis = [],[],[],[],[],[],[],[],[],[],[],[]
        indices_s, pos, ner = [], [], []
        trigger_idxs = []
        with open(path,'r') as f:
            data = json.load(f)
        for instance in data:
            if dataset =='train':
                if instance['event_type']=='None':
                    continue
            if len(instance['entities'])==0:
                continue
            tokens = instance['tokens'][:maxlen]
            words = self.get_word(tokens,word2idx,maxlen)
            trigger_posi = instance['trigger_start']
            event_type = instance['event_type']
            dependency_parsing = instance['dependency_parsing']
            start_word = 0
            current_max = 0
            indices = []

            _trigger_lexical = []
            if trigger_posi==0:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi-1])

            _trigger_lexical.append(words[trigger_posi])

            if trigger_posi==len(tokens)-1:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi+1])

            for entity in instance['entities']:
                role = entity['role']
                entity_start = entity['idx_start']
                entity_end = entity['idx_end']

                sents.append(words)
                event_types.append(constant.EVENT_TYPE_TO_ID[event_type])
                roles.append(constant.ROLE_TO_ID[role])
                if constant.ROLE_TO_ID[role] == 1:
                    flag[0] += 1
                elif constant.ROLE_TO_ID[role] == 2:
                    flag[1] += 1
                elif constant.ROLE_TO_ID[role] == 3:
                    flag[2] += 1
                elif constant.ROLE_TO_ID[role] == 4:
                    flag[3] += 1
                elif constant.ROLE_TO_ID[role] == 5:
                    flag[4] += 1
                elif constant.ROLE_TO_ID[role] == 6:
                    flag[5] += 1
                elif constant.ROLE_TO_ID[role] == 7:
                    flag[6] += 1
                elif constant.ROLE_TO_ID[role] == 8:
                    flag[7] += 1
                elif constant.ROLE_TO_ID[role] == 9:
                    flag[8] += 1
                elif constant.ROLE_TO_ID[role] == 10:
                    flag[9] += 1
                elif constant.ROLE_TO_ID[role] == 11:
                    flag[10] += 1
                elif constant.ROLE_TO_ID[role] == 12:
                    flag[11] += 1
                elif constant.ROLE_TO_ID[role] == 13:
                    flag[12] += 1
                elif constant.ROLE_TO_ID[role] == 14:
                    flag[13] += 1
                elif constant.ROLE_TO_ID[role] == 15:
                    flag[14] += 1
                elif constant.ROLE_TO_ID[role] == 16:
                    flag[15] += 1
                elif constant.ROLE_TO_ID[role] == 17:
                    flag[16] += 1
                elif constant.ROLE_TO_ID[role] == 18:
                    flag[17] += 1
                elif constant.ROLE_TO_ID[role] == 19:
                    flag[18] += 1
                elif constant.ROLE_TO_ID[role] == 20:
                    flag[19] += 1
                elif constant.ROLE_TO_ID[role] == 21:
                    flag[20] += 1
                elif constant.ROLE_TO_ID[role] == 22:
                    flag[21] += 1
                elif constant.ROLE_TO_ID[role] == 23:
                    flag[22] += 1
                elif constant.ROLE_TO_ID[role] == 24:
                    flag[23] += 1
                elif constant.ROLE_TO_ID[role] == 25:
                    flag[24] += 1
                elif constant.ROLE_TO_ID[role] == 26:
                    flag[25] += 1
                elif constant.ROLE_TO_ID[role] == 27:
                    flag[26] += 1
                elif constant.ROLE_TO_ID[role] == 28:
                    flag[27] += 1
                elif constant.ROLE_TO_ID[role] == 29:
                    flag[28] += 1
                elif constant.ROLE_TO_ID[role] == 30:
                    flag[29] += 1
                elif constant.ROLE_TO_ID[role] == 31:
                    flag[30] += 1
                elif constant.ROLE_TO_ID[role] == 32:
                    flag[31] += 1
                elif constant.ROLE_TO_ID[role] == 33:
                    flag[32] += 1
                elif constant.ROLE_TO_ID[role] == 34:
                    flag[33] += 1
                elif constant.ROLE_TO_ID[role] == 35:
                    flag[34] += 1
                elif constant.ROLE_TO_ID[role] == 0:
                    flag[35] += 1
                trigger_idxs.append(trigger_posi)
                if trigger_posi > maxlen - 1:
                    continue
                ner_tags = [constant.NER_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['ner_tags']][
                           :maxlen] + [0] * (maxlen - len(instance['ner_tags']))
                pos_tags = [constant.POS_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['pos_tags']][
                           :maxlen] + [0] * (maxlen - len(instance['pos_tags']))
                ner.append(ner_tags)
                pos.append(pos_tags)

                for edge in dependency_parsing:
                    if edge[0] == "ROOT":
                        start_word = max(start_word, current_max)
                    else:
                        if edge[1] - 1 + start_word > maxlen - 1 or edge[2] - 1 + start_word > maxlen - 1:
                            continue
                        indices.append([edge[1] - 1 + start_word, edge[2] - 1 + start_word])
                    current_max = max([current_max, edge[1] + start_word, edge[2] + start_word])
                indices_s.append(indices)
                trigger_posis.append(self.get_positions(trigger_posi,len(tokens),maxlen))
                argument_posis.append(self.get_positions(entity_start,len(tokens),maxlen))
                maskl.append(self.get_argument_mask(entity_start,trigger_posi,len(tokens),maxlen,'left'))
                maskm.append(self.get_argument_mask(entity_start,trigger_posi,len(tokens),maxlen,'mid'))
                maskr.append(self.get_argument_mask(entity_start,trigger_posi,len(tokens),maxlen,'right'))
                trigger_lexical.append(_trigger_lexical)
                
                trigger_maskl.append(self.get_trigger_mask(trigger_posi,len(tokens),maxlen,'left'))
                trigger_maskr.append(self.get_trigger_mask(trigger_posi, len(tokens),maxlen, 'right'))
                
                _argument_lexical = []
                if entity_start==0:
                    _argument_lexical.append(0)
                else:
                    _argument_lexical.append(words[entity_start-1])
                
                _argument_lexical.extend(words[entity_start:entity_end+1]+[0]*(max_argument_len-entity_end-1+entity_start))

                if entity_end==len(tokens)-1:
                    _argument_lexical.append(0)
                else:
                    _argument_lexical.append(words[entity_end+1])
                
                argument_lexical.append(_argument_lexical)
        return (np.array(sents,np.int32),np.array(event_types,np.int32),np.array(roles,np.int32),\
               np.array(maskl,np.int32),np.array(maskm,np.int32),np.array(maskr,np.int32),\
               np.array(trigger_lexical,np.int32),np.array(argument_lexical,np.int32),\
               np.array(trigger_maskl,np.int32),np.array(trigger_maskr,np.int32),\
               np.array(trigger_posis,np.int32),np.array(argument_posis,np.int32),\
               np.array(pos, np.int32), np.array(ner, np.int32), np.array(trigger_idxs, np.int32)), indices_s

    def load_argument(self):
        print('--Loading Argument--')
        word2idx,self.wordemb = self.load_embedding()
        maxlen = self.get_maxlen()
        max_argument_len = self.get_max_argument_len()
        results  = []
        flag = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for path,dataset in [(self.train_path,'train'),(self.dev_path,'dev'),(self.test_path,'test')]:
            result = self.load_one_argument(path,maxlen,word2idx,max_argument_len,flag,dataset)
            results.append(result)
        print(flag)
        return results