import sys, re, os, nltk
from nltk import word_tokenize
from nltk.corpus import words, wordnet
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from textstat.textstat import textstat
from nltk.stem.wordnet import WordNetLemmatizer
import language_check

tokenizer = RegexpTokenizer(r'\w+')
output_list = []
wordset = set(words.words())
lmtzr = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
tool = language_check.LanguageTool('en-US')

relevant_trigrams = [('IN', 'DT', 'NN'), ('VB', 'JJ', 'NNS'), ('VBZ', 'JJ', 'NNS'), ('PRP', 'TO', 'VB'), 
('VB', 'DT', 'NN'), ('DT', 'JJ', 'NNS'), ('CC', 'JJ', 'NN'), ('CC', 'PRP', 'VBZ'), ('.', 'NN', 'VBP'), 
('TO', 'VB', 'IN'), ('DT', 'NN', 'VBP'), ('DT', 'NNS', 'VBP'), ('PRP$', 'NN', 'CC'), ('NN', '.', 'WRB'), 
('JJ', 'NN', 'CC'), ('VBP', 'RB', 'JJ'), ('TO', 'VB', 'JJR'), ('VB', 'NN', 'IN'), ('VBN', 'TO', 'VB'), 
('JJ', 'IN', 'PRP'), ('NNS', '.', 'IN'), ('PRP', 'VBP', 'JJ'), ('IN', 'NN', '.'), ('RB', ',', 'NN'), 
(',', 'DT', 'NNS'), ('NN', 'CC', 'TO'), ('NNS', 'RB', 'VBP'), ('JJ', 'NNS', ','), ('NN', '.', 'IN'), 
(',', 'IN', 'NNS'), ('NN', 'IN', 'NNS'), ('VBZ', 'DT', 'JJ'), ('JJ', 'VBP', 'RB'), ('VBP', 'DT', 'NN'), 
(',', 'PRP', 'RB'), ('JJ', 'NN', 'IN'), ('NNS', 'VBP', 'JJ'), ('VBZ', 'DT', 'NN'), ('MD', 'VB', 'PRP'), 
('DT', 'NNS', '.'), ('IN', 'PRP', 'VBZ'), ('NN', 'TO', 'VB'), ('VBZ', 'VBN', 'TO'), ('NN', '.', 'NNS'), 
('PRP', 'MD', 'VB'), ('PRP', 'VBD', 'DT'), ('IN', 'PRP', 'TO'), ('VB', 'IN', 'IN'), (',', 'IN', 'PRP'), 
('RB', 'VB', 'NNS'), ('VBP', 'RB', 'VB'), ('RB', 'VB', 'NN'), ('.', 'DT', 'NN'), ('DT', 'NN', 'VBZ'), 
('NN', 'IN', 'DT'), ('VBP', 'DT', 'JJ'), ('VBG', 'JJ', 'TO'), ('NNS', 'VBP', 'NN'), ('NNS', ',', 'NN'), 
('NNS', 'IN', 'NN'), ('NN', 'IN', 'NN'), ('VBP', 'JJR', 'NN'), ('VBD', 'TO', 'VB'), ('VB', 'JJ', 'VBZ'), 
('JJR', 'NN', 'CC'), ('NNS', '.', 'RB'), ('NNS', 'WDT', 'VBP'), ('VBG', 'PRP', 'TO'), ('NN', ',', 'JJ'), 
('VBP', 'JJ', 'NN'), ('NN', ',', 'CD'), ('IN', 'PRP', 'RB'), ('MD', 'VB', 'TO'), (',', 'PRP', 'MD'), 
('IN', 'CD', 'NNS'), (',', 'NN', 'VBP'), ('DT', 'NN', 'IN'), ('PRP', 'VBD', 'IN'), ('JJ', 'NN', 'MD'), 
('NN', 'IN', 'PRP$'), ('TO', 'NNS', 'MD'), ('NN', '.', 'DT'), ('NNS', 'JJ', 'IN'), ('NNS', 'IN', 'DT'), 
('.', 'DT', 'JJ'), ('PRP', 'NNS', ','), ('NNS', ',', 'EX'), ('IN', 'NN', ','), ('NN', 'MD', 'VB'), 
('PRP', 'RB', '.'), ('NNS', 'MD', 'VB'), ('JJ', '.', 'RB'), (',', 'PRP', 'VBD'), ('NNS', 'TO', 'VB'), 
('NN', 'VBZ', 'PRP'), ('NNS', 'IN', 'PRP'), ('VBD', 'DT', 'JJ'), ('WP', 'MD', 'VB'), ('IN', 'VBG', 'CC'), 
('IN', 'NN', 'IN'), ('JJ', ',', 'VBG'), ('MD', 'VB', 'NNS'), ('CC', 'WRB', 'PRP'), ('DT', 'NNS', 'IN'), 
('WRB', 'PRP', 'VBP'), ('DT', 'NNS', 'VBD'), ('RB', 'VB', 'IN'), ('NN', 'DT', 'NN'), ('DT', 'NN', '.'), 
('CC', 'VBG', 'IN'), ('VBP', 'JJR', 'NNS'), ('.', 'IN', 'IN'), ('IN', 'PRP$', 'NN'), ('VB', 'PRP$', 'NN'), 
('.', 'DT', 'MD'), ('RB', ',', 'PRP'), ('IN', 'DT', 'JJ'), ('.', 'IN', 'NN'), (',', 'PRP', 'VBP')]

relevant_trigram_set = set(relevant_trigrams)

transition_words = [('and', 'then'), ('besides'), ('equally', 'important'), ('finally'), ('further'), 
('furthermore'), ('nor'), ('next'), ('lastly'), ('what\'s', 'more'), ('moreover'), ('in', 'addition'), 
('first'), ('second'), ('third'), ('fourth'), ('whereas'), ('yet'), ('on', 'the', 'other', 'hand'), ('however'), 
('nevertheless'), ('on', 'the', 'contrary'), ('by', 'comparison'), ('compared', 'to'), ('up', 'against'), 
('balanced', 'against'), ('vis', 'a', 'vis'), ('although'), ('conversely'), ('meanwhile'), ('after', 'all'), 
('in', 'contrast'), ('although', 'this', 'may', 'be', 'true'), ('because'), ('since'), ('for', 'the', 'same', 'reason'), 
('obviously'), ('evidently'), ('indeed'), ('in', 'fact'), ('in', 'any', 'case'), ('that', 'is'), ('still'), ('in', 'spite', 'of'), 
('despite'), ('of', 'course'), ('once', 'in', 'a', 'while'), ('sometimes'), ('immediately'), ('thereafter'), ('soon'), 
('after', 'a', 'few', 'hours'), ('then'), ('later'), ('previously'), ('formerly'), ('in', 'brief'), ('as', 'I', 'have', 'said'), 
('as', 'I', 'have', 'noted'), ('as', 'has', 'been', 'noted'), ('definitely'), ('extremely'), ('obviously'), ('absolutely'), 
('positively'), ('naturally'), ('surprisingly'), ('always'), ('forever'), ('perennially'), ('eternally'), ('never'), 
('emphatically'), ('unquestionably'), ('without', 'a', 'doubt'), ('certainly'), ('undeniably'), ('without', 'reservation'), 
('following', 'this'), ('at', 'this', 'time'), ('now'), ('at', 'this', 'point'), ('afterward'), ('subsequently'), ('consequently'), 
('previously'), ('before', 'this'), ('simultaneously'), ('concurrently'), ('thus'), ('therefore'), ('hence'), ('for', 'example'), 
('for', 'instance'), ('in', 'this', 'case'), ('in', 'another', 'case'), ('on', 'this', 'occasion'), ('in', 'this', 'situation'), 
('take', 'the', 'case', 'of'), ('to', 'demonstrate'), ('to', 'illustrate'), ('as', 'an', 'illustration'), ('on', 'the', 'whole'), 
('summing', 'up'), ('to', 'conclude'), ('in', 'conclusion'), ('as', 'I', 'have', 'shown'), ('as', 'I', 'have', 'said'), 
('accordingly'), ('as', 'a', 'result')]

transitions_set = set(transition_words)

### Import your corpus here in whatever format you have it

with open(os.path.expanduser("filename.tsv"),encoding='utf-8') as input_file:
    for line in input_file:
        # Preprocessing
        line = line.strip()
        Sex, Age, Language, Level, ID, Score, Essay = line.split('\t')
        
        essay = Essay
        
        # With punctuation, not lowered
        tokens = word_tokenize(essay)
        tagged = nltk.pos_tag(tokens)
        num_sents = len(sent_tokenize(essay))
        
        # With punctuation, lowered
        essay_low = Essay.strip().lower()
        tokens_low = word_tokenize(essay_low)
        tagged_low = nltk.pos_tag(tokens_low)
        
        # Without punctuation, not lowered
        tokens_np = tokenizer.tokenize(essay)
        num_tokens = len(tokens_np)
        
        # Without punctuation, lowered
        tokens_low_np = tokenizer.tokenize(essay_low)
        types = set(tokens_low_np)
        num_types = len(types)
        
        # Content and function words
        content_tokens = [w for w in tokens_np if w not in stopwords]
        content_types = [w for w in types if w not in stopwords]
        
        function_tokens = [w for w in tokens_np if w in stopwords]
        function_types = [w for w in types if w in stopwords]
       
        # Word frequency ranking extractors 
        rankings=[]
        rank_file = open(os.path.expanduser("~/Desktop/word_rank.tsv"),encoding='utf-8')
        for line in rank_file:
            rank, token, pos, freq, disp = line.split()
            for word in content_types:
                if word == token:
                    rankings.append(int(rank))
        rank_total = sum(rankings)
        try:
            rank_avg = round(rank_total/len(rankings),4)
        except ZeroDivisionError:
            rank_avg = 0
        
        # Length feature extractor
        len_words = []
        for word in tokens_np:
            len_words.append(len(word))
            avg_len_word = round(sum(len_words) / num_tokens, 4)
        
        # Sentence density feature extractor
        sent_density = round(num_sents / num_tokens * 100, 2)
        
        # Lexical diversity feature extractor
        ttr = round(num_types / num_tokens * 100, 2)
        
        # English words feature extractor
        english_types = []
        for word in types:
            if word in wordset:
                english_types.append(word)
        english_usage = len(english_types)
        
        # Percent of relevant trigrams in essay
        a, b = zip(*tagged)
        trigram_set = set(nltk.trigrams(b))
        found_trigrams = relevant_trigram_set & trigram_set
        pct_rel_trigrams = round(len(found_trigrams) / len(relevant_trigram_set) * 100, 2)
        
        found_transitions = transitions_set & types
        pct_transitions = round(len(found_transitions) / len(transitions_set), 4)
        
        for word in found_transitions:
            transition_word = word
        
        matches = tool.check(essay)
        grammar_chk = round(len(matches)/len(tokens_np), 5)
        
        rules =[]
        for match in matches:
            match_list = list(match)
            match_rule = match_list[4]
            rules.append(match_rule)
        for rule in set(rules):
            grammar_error = rule
        
        ## TAACO features
        
        # n_lemma_types
        lemma_types_list = []
        for word in types:
            lemma_types = lmtzr.lemmatize(word)
            lemma_types_list.append(lemma_types)
            bigram_lemma_types = nltk.bigrams(lemma_types_list)
            trigram_lemma_types = nltk.trigrams(lemma_types_list)
        nlemma_types = len(lemma_types_list)
        n_bigram_lemma_types = len(list(bigram_lemma_types))
        n_trigram_lemma_types = len(list(trigram_lemma_types))
        
        # n_lemmas
        lemma_tokens_list = []
        for word in tokens_np:
            lemma_tokens = lmtzr.lemmatize(word)
            lemma_tokens_list.append(lemma_tokens)
            bigram_lemmas = nltk.ngrams(lemma_tokens_list,2)
            trigram_lemmas = nltk.ngrams(lemma_tokens_list,3)
        nlemmas = len(lemma_tokens_list)
        n_bigram_lemmas = len(list(bigram_lemmas))
        n_trigram_lemmas = len(list(trigram_lemmas))
        
        # content_words
        ncontent_tokens = len(content_tokens)
        ncontent_types = len(content_types)
        
        try:
            content_ttr = round(ncontent_types/ncontent_tokens,4)
        except ZeroDivisionError:
            content_ttr = 1
        
        # function_words
        nfunction_tokens = len(function_tokens)
        nfunction_types = len(function_types)
        
        try:
            function_ttr = round(nfunction_types/nfunction_tokens,4)
        except ZeroDivisionError:
            function_ttr = 1
            
        # noun_ttr
        nouns = []
        for word, tag in tagged:
            if re.search(r'\b(NN(S|P|PS))\b', tag):
                nouns.append(word)
        try:
            noun_ttr = round(len(set(nouns))/len(nouns),4)
        except ZeroDivisionError:
            noun_ttr = 0
            
        # determiners
        det = len(re.findall(r'\b(DT)\b', str(tagged), flags=re.I))
        determiners = round(det/len(tokens_np), 5)
            
        # conjunctions
        conj = len(re.findall(r'\b(and|but)\W+(CC)\b', str(tagged), flags=re.I))
        conjunctions = round(conj/len(tokens_np), 5)
        
        # pronouns
        prn = len(re.findall(r'\b(he|she|it|his|hers|him|her|they|them|their)\b', str(tokens), flags=re.I))
        prn_density = round(prn/len(tokens_np), 5)
        try:
            prn_noun_ratio = round(prn/len(nouns), 2)
        except ZeroDivisionError:
            prn_noun_ratio = 0
            
        ## Readability features
        
        num_syllab = textstat.syllable_count(essay)
        avg_len_sent = textstat.avg_sentence_length(essay)
#         avg_sent_per_word = textstat.avg_sentence_per_word(essay)
#         num_polysyllab = textstat.polysyllabcount(essay)
        num_chars = textstat.char_count(essay, ignore_spaces=True)
#         avg_syllab_per_word = textstat.avg_syllables_per_word(essay)
       
        fre = textstat.flesch_reading_ease(essay)
        fkg = textstat.flesch_kincaid_grade(essay)
        cli = textstat.coleman_liau_index(essay)
        ari = textstat.automated_readability_index(essay)
        dcrs = textstat.dale_chall_readability_score(essay)
        dw = textstat.difficult_words(essay)
        lwf = textstat.linsear_write_formula(essay)
        gf = textstat.gunning_fog(essay)
        
        ## Stages of negation (features to improve validity for AES in ELL contexts)
        
        stage1a = len(re.findall(r'\b(no)\W+(DT)\W{6}\w+\W+(VB|VBG|VBD|VBZ|VBP|VBN|MD)\b', str(tagged), flags=re.I))
        stage1b = len(re.findall(r'\b(NN(S|P|PS)|PRP|VB(G|N)|MD)\W{6}(not)\W+(RB)\W+\w+\W+(VB(G|N))\b', str(tagged_low), flags=re.I))
        stage1c = len(re.findall(r'\b(not)\W+(RB)\W{6}\w+\W+(VBD|VBZ|VBP|MD)\b', str(tagged_low), flags=re.I))
        stage2a = len(re.findall(r'\b((do)\W+\w+\W+(not|n\'t)\W+(RB)|(dont)\W{6}\w+\W+)\W{6}\w+\W+(VBG|VBD|VBZ|VBN|MD)\b', str(tagged), flags=re.I))
        stage2b = len(re.findall(r'\b(he|she|it|him|her)\W+\w+\W{6}((do)\W+\w+\W+(not|n\'t)\W+(RB)|(dont))\b', str(tagged), flags=re.I))
        stage2c = len(re.findall(r'\b(i|you|we|they)\W+\w+\W{6}((does)\W+\w+\W{6}(not|n\'t)|doesnt)\b', str(tagged), flags=re.I))
        stage3a = len(re.findall(r'\b(d(o|oes|id)|ha(ve|s|d)|be|a(m|re)|is|w(as|ere))\W+\w+\W{6}(((do)\W+\w+\W+(not|n\'t))\W+(RB)|(dont))\b', str(tagged), flags=re.I))
        stage3b = len(re.findall(r'\b(ha(ve|s|d)|be|a(m|re)|is|w(as|ere))\W+\w+\W{6}(not|n\'t)\b', str(tagged), flags=re.I))
        stage3c = len(re.findall(r'\b(MD)\W+((do)\W+\w+\W+(not|n\'t)|dont|not|n\'t)\b', str(tagged), flags=re.I))
        stage4a = len(re.findall(r'\b(i|you|we|they)\W+\w+\W{6}((do|did)\W+\w+\W+(not|n\'t)|dont|didnt)\W+(RB|VBP)\W+\w+\W+(VB)\b', str(tagged), flags=re.I))
        stage4b = len(re.findall(r'\b(i|you|we|they)\W+\w+\W{6}((did)\W+\w+\W+(not|n\'t)|didnt)\W+\w+\W{6}\w+\W+(VBD)\b', str(tagged), flags=re.I))
        stage4c = len(re.findall(r'\b(he|she|it)\W+(\w+|NNP)\W{6}((does)\W+\w+\W{6}(not|n\'t)|doesnt)\W+\w+\W{6}\w+\W+(VB|VBZ)\b', str(tagged), flags=re.I))
        
        # Original stages   
        stage1 = stage1a+stage1b+stage1c
        stage2 = stage2a+stage2b+stage2c
        stage3 = stage3a+stage3b+stage3c
        stage4 = stage4a+stage4b+stage4c
           
        neg_usage = stage1+stage2+stage3+stage4
        
        try: 
            s1a = round(stage1a*100/neg_usage,2)
            s1b = round(stage1b*100/neg_usage,2)
            s1c = round(stage1c*100/neg_usage,2)
            s2a = round(stage2a*100/neg_usage,2)
            s2b = round(stage2b*100/neg_usage,2)
            s2c = round(stage2c*100/neg_usage,2)
            s3a = round(stage3a*100/neg_usage,2)
            s3b = round(stage3b*100/neg_usage,2)
            s3c = round(stage3c*100/neg_usage,2)
            s4a = round(stage4a*100/neg_usage,2)
            s4b = round(stage4b*100/neg_usage,2)
            s4c = round(stage4c*100/neg_usage,2)
        
        except ZeroDivisionError:
            s1a=0 
            s1b=0
            s1c=0 
            s2a=0
            s2b=0 
            s2c=0
            s3a=0 
            s3b=0 
            s3c=0 
            s4a=0 
            s4b=0 
            s4c=0
        try:
            s1 = s1a+s1b+s1c
            s2 = s2a+s2b+s2c
            s3 = s3a+s3b+s3c
            s4 = s4a+s4b+s4c
        
        except ZeroDivisionError:
            s1=0.0
            s2=0.0
            s3=0.0
            s4=0.0
            
        # New stages
        
        stage1_new = stage1a+stage2b+stage2a
        stage2_new = stage1b+stage4a
        stage3_new = stage3c+stage3b+stage4b
        
        neg_usage_new = stage1_new+stage2_new+stage3_new
               
        try:
            
            s1a_new = round(stage1a*100/neg_usage_new,2)
            s1b_new = round(stage1b*100/neg_usage_new,2)
            s1c_new = round(stage1c*100/neg_usage_new,2)
            s2a_new = round(stage2a*100/neg_usage_new,2)
            s2b_new = round(stage2b*100/neg_usage_new,2)
            s3b_new = round(stage3b*100/neg_usage_new,2)
            s3c_new = round(stage3c*100/neg_usage_new,2)
            s4a_new = round(stage4a*100/neg_usage_new,2)
            s4b_new = round(stage4b*100/neg_usage_new,2)
            s4c_new = round(stage4c*100/neg_usage_new,2)
         
        except ZeroDivisionError:
            s1a_new=0 
            s1b_new=0
            s1c_new=0 
            s2a_new=0
            s2b_new=0 
            s3b_new=0 
            s3c_new=0 
            s4a_new=0 
            s4b_new=0 
            s4c_new=0
            
        try:
            s1_new = s1a_new+s2b_new+s2a_new
            s2_new = s1b_new+s4a_new
            s3_new = s3c_new+s3b_new+s4b_new
            
        except ZeroDivisionError:
            s1_new=0.0
            s2_new=0.0
            s3_new=0.0

# If you add to the features extracted, add them to the output list

        output_list.append([Sex, Age, Language, Level, ID, Score, rank_total, rank_avg, pct_transitions, transition_word, 
                            grammar_chk, grammar_error, determiners, conjunctions, prn_density, prn_noun_ratio, 
                            n_trigram_lemma_types, n_bigram_lemma_types, nlemma_types, nlemmas, n_bigram_lemmas, n_trigram_lemmas, 
                            ncontent_tokens, ncontent_types, content_ttr, nfunction_tokens, nfunction_types, function_ttr, noun_ttr,
                            neg_usage, s1a, s1b, s1c, s2a, s2b, s2c, s3a, s3b, s3c, s4a, s4b, s4c, s1, s2, s3, s4, 
                            neg_usage_new, s1a_new, s1b_new, s1c_new, s2a_new, s2b_new, s3b_new, s3c_new, s4a_new, s4b_new, s4c_new, s1_new, s2_new, s3_new, 
                            fre, fkg, cli, ari, dcrs, dw, lwf, gf, num_tokens, num_types, 
                            avg_len_word, num_sents, avg_len_sent, num_syllab, num_chars, sent_density, ttr, english_usage, pct_rel_trigrams, Essay])
        
# Then iterate over output_list and write it to an output file.

with open('output_file.tsv', 'w', encoding='utf-8') as output_file:
    # then write the column names...
    print('Sex', 'Age', 'Language', 'Level', 'ID', 'Score', 'rank_total', 'rank_avg', 'pct_transitions', 'transition_word', 
          'grammar_chk', 'grammar_error', 'determiners', 'conjunctions', 'prn_density', 'prn_noun_ratio', 
          'n_trigram_lemma_types', 'n_bigram_lemma_types', 'nlemma_types', 'nlemmas', 'n_bigram_lemmas', 'n_trigram_lemmas', 
          'ncontent_tokens', 'ncontent_types', 'content_ttr', 'nfunction_tokens', 'nfunction_types', 'function_ttr', 'noun_ttr',
          'neg_usage', 's1a', 's1b', 's1c', 's2a', 's2b', 's2c', 's3a', 's3b', 's3c', 's4a', 's4b', 's4c','s1', 's2', 's3', 's4', 
          'neg_usage_new', 's1a_new', 's1b_new', 's1c_new', 's2a_new', 's2b_new', 's3b_new', 's3c_new', 's4a_new', 's4b_new', 's4c_new','s1_new', 's2_new', 's3_new', 
          'fre', 'fkg', 'cli', 'ari', 'dcrs', 'dw', 'lwf', 'gf', 'num_tokens', 'num_types', 
          'avg_len_word', 'num_sents', 'avg_len_sent', 'num_syllab', 'num_chars', 'sent_density', 'ttr', 'english_usage', 'pct_rel_trigrams', 'Essay', sep='\t', file=output_file)
    # then print each record...
    for line in output_list:
        print(*line, sep='\t', file=output_file)
