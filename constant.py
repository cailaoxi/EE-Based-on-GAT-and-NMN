
ACE_FILES = './English'
ACE_DUMP = './data/ACE05/'
GloVe_file = './glove/glove.6B.100d.txt'


EVENT_TYPE_TO_ID = {'None': 0, 'Be-Born': 1, 'Die': 2, 'Marry': 3, 'Divorce': 4, 'Injure': 5, 'Transfer-Ownership': 6, 'Transfer-Money': 7, 'Transport': 8, 'Start-Org': 9, 'End-Org': 10, 'Declare-Bankruptcy': 11, 'Merge-Org': 12, 'Attack': 13, 'Demonstrate': 14, 'Meet': 15, 'Phone-Write': 16, 'Start-Position': 17, 'End-Position': 18, 'Nominate': 19, 'Elect': 20, 'Arrest-Jail': 21, 'Release-Parole': 22, 'Charge-Indict': 23, 'Trial-Hearing': 24, 'Sue': 25, 'Convict': 26, 'Sentence': 27, 'Fine': 28, 'Execute': 29, 'Extradite': 30, 'Acquit': 31, 'Pardon': 32, 'Appeal': 33}

ROLE_TO_ID = {'None': 0, 'Person': 1, 'Place': 2, 'Buyer': 3, 'Seller': 4, 'Beneficiary': 5, 'Price': 6, 'Artifact': 7, 'Origin': 8, 'Destination': 9, 'Giver': 10, 'Recipient': 11, 'Money': 12, 'Org': 13, 'Agent': 14, 'Victim': 15, 'Instrument': 16, 'Entity': 17, 'Attacker': 18, 'Target': 19, 'Defendant': 20, 'Adjudicator': 21, 'Prosecutor': 22, 'Plaintiff': 23, 'Crime': 24, 'Position': 25, 'Sentence': 26, 'Vehicle': 27, 'Time-Within': 28, 'Time-Starting': 29, 'Time-Ending': 30, 'Time-Before': 31, 'Time-After': 32, 'Time-Holds': 33, 'Time-At-Beginning': 34, 'Time-At-End': 35}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6,
             'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22,
             '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

corenlp_path = './stanford-corenlp-full-2018-10-05'

INF = 1e8

#general hyperparams
embedding_dim = 100
posi_embedding_dim = 50
event_type_embedding_dim = 5
cut_len = None

#trigger hyperparameters
t_filters = 200
t_batch_size = 30
t_lr = 0.0002 #0.00085达咩 0.00015ok子
t_epoch = 5
t_keepprob = 0.5
t_bias_lambda = 1

#GAT hypers
pos_dim = 50
ner_dim = 50
hidden_dim  = 100

Watt_dim = 100
s_dim = 100

leaky_alpha = 0.2
graph_dim = 150

K=3
#DMCNN&HMEAE argument hyperparameters
a_filters = 300
a_lr = 0.00074
a_keepprob = 0.5
a_epoch = 20
a_batch_size = 30
a_u_c_dim = 900
a_W_a_dim = 900

# Module Design
                                               #Role              Module
module_design     =       [[0,0,0,0,0,0,0,1],  #NA                 NA
                           [1,0,0,0,0,0,0,0],  #Person             Person
                           [0,1,0,0,0,0,0,0],  #Place              Place
                           [1,0,1,0,0,0,0,0],  #Buyer              Person, Org
                           [1,0,1,0,0,0,0,0],  #Seller             Person, Org
                           [1,0,1,0,0,0,0,0],  #Beneficiary        Person, Org
                           [0,0,0,0,1,0,0,0],  #Price              Good
                           [0,0,0,0,0,0,1,0],  #Artifact           Entity
                           [0,1,0,0,0,0,0,0],  #Origin             Place
                           [0,1,0,0,0,0,0,0],  #Destination        Place
                           [1,0,1,0,0,0,0,0],  #Giver              Person, Org
                           [1,0,1,0,0,0,0,0],  #Recipient          Person, Org
                           [0,0,0,0,1,0,0,0],  #Money              Good
                           [0,0,1,0,0,0,0,0],  #Org                Org
                           [1,1,1,0,0,0,0,0],  #Agent              Person, Place, Org
                           [1,0,0,0,0,0,0,0],  #Victim             Person
                           [0,0,0,0,1,0,0,0],  #Instrument         Good
                           [0,0,0,0,0,0,1,0],  #Entity             Entity
                           [1,1,1,0,1,0,0,0],  #Attacker           Person, Org, Place, Good
                           [1,1,1,0,1,0,0,0],  #Target             Person, Org, Place, Good
                           [1,0,1,0,0,0,0,0],  #Defendent          Person, Org
                           [1,0,1,0,0,0,0,0],  #Adjudicator        Person, Org
                           [1,0,1,0,0,0,0,0],  #Prosecutor         Person, Org
                           [1,0,1,0,0,0,0,0],  #Plantiff           Person, Org
                           [0,0,0,0,0,1,0,0],  #Crime              Behavior
                           [0,1,0,0,0,0,0,0],  #Position           Place
                           [0,0,0,0,0,1,0,0],  #Sentence           Behavior
                           [0,0,0,0,1,0,0,0],  #Vehicle            Goods
                           [0,0,0,1,0,0,0,0],  #Time-Within        Time
                           [0,0,0,1,0,0,0,0],  #Time-Starting      Time
                           [0,0,0,1,0,0,0,0],  #Time-Ending        Time
                           [0,0,0,1,0,0,0,0],  #Time-Before        Time
                           [0,0,0,1,0,0,0,0],  #Time-After         Time
                           [0,0,0,1,0,0,0,0],  #Time-Holds         Time
                           [0,0,0,1,0,0,0,0],  #Time-At-Beginning  Time
                           [0,0,0,1,0,0,0,0]]  #Time-At-End        Time

module_num = len(module_design[0])
type_num = len(module_design)
module_weight = [282258, 689, 1121, 104, 45, 29, 12, 729, 190, 564, 135, 151, 88, 124, 427, 649, 302, 876, 683, 508, 377,
                     102, 27, 83, 260, 140, 78, 85, 849, 61, 24, 30, 27, 78, 20, 16]