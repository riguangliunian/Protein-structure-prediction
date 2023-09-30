##加载下载ProtT5模型
#@title Install requirements. { display-mode: "form" }
# Install requirements
!pip install torch transformers sentencepiece h5py

#@title Set up working directories and download files/checkpoints. { display-mode: "form" }
# Create directory for storing model weights (2.3GB) and example sequences.
# Here we use the encoder-part of ProtT5-XL-U50 in half-precision (fp16) as
# it performed best in our benchmarks (also outperforming ProtBERT-BFD).
# Also download secondary structure prediction checkpoint to show annotation extraction from embeddings
!mkdir protT5 # root directory for storing checkpoints, results etc
!mkdir protT5/protT5_checkpoint # directory holding the ProtT5 checkpoint
!mkdir protT5/sec_struct_checkpoint # directory storing the supervised classifier's checkpoint
!mkdir protT5/output # directory for storing your embeddings & predictions
# Huge kudos to the bio_embeddings team here! We will integrate the new encoder, half-prec ProtT5 checkpoint soon
!wget -nc -P protT5/sec_struct_checkpoint http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt


# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddings
# secondary structure predictions


# whether to retrieve embeddings for each residue in a protein
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True
per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = True
per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = True
sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

#检查GPU能否用
#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

##预测模型
#@title Network architecture for secondary structure prediction. { display-mode: "form" }
# Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet( torch.nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )

        self.dssp8_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )


    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x         = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat   = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat   = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(  x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat
    

##@title Load the checkpoint for secondary structure prediction. { display-mode: "form" }
def load_sec_struct_model():
  checkpoint_dir="./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
  state = torch.load( checkpoint_dir )
  model = ConvNet()
  model.load_state_dict(state['state_dict'])
  model = model.eval()
  model = model.to(device)
  print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

  return model

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


##读取TXT文本
seq_path2="/content/data1.txt"
sec_path2="/content/data2.txt"
def normalize_sequence(seq):
    '''
        Normalize a protein sequence by converting to uppercase,
        replacing non-standard amino acids, and removing gaps.
    '''
    seq = seq.upper()
    seq = seq.replace("-", "")
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    return seq

def read_text(txt_path):
    """
    Reads sequences from a text file.
    Each line in the text file is treated as a separate sequence.

    Args:
        txt_path (str): Path to the input text file.

    Returns:
        list: A list containing the normalized sequences read from the file.
    """
    sequences = []
    with open(txt_path, 'r') as txt_file:
        current_sequence = ''
        for line in txt_file:
            line = line.strip()
            if line:
                current_sequence += line
                sequences.append(normalize_sequence(current_sequence))
                current_sequence = ''

    if current_sequence:
        sequences.append(normalize_sequence(current_sequence))

    return sequences

def convert_predictions_to_dict(predictions):
    '''
    将预测结果列表转换为字典对象。

    参数：
    predictions (list): 包含预测结果的列表。

    返回：
    dict: 包含转换后的字典对象。
    '''
    converted_predictions = {
        str(idx): pred
        for idx, pred in enumerate(predictions)
    }
    return converted_predictions

seq2=read_text(seq_path2)
sec2=read_text(sec_path2)

Y=convert_predictions_to_dict(sec2)

#@title Generate embeddings with Learning Rate and Optimizer { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=1000, max_batch=100, learning_rate=0.001 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d8_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()

                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器，学习率为指定的learning_rate

    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results



##设置对应输出键值
#八维
def convert_predictions_to_letters(predictions):
    class_mapping = {0:"G",1:"H",2:"I",3:"B",4:"E",5:"S",6:"T",7:"C"}
#    class_mapping = {0:"H",1:"E",2:"C"} 三维情况
    converted_predictions = {
        seq_id: [class_mapping[j] for j in yhat]
        for seq_id, yhat in predictions.items()
    }
    return converted_predictions


#加载计算模型
# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
#seqs = read_fasta( seq_path )


# 这里写入您要运行的代码块

# Compute embeddings and/or secondary structure predictions
results = get_embeddings( model, tokenizer, seqs,
                         per_residue, per_protein, sec_struct,learning_rate=0.0001)