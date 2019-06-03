import torch
import torch.nn as nn
class PointerGenerator(nn.Module):
    def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, word_vectors=None, eps=1e-10, coverage=False):
        super(PointerGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.input_len = input_len
        self.output_len = output_len
        self.voc_size = voc_size
        self.teacher_prob = 1.
        self.epsilon = eps
        self.coverage = coverage
        
        if word_vectors is None:
            self.emb_layer = nn.Embedding(voc_size, emb_dim)
        else:
#             self.emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_vectors).float(), freeze=True)
            self.emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_vectors).float(), freeze=False)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)        
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=1, batch_first=True)
        
        self.attention_softmax = nn.Softmax(dim=1)
        
        self.pro_layer = nn.Sequential(
            nn.Linear(hidden_dim*4, voc_size, bias=True),
            nn.Softmax(dim=-1)
        )
        self.pgen_layer = nn.Sequential(
            nn.Linear(4*hidden_dim+emb_dim, 1, bias=True),
            nn.Sigmoid()
        )
        
        self.cov_weight = nn.Parameter(torch.randn(1, dtype=torch.float)/10)
        
    def scheduled_sampling(self, rate):
        self.teacher_prob = 1. - rate
    
    def eval(self):
        self.scheduled_sampling(1.)
        self.training = False
        return self.train(False)
        
    def forward(self, x, ans):
        batch_size =x.shape[0]
        device = x.device
        
        # encoder
        x_emb = self.emb_layer(x)
        x_bilstm,(h, c) = self.encoder(x_emb)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        h = h.view(batch_size, 1, h.shape[-1]*2)
        c = c.view(batch_size, 1, c.shape[-1]*2)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()        

        
        ## decoder
        ans_emb = self.emb_layer(ans)
        out_h, out_c = (h, c)
        first = True
        
        if self.coverage:
            covloss = torch.zeros(1, dtype=torch.float).to(device)
            coverage = [torch.zeros([batch_size,x_bilstm.shape[1]], dtype=torch.float).to(device)]
        
        for w in ans_emb.transpose(0,1):
            w = w.view(w.shape[0], 1, w.shape[1])
            
            ## Scheduled Sampling
            if not first and self.teacher_prob < 1.:                
                useTeacher = (torch.rand(batch_size) < self.teacher_prob).float().view(batch_size, 1, 1).to(device)
                useSample = 1.0 - useTeacher
                
                #get previous output
                ss_distri = self.pro_layer(feature)
                ss_pgen = self.pgen_layer(pgen_feat)
                ss_final_dis = ss_pgen*ss_distri + (1.-ss_pgen)*pointer_prob # + self.epsilon
                
                ans_indices = torch.argmax(ss_final_dis, dim=-1, keepdim=False)
                preword = self.emb_layer(ans_indices).float()
                
                #mixture
                w = useTeacher*w + useSample*preword
    
    
            out, (out_h, out_c) = self.decoder(w, (out_h, out_c))

            if self.coverage:
                attention = torch.bmm(x_bilstm, out.transpose(1, 2)).view(batch_size,x_bilstm.shape[1])
                attention = attention + coverage[-1] * self.cov_weight
                attention = self.attention_softmax(attention)
                covloss += torch.min(attention, coverage[-1]).sum() / batch_size
                coverage.append(coverage[-1]+attention)
            else:
                attention = torch.bmm(x_bilstm, out.transpose(1, 2)).view(batch_size,x_bilstm.shape[1])
                attention = self.attention_softmax(attention)
        
            
            pointer_prob = torch.zeros([batch_size, self.voc_size], dtype=torch.float).to(device)
            pointer_prob = pointer_prob.scatter_add_(dim=1, index=x, src=attention).view(batch_size, 1, self.voc_size)
            
            context_vector = torch.bmm(attention.view(batch_size, 1, self.input_len), x_bilstm)
            
            feature = torch.cat((out, context_vector), -1)
            
            pgen_feat = torch.cat((context_vector, out, w), -1)
            
            if first:
                first = False
                out_seq = feature
                pgen_seq = pgen_feat
                pointer_prob_seq = pointer_prob
            else:
                out_seq = torch.cat((out_seq, feature), 1)
                pgen_seq = torch.cat((pgen_seq, pgen_feat), 1)
                pointer_prob_seq = torch.cat((pointer_prob_seq, pointer_prob), 1)
        
        distri = self.pro_layer(out_seq)
        pgen = self.pgen_layer(pgen_seq)
        
        assert (pgen >= 0).all()
        assert (pointer_prob_seq >= 0).all()
        
        final_dis = pgen*distri + (1.-pgen)*pointer_prob_seq + self.epsilon
        assert (final_dis > 0).all()
        final_dis = torch.log(final_dis)       
        
        if self.coverage:
            return final_dis, covloss / x.shape[1]
        else:
            return final_dis
    
    def inference(self, x, bos):
        if self.training:
            print("[info] Training mode detected. Switched to evaluation mode.")
            self.eval()
        batch_size =x.shape[0]
        device = x.device
        t_bos = torch.tensor([bos]*self.output_len).repeat(batch_size, 1).to(device)
        final_dis = self.forward(x, t_bos)
        ans_indices = torch.argmax(final_dis, dim=-1, keepdim=False)
        return ans_indices