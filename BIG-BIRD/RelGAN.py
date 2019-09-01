import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import torch.autograd as autograd
from RelationalMemory import *
from Transformer import *


class BigBird():
    #generator is translator here
    def __init__(self, generator, discriminator, reconstructor, dictionary, gamma = 0.99, clip_value = 0.1, lr_G = 5e-5, lr_D = 5e-5, lr_R = 1e-4, LAMBDA = 10,  TEMP_END = 0.5, vq_coef =0.8, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(BigBird, self).__init__()
        
        self.device = device
        
        self.dictionary = dictionary
        
        self.generator = generator.to(self.device)
        self.reconstructor = reconstructor.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        
        self.optimizer_R = torch.optim.Adam(list(self.generator.parameters()) + list(self.reconstructor.parameters()), lr=lr_R)
        
        #normal WGAN
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr_G)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr_D)
        
        #WGAN GP
        
        #self.LAMBDA = LAMBDA # Gradient penalty lambda hyperparameter
        #self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G,  betas=(0.0, 0.9))
        #self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D,  betas=(0.0, 0.9))
               
        self.clip_value = clip_value
        self.TEMP_END = TEMP_END
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.lr_R = lr_R
        
        self.total_steps = 0

        self.vq_coef = 0.8
        
        self.epoch = 0
        
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        #print real_data.size()
        BATCH_SIZE = real_data.shape[0]
        dim_1 = real_data.shape[1]
        dim_2 = real_data.shape[2]
        alpha = torch.rand(BATCH_SIZE, dim_1)
        alpha = alpha.view(-1,1).expand(dim_1 * BATCH_SIZE, dim_2).view(BATCH_SIZE, dim_1, dim_2)
        alpha = alpha.to(self.device)
        
        #print(real_data.shape) #[BATCH_SIZE, 19, vocab_sz]
        #print(fake_data.shape) #[BATCH_SIZE, 19, vocab_sz]
        interpolates_data = ( alpha * real_data.float() + ((1 - alpha) * fake_data.float()) )

        interpolates = interpolates_data.to(self.device)
        
        #interpolates = netD.disguised_embed(interpolates_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        
        src_mask = (interpolates_data.argmax(-1) != netD.padding_index).type_as(interpolates_data).unsqueeze(-2)
        disc_interpolates = netD.transformer_encoder( interpolates, src_mask )

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty
    
    def _to_one_hot(self, y, n_dims):

        scatter_dim = len(y.size())

        y_tensor = y.to(self.device).long().view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).to(self.device)

        return zeros.scatter(scatter_dim, y_tensor, 1) 
    
    def train_D(self, fake_datas, real_datas):
        ## train discriminator       
#         print("real")
#         print(real_datas[:10])
        real_score = torch.mean(self.discriminator(real_datas))
#         print("fake")
#         print(fake_datas[:10])
        fake_score = torch.mean(self.discriminator(fake_datas))

        batch_d_loss = -real_score + fake_score #+ self.calc_gradient_penalty(self.discriminator, real_datas, fake_datas)
        
        return batch_d_loss, real_score.item(), fake_score.item()
    
    def train_G(self, fake_datas): 

        self.optimizer_G.zero_grad()
          
        batch_g_loss = -torch.mean(self.discriminator(fake_datas))
        
        batch_g_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        return batch_g_loss.item()      
    
    def indicies2string(self, indices):
        inv_map = {v: k for k, v in self.dictionary.items()}
        return ' '.join([inv_map[i.item()] for i in indices])
        
    def train(self):
        self.generator.train()
        self.reconstructor.train()
        self.discriminator.train()
        
    def eval(self):
        self.generator.eval()
        self.reconstructor.eval()
        self.discriminator.eval()
    
    def load(self, load_path):
        print('load Bird from', load_path)
        loader = torch.load(load_path)
        self.generator.load_state_dict(loader['generator'])
        self.discriminator.load_state_dict(loader['discriminator'])
        self.reconstructor.load_state_dict(loader['reconstructor'])
        self.total_steps = loader['total_steps']
        self.epoch = loader['epoch']
        self.gumbel_temperature = loader['gumbel_temperature']
    
    def save(self, save_path):
        print('lay egg to ./Nest ... save as', save_path)

        torch.save({'generator':self.generator.state_dict(), 
                    'reconstructor':self.reconstructor.state_dict(), 
                    'discriminator':self.discriminator.state_dict(),
                    'total_steps':self.total_steps,
                    'epoch':self.epoch,
                    'gumbel_temperature':self.gumbel_temperature
                    },save_path)
    
    def eval_iter(self, src, src_mask, max_len, real_data, ct, verbose = 1):
        with torch.no_grad():
            summary_sample, summary_log_values, critic_values, summary_log_probs, gumbel_one_hot = self.generator(src, src_mask, max_len, self.dictionary['[CLS]'], mode = 'sample')
            summary_mask = (summary_sample != self.dictionary['[SEP]']).type_as(summary_sample).unsqueeze(-2)
            rewards, acc, out = self.reconstructor(gumbel_one_hot, summary_mask, src.shape[1], self.dictionary['[CLS]'], src)
            if verbose == 1 and ct % 100 == 0:
                print("origin:")
                print(self.indicies2string(src[0]))
                print("summary:")
                print(self.indicies2string(summary_sample[0]))
                print("real summary:")
                print(self.indicies2string(real_data[0]))
                print("reconsturct out:")
                print(self.indicies2string(out[0]))
                print("")
                
            return acc, rewards.mean().item()   
    
    def pretrainGAN_run_iter(self, src, src_mask, max_len, real_data,  D_iters = 5, D_toggle = 'On', verbose = 1):
        
        batch_size = src.shape[0]
        memory = self.generator.initial_state(batch_size, trainable=True).to(self.device)
        self.gumbel_temperature = max(self.TEMP_END, math.exp(-1e-4*self.total_steps))
        summary_sample, summary_log_values, summary_probs, gumbel_one_hot = self.generator(src, max_len, memory, self.dictionary['[CLS]'], temperature = self.gumbel_temperature)
        
        batch_G_loss = 0 

            
        NNcriterion = nn.NLLLoss().to(self.device)
        batch_G_loss = NNcriterion(summary_probs.log().contiguous().view(batch_size * max_len, -1), real_data.contiguous().view(-1))
        
        self.optimizer_G.zero_grad()
        batch_G_loss.backward()
        self.optimizer_G.step()
        
        self.total_steps += 1
        
        if self.total_steps % 500 == 0:
            if not os.path.exists("./Nest"):
                os.makedirs("./Nest")
            self.save("./Nest/Pretrain_RelGAN")

        if verbose == 1 and self.total_steps % 1000 == 0:
                print("origin:")
                print(self.indicies2string(src[0]))
                print("summary:")
                print(self.indicies2string(summary_sample[0]))
                print("real summary:")
                print(self.indicies2string(real_data[0]))
                print("")
        distrib = summary_probs[0,0, :100].cpu().detach().numpy()
        one_hot_out = gumbel_one_hot[0,0, :100].cpu().detach().numpy()
        return [batch_G_loss, 0], [0], [0, 0, 0], [self.indicies2string(src[0]), self.indicies2string(summary_sample[0]), 0], distrib, one_hot_out
        
    def run_iter(self, src, src_mask, max_len, real_data,  D_iters = 5, D_toggle = 'On', verbose = 1, writer = None):
        #summary_logits have some problem

        
        #summary = self.generator(src, src_mask, max_len, self.dictionary['[CLS]'])
        batch_size = src.shape[0]
        memory = self.generator.initial_state(batch_size, trainable=True).to(self.device)
        self.gumbel_temperature = max(self.TEMP_END, math.exp(-1e-4*self.total_steps))
        summary_sample, summary_log_values, summary_probs, gumbel_one_hot = self.generator(src, max_len, memory, self.dictionary['[CLS]'], temperature = self.gumbel_temperature)
        
        batch_D_loss = 0
        if(D_toggle == 'On'):
            for i in range(D_iters):
                self.optimizer_D.zero_grad()
                
                batch_d_loss, real_score, fake_score = self.train_D(gumbel_one_hot, self._to_one_hot(real_data, len(self.dictionary)))
                batch_D_loss += batch_d_loss
                batch_d_loss.backward(retain_graph=True);

                #Clip critic weights
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)
        
            self.optimizer_D.step();
        
        batch_D_loss = batch_D_loss.item()/D_iters
        
        batch_G_loss = 0 
        if(D_toggle == 'On'):
            #print(gumbel_one_hot.shape)
            batch_G_loss = self.train_G(gumbel_one_hot)          
        
        self.gumbel_temperature = max(self.TEMP_END, math.exp(-1e-4*self.total_steps))
        
        
        memory = self.reconstructor.initial_state(batch_size, trainable=True).to(self.device)
        CE_loss, acc, out = self.reconstructor.reconstruct_forward(gumbel_one_hot, src, memory, self.dictionary['[CLS]'])

        rec_loss = CE_loss #+ self.vq_coef * vq_loss + 0.25 * self.vq_coef * commit_loss
        
        self.optimizer_R.zero_grad()
        rec_loss.backward()
        nn.utils.clip_grad_norm_(list(self.generator.parameters()) + list(self.reconstructor.parameters()), 0.1)
        self.optimizer_R.step()
        
        
        self.total_steps += 1
        
        if self.total_steps % 500 == 0:
            if not os.path.exists("./Nest"):
                os.makedirs("./Nest")
            self.save("./Nest/DoubleRelationMEM_GAN")

            #for i in range(5):
                #plt.plot(range(1000),summary_probs.cpu().detach().numpy()[0,i,:1000] )
            #    wandb.log({"prob {}".format(i): wandb.Histogram(summary_probs.cpu().detach().numpy()[0,i,:1000])},step=step)
        
        if verbose == 1 and self.total_steps % 100 == 0:
            print("origin:")
            print(self.indicies2string(src[0]))
            print("summary:")
            print(self.indicies2string(summary_sample[0]))
            print("real summary:")
            print(self.indicies2string(real_data[0]))
            print("reconsturct out:")
            print(self.indicies2string(out[0]))
#             print("sentiment:",label[0].item())
#             print("y:",sentiment_label[0].item())
#            print("reward:",rewards[0].item())
            
            print("")
        
#         for name, param in self.generator.named_parameters():
#             writer.add_histogram(name, param.clone().cpu().data.numpy(), self.total_steps)
            
#         for name, param in self.reconstructor.named_parameters():
#             writer.add_histogram(name, param.clone().cpu().data.numpy(), self.total_steps)    
        distrib = summary_probs.cpu().detach().numpy()[0,0, :100]
        one_hot_out = gumbel_one_hot.cpu().detach().numpy()[0,0, :100]
        return [batch_G_loss, batch_D_loss], [CE_loss.item()], [real_score, fake_score, acc], [self.indicies2string(src[0]), self.indicies2string(summary_sample[0]), self.indicies2string(out[0])], distrib, one_hot_out

class LSTMEncoder(nn.Module):    
    def __init__(self, vocab_sz, hidden_dim, padding_index):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_sz, hidden_dim)
        self.rnn_cell = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.padding_index = padding_index
        self.outsize = hidden_dim*2
        
    def forward(self, x):
        #src_mask = (x != self.padding_index).type_as(x).unsqueeze(-2)
        out, (h,c) = self.rnn_cell( self.src_embed(x))
        return out


    

# class LSTM_Gumbel_Encoder_Decoder(nn.Module):
#     def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, device, eps=1e-8, num_layers = 2):
#         super().__init__()
        
#         self.hidden_dim = hidden_dim
#         self.emb_dim = emb_dim
#         #self.input_len = input_len
#         #self.output_len = output_len
#         #self.voc_size = voc_size
#         #self.teacher_prob = 1.
#         #self.epsilon = eps
        
#         self.emb_layer = nn.Embedding(voc_size, emb_dim)
#         self.num_layers = num_layers
        
#         self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=num_layers, batch_first=True)
        
#         self.device = device
        
#         self.attention_softmax = nn.Softmax(dim=1)
        
# #         self.pro_layer = nn.Sequential(
# #             nn.Linear(hidden_dim*4, voc_size, bias=True)
# #         )
#         self.adaptive_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(hidden_dim*4, voc_size, [100, 1000, 10000], div_value=4.0, head_bias=False)

        
#     def forward(self, x, src_mask, max_len, start_symbol, mode = 'argmax', temp = 2.0):
        
#         batch_size = x.shape[0]
#         input_len = x.shape[1]
#         device = x.device
        
#         # encoder
#         x_emb = self.emb_layer(x)
#         memory, (h, c) = self.encoder(x_emb)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()
#         h = h.view(batch_size, self.num_layers, h.shape[-1]*2)
#         c = c.view(batch_size, self.num_layers, c.shape[-1]*2)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()        

        
#         ## decoder
#         out_h, out_c = (h, c)
#         ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(x.data)
        
#         values = []
#         all_probs = []
#         gumbel_one_hots = []
        
#         for i in range(max_len-1):
#             ans_emb = self.emb_layer(ys[:,-1]).view(batch_size, 1, self.emb_dim)
#             out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c))
            
#             attention = torch.bmm(memory, out.transpose(1, 2)).view(batch_size, input_len)
#             attention = self.attention_softmax(attention)            
            
#             context_vector = torch.bmm(attention.view(batch_size, 1, input_len), memory)
            
#             logits = torch.cat((out, context_vector), -1).view(batch_size, -1)
            
#             one_hot, next_words, value, prob = self.gumbel_softmax(logits, temp)
            
# #             print(feature.shape)
# #             print(one_hot.shape)
# #             print(next_words.shape)
# #             print(values.shape)
# #             print(log_probs.shape)
# #             input("")
                
#             ys = torch.cat((ys, next_words.view(batch_size, 1)), dim=1)
                 
#             values.append(value)
#             all_probs.append(prob)
#             gumbel_one_hots.append(one_hot)
            
#         values = torch.stack(values,1)
#         all_probs = torch.stack(all_probs,1)
#         gumbel_one_hots = torch.stack(gumbel_one_hots, 1)
        
#         return ys, values, all_probs, gumbel_one_hots  
    
#     def sample_gumbel(self, shape, eps=1e-20):
#         U = torch.rand(shape).to(self.device)
#         return -Variable(torch.log(-torch.log(U + eps) + eps))

#     def gumbel_softmax_sample(self, logits, temperature):
#         y = logits + self.sample_gumbel(logits.size())
        
#         #the formula should be prob not logprob, I guess it still works
#         return self.adaptive_softmax.log_prob(logits).exp()
#         #return F.softmax(y / temperature, dim=-1)

#     def gumbel_softmax(self, logits, temperature):
#         """
#         ST-gumple-softmax
#         input: [*, n_class]
#         return: flatten --> [*, n_class] an one-hot vector
#         """
#         y = self.gumbel_softmax_sample(logits, temperature)
#         shape = y.size()
#         values, ind = y.max(dim=-1)
#         y_hard = torch.zeros_like(y).view(-1, shape[-1])
#         y_hard.scatter_(1, ind.view(-1, 1), 1)
#         y_hard = y_hard.view(*shape)
#         y_hard = (y_hard - y).detach() + y
#         return y_hard.view(logits.shape[0], -1), ind, values, y    

# class LSTM_Normal_Encoder_Decoder(nn.Module):
#     def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, pad_index, device, eps=1e-8, num_layers = 2):
#         super().__init__()
        
#         self.hidden_dim = hidden_dim
#         self.emb_dim = emb_dim
#         self.device = device
          
#         #self.input_len = input_len
#         #self.output_len = output_len
#         #self.voc_size = voc_size
#         #self.teacher_prob = 1.
#         #self.epsilon = eps
        
#         self.num_layers = num_layers
        
#         #self.emb_layer = nn.Embedding(voc_size, emb_dim)
#         self.disguise_embed = nn.Linear(voc_size, emb_dim)
#         self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=num_layers, batch_first=True)
        
#         self.attention_softmax = nn.Softmax(dim=1)
#         self.vocab_sz = voc_size
        

#         self.criterion = torch.nn.AdaptiveLogSoftmaxWithLoss(hidden_dim*4, voc_size, [1000, 5000, 20000], div_value=4.0, head_bias=False)
        
#     def forward(self, x, src_mask, max_len, start_symbol, y, mode = 'argmax', temp = 2.0):
        
#         batch_size = x.shape[0]
#         input_len = x.shape[1]
#         device = x.device
        
#         # encoder
#         x_emb = self.disguise_embed(x)
#         memory, (h, c) = self.encoder(x_emb)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()
#         h = h.view(batch_size, self.num_layers, h.shape[-1]*2)
#         c = c.view(batch_size, self.num_layers, c.shape[-1]*2)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()        

        
#         ## decoder
#         out_h, out_c = (h, c)
        
#         logits = []
       
        
#         for i in range(max_len):
#             ans_emb = self.disguise_embed(self._to_one_hot(y[:,i], self.vocab_sz)).view(batch_size, 1, self.emb_dim)
#             out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c))
#             attention = torch.bmm(memory, out.transpose(1, 2)).view(batch_size, input_len)
#             attention = self.attention_softmax(attention)            
            
#             context_vector = torch.bmm(attention.view(batch_size, 1, input_len), memory)
            
#             logit = torch.cat((out, context_vector), -1).view(batch_size, -1)
            
            
# #             if mode == 'argmax':
# #                 values, next_words = torch.max(log_probs, dim=-1, keepdim=True)
# #             if mode == 'sample':
# #                 m = torch.distributions.Categorical(logits=log_probs)
# #                 next_words = m.sample()
# #                 values = m.log_prob(next_words)
             
#             logits.append(logit)
                
#         logits = torch.stack(logits, 1)
        

        
        
#         _ ,loss = self.criterion(logits[:,:-1].contiguous().view(batch_size * (max_len - 1), -1), y[:,1:].contiguous().view(batch_size * (max_len-1)))
        
#         #y from one to get rid of [CLS]
#         log_argmaxs = self.criterion.predict(logits[:,:-1].contiguous().view(batch_size * (max_len - 1), -1)).view(batch_size, max_len-1)
#         acc = ( log_argmaxs== y[:,1:]).float().mean()

        
#         return loss, acc, log_argmaxs   
    
#     def _to_one_hot(self, y, n_dims):

#         scatter_dim = len(y.size())

#         y_tensor = y.to(self.device).long().view(*y.size(), -1)
#         zeros = torch.zeros(*y.size(), n_dims).to(self.device)

#         return zeros.scatter(scatter_dim, y_tensor, 1)
    

class Discriminator(nn.Module):
    def __init__(self, transformer_encoder, hidden_dim, vocab_sz, padding_index):
        super(Discriminator, self).__init__()
        
        self.padding_index = padding_index
        
        self.disguise_embed = nn.Linear(vocab_sz, hidden_dim)  
        self.transformer_encoder = transformer_encoder
        self.linear = nn.Linear(self.transformer_encoder.layers[-1].size, 1)
        #self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        src_mask = (x.argmax(-1) != self.padding_index).type_as(x).unsqueeze(-2)
        x = self.transformer_encoder(self.disguise_embed(x), src_mask)
        score = self.linear(x)
        return score
        
       
        
        