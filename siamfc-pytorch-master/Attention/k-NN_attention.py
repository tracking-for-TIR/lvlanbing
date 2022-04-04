# paper: https://arxiv.org/pdf/2106.00515.pdf

import torch
import torch.nn as nn
class kNN_Attention (nn.Module):
 def __init__ ( self , dim , num_heads =8 , qkv_bias = False , qk_scale = None , attn_drop =0. , proj_drop=0. , topk =100) :
   super () . __init__ ()
   self.num_heads = num_heads
   head_dim = dim // num_heads
   self . scale = qk_scale or head_dim ** -0.5
   self . topk = topk

   self . qkv = nn . Linear ( dim , dim *3 , bias = qkv_bias )
   self . attn_drop = nn . Dropout ( attn_drop )
   self . proj = nn . Linear ( dim , dim )
   self . proj_drop = nn . Dropout ( proj_drop )

 def forward ( self ,x):
   B ,N , C=x. shape
   qkv = self . qkv ( x). reshape (B ,N ,3 , self . num_heads ,C // self . num_heads ). permute(2 ,0 ,3 ,1 ,4)
   q,k , v= qkv[0], qkv[1], qkv[2] #B,H,N,C
   attn =(q@k . transpose (-2, -1)) * self . scale #B,H,N,N
   # the core code block
   mask = torch.zeros(B, self.num_heads ,N ,N , device =x . device , requires_grad = False )
   index = torch . topk ( attn ,k= self . topk , dim = -1 , largest = True ) [1]
   mask . scatter_ ( -1 , index ,1.)
   attn = torch . where ( mask >0 , attn , torch . full_like ( attn , float ('-inf' )))
   # end of the core code block
   attn = torch . softmax ( attn , dim = -1)
   attn = self . attn_drop ( attn )
   x =( attn@v ). transpose (1 ,2) . reshape (B ,N , C)
   x= self . proj (x)
   x= self . proj_drop (x)

   return x
