import torch
from torch import  mode, nn
import torch.nn.functional as F

class CO_Attention(nn.Module):
    def  __init__(self,in_dim, co_attention_dim):
        super(CO_Attention, self).__init__()
        # self.linear_r = nn.Bilinear(in_dim, in_dim, co_attention_dim)
        # self.linear_p = nn.Bilinear(in_dim, in_dim, co_attention_dim)
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,co_attention_dim,kernel_size=3,padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.gate1 = nn.Sequential(self.conv1, self.relu, self.pool, self.conv2, self.relu)

        self.conv3 = nn.Conv2d(in_dim,64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(64,co_attention_dim,kernel_size=3,padding=1)
        self.gate2 = nn.Sequential(self.conv3, self.relu, self.pool, self.conv4, self.relu)

        self.bilinear = torch.nn.Bilinear(co_attention_dim,co_attention_dim,1)
        self.conv = nn.Conv2d(1,1,kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_embed, prev_embed, refer_embed, refer_label,ori_size=None,use_float16=True):
        """
        Args:
            query_embed:[height,width,embed_dim]
            prev_embed:[height,width,embed_dim]
            ref_embed:[height,width,embed_dim]
            ref_label:[height,width,obj_nums]
        Return:
            x:[h,w,obj_num,1]
        """
        h,w,_ =  query_embed.size()
        obj_nums = refer_label.size(2)
        query_embed = query_embed.permute(2,0,1)
        prev_embed = prev_embed.permute(2,1,0)
        refer_embed = refer_embed.permute(2,1,0)
        refer_label = refer_label.permute(2,0,1)
        # all_ref_fg = torch.sum(refer_label_flat, dim=1, keepdim=True) > 0.9
        # refer_label_flat = torch.masked_select(refer_label_flat, 
        #     all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
        
        r_attention = self.leak_relu(torch.bmm(refer_embed, query_embed)).unsqueeze(0)
        p_attention = self.leak_relu(torch.bmm(prev_embed, query_embed)).unsqueeze(0)

        r_attention = self.gate1(r_attention).squeeze(0)
        p_attention = self.gate2(p_attention).squeeze(0)

        attention_h  = r_attention.size(1)
        attention_w = r_attention.size(2)
        r_attention = r_attention.reshape(attention_h*attention_w,-1)
        p_attention = p_attention.reshape(attention_h*attention_w,-1)
        attention = self.bilinear(r_attention,p_attention).reshape(-1,1,attention_h,attention_w)
        if(attention.size(2)!=64 or attention.size(3)!=64):
            attention = F.interpolate(attention, size=[64,64], mode='bilinear',align_corners=True)
        attention = self.conv(attention)

        attention = F.interpolate(attention, size=(h,w), 
                mode='bilinear', align_corners=True).squeeze(0)
        x = torch.mul(attention,refer_label).unsqueeze(-1).permute(1,2, 0, 3)
        attention = attention.permute(1,2,0)
        x = self.sigmoid(x)

        if ori_size is not None:
            x = x.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            x = F.interpolate(x, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            x = x.float()
        return x
    
if __name__=='__main__':
    query_embed = torch.rand(117,117,100)
    prev_embed = torch.rand(117,117,100)
    refer_embed = torch.rand(117,117,100)
    refer_label = torch.rand(117,117,3)
    co_attention = CO_Attention(100,32)
    out, attention = co_attention(query_embed,prev_embed,refer_embed,refer_label)
    assert out.shape == (117,117,3,1)
    assert attention.shape ==(117,117,1)
    print(attention.shape)
    # print(co_attention)
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='co_attention') as w:
    #     w.add_graph(co_attention,(query_embed,prev_embed,refer_embed,refer_label))
    #     print('-------')
    # print('-----------------')
