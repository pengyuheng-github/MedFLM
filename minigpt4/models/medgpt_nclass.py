import logging
import random

import torch
import numpy
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("medgpt_nclass")
class MedGPT_NClass(MiniGPTBase):
    """
    medgpt model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/medgpt_vicuna0.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            has_qformer=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=0,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
        )

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, freeze_qformer
            )
            self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        # return the multi-modal embedding in right padding
        emb_lists = []
        prompts_bak = []
        for que in prompts:
            prompts_bak.append("<Img><ImageHere></Img>.Please briefly answer the following medically relevant questions about the above image"+que)
        prompts = prompts_bak
        for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
            pn = each_img_embed.shape[-2]
            p_segs = each_prompt.split('<ImageHere>')
            interleave_emb = []
            for idx, seg in enumerate(p_segs[:-1]): #[:-1]：除了最后一个元素，获取其他所有的元素；
                p_tokens = self.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
            wrapped_emb = torch.cat(interleave_emb, dim=1)
            p_tokens = self.llama_tokenizer(
                p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_embed = self.embed_tokens(p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
            emb_lists.append(wrapped_emb)

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

        max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)

        for i, emb in enumerate(emb_lists):
            length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1
        return wrapped_embs, wrapped_atts

    def preparing_embedding(self, samples):
        ### prepare input tokens
        img_embeds, img_atts = self.encode_img(samples["image"])

        instruction = samples['question']

        cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction) #用promt包裹图片

        ### prepare target tokens
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)

        regress_token_ids = regress_tokens.input_ids
        regress_atts = regress_tokens.attention_mask
        part_targets = regress_token_ids.masked_fill(
            regress_token_ids == self.llama_tokenizer.pad_token_id, -100
        )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets
    def preparing_embedding_withAnsList(self, samples,anslist):
        ### prepare input tokens
        img_embeds, img_atts = self.encode_img(samples["image"])

        instruction = samples['question']

        cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction) #用promt包裹图片

        ### prepare target tokens
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in anslist]

        regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)

        regress_token_ids = regress_tokens.input_ids
        regress_atts = regress_tokens.attention_mask
        part_targets = regress_token_ids.masked_fill(
            regress_token_ids == self.llama_tokenizer.pad_token_id, -100
        )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets
    @torch.no_grad()
    def predict_answers(self,samples):
        question = samples['question']
        answer_list = numpy.array(samples["answer_list"])[:, 1].tolist()
        # answer_list = samples["answer_list"]
        p = "<Img><ImageHere></Img>.Please briefly answer the following medically relevant questions about the above image";
        question_b = [p+que for que in question]
        output_text = []
        for i in range(len(question)):
            output = self.multi_select(samples['image'][i].unsqueeze(0),[question_b[i]],answer_list)[0]
            output_text.append(answer_list[output[0]])
        return output_text

    @torch.no_grad()
    def multi_select(self, samples, candidates, n_segments=1):
        image = samples["image"]
        bs = image.size(0)
        # prompt包裹的图片         answer的embeds
        cond_embeds, cond_atts, regress_embeds_list, regress_atts_list, part_targets_list = self.preparing_embedding_withAnsList(samples,candidates)
        #拼接输入和答案
        for regress_embeds in regress_embeds_list:
            inputs_embeds, attention_mask, input_lens = self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        return ""

    def forward(self, samples, reduction='mean'):
        # prepare the embedding to condition and the embedding to regress
        #prompt包裹的图片         answer的embeds
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)
        # print("cond_embeds",cond_embeds.shape,cond_embeds) #torch.Size([2, 58, 4096])
        # print("cond_atts",cond_atts.shape,cond_atts) # torch.Size([2, 58]
        # print("regress_embeds",regress_embeds.shape,regress_embeds) #torch.Size([2, 9, 4096])
        # print("regress_atts",regress_atts.shape,regress_atts) # torch.Size([2, 9])
        # print("part_targets",part_targets.shape,part_targets) #torch.Size([2, 9])

        # concat the embedding to condition and the embedding to regress
        #拼接输入和答案
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        # BOS begining of sequence，代表序列开始。
        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        #print("inputs_embeds",inputs_embeds.shape,inputs_embeds) #inputs_embeds torch.Size([b, 62, 4096])
        #print("attention_mask",attention_mask.shape,attention_mask) #torch.Size([b, 62])
        #print("targets",targets.shape,targets) #torch.Size([b, 62])
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )

        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model


